import csv
import math

import numpy as np
import pytest
import torch

from cleanrl_utils.buffers import ReplayBufferSamples

from last_nine_rl.config import SACConfig
from last_nine_rl.envs import make_env
from last_nine_rl.replay import InstrumentedReplayBuffer, InstrumentedReplaySamples, SACNReplaySamples
from last_nine_rl.sac import SACAgent
from last_nine_rl.simba_v2 import SimbaHyperDense, project_simba_weights_to_unit_norm


def test_prioritized_replay_applies_is_correction_and_emits_bellman_priorities():
    common = dict(
        device="cpu",
        actor_update_start_step=999,
        update_diagnostics=False,
    )
    torch.manual_seed(812)
    uniform_agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        SACConfig(**common),
        device="cpu",
    )
    torch.manual_seed(812)
    priority_agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        SACConfig(
            **common,
            replay_priority_mode="bellman_residual",
            replay_priority_clip=0.25,
        ),
        device="cpu",
    )
    batch_size = 6
    batch = InstrumentedReplaySamples(
        observations=torch.linspace(-1.0, 1.0, batch_size * 3).reshape(batch_size, 3),
        actions=torch.linspace(-0.5, 0.5, batch_size).reshape(batch_size, 1),
        next_observations=torch.linspace(-0.9, 1.1, batch_size * 3).reshape(batch_size, 3),
        dones=torch.zeros(batch_size, 1),
        rewards=-torch.linspace(0.0, 2.0, batch_size).reshape(batch_size, 1),
        replay_indices=torch.arange(batch_size).reshape(-1, 1),
        sampling_probabilities=torch.full((batch_size, 1), 1.0 / batch_size),
        importance_weights=torch.full((batch_size, 1), 0.5),
    )

    torch.manual_seed(1234)
    uniform_metrics = uniform_agent.update(batch, update_step=1)
    torch.manual_seed(1234)
    priority_metrics = priority_agent.update(batch, update_step=1)

    assert priority_metrics["q1_loss"] == pytest.approx(0.5 * uniform_metrics["q1_loss"], rel=1e-6)
    assert priority_metrics["q2_loss"] == pytest.approx(0.5 * uniform_metrics["q2_loss"], rel=1e-6)
    assert priority_metrics["replay_priority_importance_weight_mean"] == pytest.approx(0.5)
    assert priority_metrics["replay_priority_importance_correction_applied_to_critic"] == 1.0
    assert priority_agent.last_replay_priority_values is not None
    assert priority_agent.last_replay_priority_values.shape == (batch_size,)
    assert np.all(priority_agent.last_replay_priority_values <= 0.25)
    assert uniform_agent.last_replay_priority_values is None


@pytest.mark.parametrize(
    "simba_kwargs",
    [
        {},
        {"simba_backbone": True, "simba_actor_hidden_dim": 16, "simba_critic_hidden_dim": 16},
        {
            "simba_backbone": True,
            "simba_actor_hidden_dim": 16,
            "simba_critic_hidden_dim": 16,
            "simba_weight_projection": True,
        },
        {
            "simba_backbone": True,
            "simba_actor_hidden_dim": 16,
            "simba_critic_hidden_dim": 16,
            "simba_weight_projection": True,
            "simba_distributional_critic": True,
            "simba_critic_num_bins": 11,
            "simba_reward_scaling": True,
        },
    ],
)
def test_sac_action_bounds_and_single_update(simba_kwargs):
    env = make_env("Pendulum-v1", seed=0)
    try:
        cfg = SACConfig(
            buffer_size=32,
            learning_starts=4,
            batch_size=4,
            device="cpu",
            **simba_kwargs,
        )
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        agent = SACAgent(obs_dim, env.action_space.low, env.action_space.high, cfg, device="cpu")
        replay = InstrumentedReplayBuffer(
            32,
            env.observation_space,
            env.action_space,
            device="cpu",
            n_envs=1,
            handle_timeout_termination=False,
        )

        obs, _ = env.reset(seed=0)
        agent.observe(obs)
        action = agent.act(obs, deterministic=False)
        assert action.shape == env.action_space.shape
        assert np.all(action <= env.action_space.high + 1e-5)
        assert np.all(action >= env.action_space.low - 1e-5)

        for step in range(8):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.observe_reward(float(reward), bool(terminated or truncated))
            replay.add(
                obs.reshape(1, obs_dim),
                next_obs.reshape(1, obs_dim),
                action.reshape(1, action_dim),
                np.asarray([reward], dtype=np.float32),
                np.asarray([terminated]),
                [{}],
                step=step,
                episode_id=0,
            )
            obs = next_obs
            agent.observe(obs)
            if terminated or truncated:
                obs, _ = env.reset()
                agent.observe(obs)

        batch = replay.sample(batch_size=4)
        metrics = agent.update(batch, update_step=2)
        assert metrics["q_loss"] >= 0.0
        assert "actor_loss" in metrics
        assert metrics["alpha"] > 0.0
        assert metrics["q_update_norm_ratio"] > 0.0
        silent_metrics = agent.update(batch, update_step=3, collect_metrics=False)
        assert "q_loss" not in silent_metrics
    finally:
        env.close()


def test_actor_q_aggregation_switch_schedule():
    cfg = SACConfig(
        device="cpu",
        actor_q_aggregation="min",
        actor_q_aggregation_late="mean",
        actor_q_aggregation_switch_step=10,
    )
    agent = SACAgent(3, np.asarray([-2.0], dtype=np.float32), np.asarray([2.0], dtype=np.float32), cfg, device="cpu")

    assert agent._actor_q_aggregation_for_step(9) == "min"
    assert agent._actor_q_aggregation_for_step(10) == "mean"


@pytest.mark.parametrize(("actor_updates_per_trigger", "expected_actor_steps"), [(0, 4), (1, 1)])
def test_actor_updates_per_trigger_controls_actor_utd(
    monkeypatch,
    actor_updates_per_trigger,
    expected_actor_steps,
):
    env = make_env("Pendulum-v1", seed=0)
    try:
        cfg = SACConfig(
            buffer_size=32,
            batch_size=4,
            device="cpu",
            update_diagnostics=False,
            policy_frequency=4,
            actor_updates_per_trigger=actor_updates_per_trigger,
        )
        agent = SACAgent(3, env.action_space.low, env.action_space.high, cfg, device="cpu")
        replay = InstrumentedReplayBuffer(
            32,
            env.observation_space,
            env.action_space,
            device="cpu",
            n_envs=1,
            handle_timeout_termination=False,
        )
        obs, _ = env.reset(seed=0)
        for step in range(8):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            replay.add(
                obs.reshape(1, 3),
                next_obs.reshape(1, 3),
                action.reshape(1, 1),
                np.asarray([reward], dtype=np.float32),
                np.asarray([terminated]),
                [{}],
                step=step,
                episode_id=0,
            )
            obs = next_obs

        actor_steps = 0
        original_step = agent.actor_optimizer.step

        def counted_step(*args, **kwargs):
            nonlocal actor_steps
            actor_steps += 1
            return original_step(*args, **kwargs)

        monkeypatch.setattr(agent.actor_optimizer, "step", counted_step)
        metrics = agent.update(replay.sample(batch_size=4), update_step=4)

        assert actor_steps == expected_actor_steps
        assert metrics["actor_updates_per_trigger"] == pytest.approx(float(expected_actor_steps))
        assert metrics["actor_updates_executed"] == pytest.approx(float(expected_actor_steps))
    finally:
        env.close()


def test_critic_search_actor_loss_uses_global_clipped_q_improvement():
    class QuadraticQ(torch.nn.Module):
        def forward(self, observations, actions):
            del observations
            return -(actions[:, :1] - 1.0).pow(2)

    cfg = SACConfig(
        device="cpu",
        critic_search_actor_weight=0.1,
        critic_search_num_actions=41,
        critic_search_margin=0.1,
    )
    agent = SACAgent(3, np.asarray([-2.0], dtype=np.float32), np.asarray([2.0], dtype=np.float32), cfg, "cpu")
    agent.q_networks = torch.nn.ModuleList([QuadraticQ(), QuadraticQ()])
    observations = torch.zeros((4, 3), dtype=torch.float32)
    actor_actions = torch.zeros((4, 1), dtype=torch.float32, requires_grad=True)

    loss, metrics = agent._critic_search_actor_loss(observations, actor_actions)
    loss.backward()

    assert float(loss.detach()) == pytest.approx(0.25)
    assert metrics["critic_search_advantage_mean"] == pytest.approx(1.0)
    assert metrics["critic_search_selected_fraction"] == pytest.approx(1.0)
    assert metrics["critic_search_action_abs_error_mean"] == pytest.approx(1.0)
    assert actor_actions.grad is not None
    assert torch.all(actor_actions.grad < 0.0)


def test_critic_search_log_prob_loss_is_finite_for_saturated_target():
    class IncreasingQ(torch.nn.Module):
        def forward(self, observations, actions):
            del observations
            return actions[:, :1]

    cfg = SACConfig(
        device="cpu",
        critic_search_actor_weight=0.1,
        critic_search_num_actions=41,
        critic_search_margin=0.0,
        critic_search_actor_loss_type="log_prob",
    )
    agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        cfg,
        "cpu",
    )
    agent.q_networks = torch.nn.ModuleList([IncreasingQ(), IncreasingQ()])
    observations = torch.zeros((4, 3), dtype=torch.float32)
    _sampled, _log_prob, actor_actions = agent.actor.get_action(observations)

    loss, metrics = agent._critic_search_actor_loss(observations, actor_actions)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["critic_search_loss_type_is_log_prob"] == pytest.approx(1.0)
    assert metrics["critic_search_selected_fraction"] == pytest.approx(1.0)
    assert any(parameter.grad is not None for parameter in agent.actor.parameters())


def test_critic_search_actor_loss_online_target_unanimity_allows_target_veto():
    class PreferOneQ(torch.nn.Module):
        def forward(self, observations, actions):
            del observations
            return -(actions[:, :1] - 1.0).pow(2)

    class PreferActorQ(torch.nn.Module):
        def forward(self, observations, actions):
            del observations
            return -actions[:, :1].pow(2)

    observations = torch.zeros((4, 3), dtype=torch.float32)
    actor_actions = torch.zeros((4, 1), dtype=torch.float32, requires_grad=True)

    online_agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        SACConfig(
            device="cpu",
            critic_search_filter_mode="unanimous_advantage",
            critic_search_num_actions=41,
        ),
        "cpu",
    )
    online_agent.q_networks = torch.nn.ModuleList([PreferOneQ(), PreferOneQ()])
    online_agent.q_target_networks = torch.nn.ModuleList(
        [PreferOneQ(), PreferActorQ()]
    )
    online_loss, online_metrics = online_agent._critic_search_actor_loss(
        observations, actor_actions
    )

    veto_agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        SACConfig(
            device="cpu",
            critic_search_filter_mode="online_target_unanimous_advantage",
            critic_search_num_actions=41,
        ),
        "cpu",
    )
    veto_agent.q_networks = torch.nn.ModuleList([PreferOneQ(), PreferOneQ()])
    veto_agent.q_target_networks = torch.nn.ModuleList(
        [PreferOneQ(), PreferActorQ()]
    )
    veto_loss, veto_metrics = veto_agent._critic_search_actor_loss(
        observations, actor_actions
    )

    assert float(online_loss.detach()) > 0.0
    assert online_metrics["critic_search_selected_fraction"] == pytest.approx(1.0)
    assert float(veto_loss.detach()) == pytest.approx(0.0)
    assert veto_metrics["critic_search_selected_fraction"] == pytest.approx(0.0)
    assert veto_metrics["critic_search_filter_is_online_target"] == pytest.approx(1.0)


def test_critic_search_action_selection_respects_q_margin(monkeypatch):
    class QuadraticQ(torch.nn.Module):
        def forward(self, observations, actions):
            del observations
            return -(actions[:, :1] - 1.0).pow(2)

    cfg = SACConfig(device="cpu")
    agent = SACAgent(3, np.asarray([-2.0], dtype=np.float32), np.asarray([2.0], dtype=np.float32), cfg, "cpu")
    agent.q_networks = torch.nn.ModuleList([QuadraticQ(), QuadraticQ()])

    def zero_actor(observations):
        batch_size = observations.shape[0]
        actions = torch.zeros((batch_size, 1), dtype=observations.dtype, device=observations.device)
        log_prob = torch.zeros((batch_size, 1), dtype=observations.dtype, device=observations.device)
        return actions, log_prob, actions

    monkeypatch.setattr(agent.actor, "get_action", zero_actor)
    observations = np.zeros((3, 3), dtype=np.float32)

    searched = agent.act_batch_critic_search(observations, num_actions=41, margin=0.1)
    halfway = agent.act_batch_critic_search(
        observations, num_actions=41, margin=0.1, blend_fraction=0.5
    )
    filtered = agent.act_batch_critic_search(observations, num_actions=41, margin=2.0)

    np.testing.assert_allclose(searched, 1.0, atol=1e-6)
    np.testing.assert_allclose(halfway, 0.5, atol=1e-6)
    np.testing.assert_allclose(filtered, 0.0, atol=1e-6)


def test_critic_search_unanimous_filter_rejects_critic_switching(monkeypatch):
    class FirstQ(torch.nn.Module):
        def forward(self, observations, actions):
            del observations
            return -(actions[:, :1] - 1.0).pow(2)

    class SecondQ(torch.nn.Module):
        def forward(self, observations, actions):
            del observations
            return -0.2 * actions[:, :1].pow(2)

    agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        SACConfig(device="cpu"),
        "cpu",
    )
    agent.q_networks = torch.nn.ModuleList([FirstQ(), SecondQ()])

    def zero_actor(observations):
        batch_size = observations.shape[0]
        actions = torch.zeros((batch_size, 1), dtype=observations.dtype, device=observations.device)
        log_prob = torch.zeros((batch_size, 1), dtype=observations.dtype, device=observations.device)
        return actions, log_prob, actions

    monkeypatch.setattr(agent.actor, "get_action", zero_actor)
    observations = np.zeros((3, 3), dtype=np.float32)

    clipped = agent.act_batch_critic_search(observations, num_actions=41, margin=0.0)
    unanimous = agent.act_batch_critic_search(
        observations,
        num_actions=41,
        margin=0.0,
        filter_mode="unanimous_advantage",
    )
    assert np.all(clipped > 0.0)
    np.testing.assert_allclose(unanimous, 0.0, atol=1e-6)


def test_critic_search_symmetric_actor_removes_reflection_bias(monkeypatch):
    class FlatQ(torch.nn.Module):
        def forward(self, observations, actions):
            del observations
            return torch.zeros_like(actions[:, :1])

    agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        SACConfig(device="cpu"),
        "cpu",
    )
    agent.q_networks = torch.nn.ModuleList([FlatQ(), FlatQ()])

    def biased_actor(observations):
        actions = observations[:, 1:2] + 0.3
        log_prob = torch.zeros_like(actions)
        return actions, log_prob, actions

    monkeypatch.setattr(agent.actor, "get_action", biased_actor)
    observations = np.asarray([[0.8, 0.4, 0.2], [0.8, -0.4, -0.2]], dtype=np.float32)

    actions = agent.act_batch_critic_search(
        observations,
        num_actions=41,
        margin=10.0,
        filter_mode="symmetric_actor_unanimous_advantage",
    )

    np.testing.assert_allclose(actions[:, 0], [0.4, -0.4], atol=1e-6)


def test_critic_search_symmetric_critics_average_mirrored_values(monkeypatch):
    class BiasedStateQ(torch.nn.Module):
        def forward(self, observations, actions):
            preferred = observations[:, 1:2] + 0.4
            return -(actions[:, :1] - preferred).pow(2)

    agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        SACConfig(device="cpu"),
        "cpu",
    )
    agent.q_networks = torch.nn.ModuleList([BiasedStateQ(), BiasedStateQ()])

    def zero_actor(observations):
        actions = torch.zeros((observations.shape[0], 1), dtype=observations.dtype)
        log_prob = torch.zeros_like(actions)
        return actions, log_prob, actions

    monkeypatch.setattr(agent.actor, "get_action", zero_actor)
    observations = np.asarray([[0.8, 0.6, 0.2], [0.8, -0.6, -0.2]], dtype=np.float32)

    ordinary = agent.act_batch_critic_search(
        observations,
        num_actions=41,
        margin=0.0,
        filter_mode="unanimous_advantage",
    )
    symmetric = agent.act_batch_critic_search(
        observations,
        num_actions=41,
        margin=0.0,
        filter_mode="symmetric_critic_unanimous_advantage",
    )

    np.testing.assert_allclose(ordinary[:, 0], [1.0, -0.2], atol=1e-6)
    np.testing.assert_allclose(symmetric[:, 0], [0.6, -0.6], atol=1e-6)


def test_critic_search_online_target_unanimous_filter_allows_target_veto(monkeypatch):
    class PreferOneQ(torch.nn.Module):
        def forward(self, observations, actions):
            del observations
            return -(actions[:, :1] - 1.0).pow(2)

    class PreferActorQ(torch.nn.Module):
        def forward(self, observations, actions):
            del observations
            return -actions[:, :1].pow(2)

    agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        SACConfig(device="cpu"),
        "cpu",
    )
    agent.q_networks = torch.nn.ModuleList([PreferOneQ(), PreferOneQ()])
    agent.q_target_networks = torch.nn.ModuleList([PreferOneQ(), PreferActorQ()])

    def zero_actor(observations):
        batch_size = observations.shape[0]
        actions = torch.zeros((batch_size, 1), dtype=observations.dtype, device=observations.device)
        log_prob = torch.zeros((batch_size, 1), dtype=observations.dtype, device=observations.device)
        return actions, log_prob, actions

    monkeypatch.setattr(agent.actor, "get_action", zero_actor)
    observations = np.zeros((3, 3), dtype=np.float32)

    online_only = agent.act_batch_critic_search(
        observations,
        num_actions=41,
        margin=0.0,
        filter_mode="unanimous_advantage",
    )
    online_and_target = agent.act_batch_critic_search(
        observations,
        num_actions=41,
        margin=0.0,
        filter_mode="online_target_unanimous_advantage",
    )
    np.testing.assert_allclose(online_only, 1.0, atol=1e-6)
    np.testing.assert_allclose(online_and_target, 0.0, atol=1e-6)


def test_critic_search_joint_online_target_selection_uses_all_four_critics(monkeypatch):
    class PreferOneQ(torch.nn.Module):
        def forward(self, observations, actions):
            del observations
            return -(actions[:, :1] - 1.0).pow(2)

    class PreferHalfQ(torch.nn.Module):
        def forward(self, observations, actions):
            del observations
            return -(actions[:, :1] - 0.5).pow(2)

    agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        SACConfig(device="cpu"),
        "cpu",
    )
    agent.q_networks = torch.nn.ModuleList([PreferOneQ(), PreferOneQ()])
    agent.q_target_networks = torch.nn.ModuleList([PreferHalfQ(), PreferHalfQ()])

    def zero_actor(observations):
        batch_size = observations.shape[0]
        actions = torch.zeros((batch_size, 1), dtype=observations.dtype, device=observations.device)
        log_prob = torch.zeros((batch_size, 1), dtype=observations.dtype, device=observations.device)
        return actions, log_prob, actions

    monkeypatch.setattr(agent.actor, "get_action", zero_actor)
    observations = np.zeros((3, 3), dtype=np.float32)

    online_candidate_with_target_veto = agent.act_batch_critic_search(
        observations,
        num_actions=41,
        margin=0.0,
        filter_mode="online_target_unanimous_advantage",
    )
    joint_candidate = agent.act_batch_critic_search(
        observations,
        num_actions=41,
        margin=0.0,
        filter_mode="online_target_joint_unanimous_advantage",
    )
    np.testing.assert_allclose(online_candidate_with_target_veto, 0.0, atol=1e-6)
    assert np.all(joint_candidate >= 0.7 - 1e-6)
    assert np.all(joint_candidate <= 0.8 + 1e-6)


def test_critic_search_mean_proposal_differs_from_clipped_min_proposal(monkeypatch):
    class StrongPreferOneQ(torch.nn.Module):
        def forward(self, observations, actions):
            del observations
            return -(actions[:, :1] - 1.0).pow(2)

    class WeakPreferZeroQ(torch.nn.Module):
        def forward(self, observations, actions):
            del observations
            return -0.1 * actions[:, :1].pow(2)

    agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        SACConfig(device="cpu"),
        "cpu",
    )
    agent.q_networks = torch.nn.ModuleList([StrongPreferOneQ(), WeakPreferZeroQ()])

    def minus_one_actor(observations):
        batch_size = observations.shape[0]
        actions = -torch.ones((batch_size, 1), dtype=observations.dtype, device=observations.device)
        log_prob = torch.zeros((batch_size, 1), dtype=observations.dtype, device=observations.device)
        return actions, log_prob, actions

    monkeypatch.setattr(agent.actor, "get_action", minus_one_actor)
    observations = np.zeros((3, 3), dtype=np.float32)

    clipped_min = agent.act_batch_critic_search(
        observations,
        num_actions=41,
        margin=0.0,
        filter_mode="unanimous_advantage",
    )
    mean_proposal = agent.act_batch_critic_search(
        observations,
        num_actions=41,
        margin=0.0,
        filter_mode="mean_proposal_unanimous_advantage",
    )
    assert np.all(mean_proposal > clipped_min + 0.1)
    np.testing.assert_allclose(mean_proposal, 0.9, atol=1e-6)


def test_critic_search_midpoint_aggregations_interpolate_mean_and_min():
    stacked_q = torch.tensor([[[1.0]], [[3.0]]])

    mean = SACAgent._aggregate_critic_search_values(stacked_q, "mean")
    mid0125 = SACAgent._aggregate_critic_search_values(stacked_q, "mid_0.125")
    mid025 = SACAgent._aggregate_critic_search_values(stacked_q, "mid_0.25")
    mid0375 = SACAgent._aggregate_critic_search_values(stacked_q, "mid_0.375")
    clipped_min = SACAgent._aggregate_critic_search_values(stacked_q, "min")

    np.testing.assert_allclose(mean.numpy(), [[2.0]])
    np.testing.assert_allclose(mid0125.numpy(), [[1.75]])
    np.testing.assert_allclose(mid025.numpy(), [[1.5]])
    np.testing.assert_allclose(mid0375.numpy(), [[1.25]])
    np.testing.assert_allclose(clipped_min.numpy(), [[1.0]])


def test_critic_search_uncertainty_penalizes_only_disagreement_increase():
    class TwiceActionQ(torch.nn.Module):
        def forward(self, observations, actions):
            del observations
            return 2.0 * actions[:, :1]

    class ActionQ(torch.nn.Module):
        def forward(self, observations, actions):
            del observations
            return actions[:, :1]

    agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        SACConfig(device="cpu"),
        "cpu",
    )
    agent.q_networks = torch.nn.ModuleList([TwiceActionQ(), ActionQ()])
    observations = torch.zeros((2, 3), dtype=torch.float32)
    actor_actions = torch.zeros((2, 1), dtype=torch.float32)
    best_actions = torch.ones((2, 1), dtype=torch.float32)

    adjusted = agent._critic_search_filter_advantage(
        observations,
        actor_actions,
        best_actions,
        best_q=torch.zeros((2, 1)),
        actor_q=torch.zeros((2, 1)),
        filter_mode="unc05_increase_unanimous_advantage",
    )

    np.testing.assert_allclose(adjusted.numpy(), [[0.5], [0.5]])


def test_critic_search_target_proposal_uses_target_critics_and_respects_online_veto(monkeypatch):
    class PreferOneQ(torch.nn.Module):
        def forward(self, observations, actions):
            del observations
            return -(actions[:, :1] - 1.0).pow(2)

    class PreferMinusHalfQ(torch.nn.Module):
        def forward(self, observations, actions):
            del observations
            return -(actions[:, :1] + 0.5).pow(2)

    agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        SACConfig(device="cpu"),
        "cpu",
    )
    agent.q_networks = torch.nn.ModuleList([PreferOneQ(), PreferOneQ()])
    agent.q_target_networks = torch.nn.ModuleList([PreferMinusHalfQ(), PreferMinusHalfQ()])

    def zero_actor(observations):
        batch_size = observations.shape[0]
        actions = torch.zeros((batch_size, 1), dtype=observations.dtype, device=observations.device)
        log_prob = torch.zeros((batch_size, 1), dtype=observations.dtype, device=observations.device)
        return actions, log_prob, actions

    monkeypatch.setattr(agent.actor, "get_action", zero_actor)
    observations = np.zeros((3, 3), dtype=np.float32)

    target_only = agent.act_batch_critic_search(
        observations,
        num_actions=41,
        margin=0.0,
        filter_mode="target_unanimous_advantage",
    )
    online_veto = agent.act_batch_critic_search(
        observations,
        num_actions=41,
        margin=0.0,
        filter_mode="target_proposal_online_unanimous_advantage",
    )
    all_critic_veto = agent.act_batch_critic_search(
        observations,
        num_actions=41,
        margin=0.0,
        filter_mode="target_proposal_online_target_unanimous_advantage",
    )
    np.testing.assert_allclose(target_only, -0.5, atol=1e-6)
    np.testing.assert_allclose(online_veto, 0.0, atol=1e-6)
    np.testing.assert_allclose(all_critic_veto, 0.0, atol=1e-6)


def test_critic_search_lcb_aggregation_penalizes_disagreement():
    stacked_q = torch.tensor([[1.0, 3.0], [3.0, 1.0]])
    torch.testing.assert_close(
        SACAgent._aggregate_critic_search_values(stacked_q, "min"),
        torch.tensor([1.0, 1.0]),
    )
    torch.testing.assert_close(
        SACAgent._aggregate_critic_search_values(stacked_q, "lcb_0.25"),
        torch.tensor([0.5, 0.5]),
    )
    torch.testing.assert_close(
        SACAgent._aggregate_critic_search_values(stacked_q, "lcb_1.0"),
        torch.tensor([-1.0, -1.0]),
    )


@pytest.mark.parametrize("distributional", [False, True])
def test_sacn_update_smoke_for_scalar_and_distributional_critics(distributional):
    env = make_env("Pendulum-v1", seed=0)
    try:
        cfg = SACConfig(
            buffer_size=64,
            learning_starts=4,
            batch_size=4,
            device="cpu",
            update_diagnostics=False,
            sacn_n_step=3,
            simba_backbone=distributional,
            simba_actor_hidden_dim=16,
            simba_critic_hidden_dim=16,
            simba_distributional_critic=distributional,
            simba_critic_num_bins=11,
            simba_reward_scaling=distributional,
        )
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        agent = SACAgent(obs_dim, env.action_space.low, env.action_space.high, cfg, device="cpu")
        replay = InstrumentedReplayBuffer(
            64,
            env.observation_space,
            env.action_space,
            device="cpu",
            n_envs=1,
            handle_timeout_termination=False,
        )
        uniform_log_prob = -float(np.sum(np.log(env.action_space.high - env.action_space.low)))

        obs, _ = env.reset(seed=0)
        for step in range(10):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.observe(obs)
            agent.observe_reward(float(reward), bool(terminated or truncated))
            replay.add(
                obs.reshape(1, obs_dim),
                next_obs.reshape(1, obs_dim),
                action.reshape(1, action_dim),
                np.asarray([reward], dtype=np.float32),
                np.asarray([terminated]),
                [{}],
                step=step + 1,
                episode_id=0,
                action_log_prob=np.asarray([uniform_log_prob], dtype=np.float32),
            )
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()

        batch = replay.sample_sacn(batch_size=4, n_step=3)
        metrics = agent.update(batch, update_step=2)

        assert metrics["q_loss"] >= 0.0
        assert "actor_loss" in metrics
        assert metrics["sacn_n_step"] == pytest.approx(3.0)
        assert metrics["sacn_weight_min"] >= 0.0
        assert metrics["sacn_weight_max"] <= 1.0
        assert metrics["sacn_entropy_samples_max"] >= 1.0
        assert metrics["sacn_horizon_ess_first_fraction"] == pytest.approx(1.0)
        assert 0.0 <= metrics["sacn_horizon_ess_last_fraction"] <= 1.0
        assert metrics["sacn_horizon_active_count"] == pytest.approx(3.0)
        assert metrics["sacn_horizon_loss_weight_sum"] == pytest.approx(3.0)
        assert metrics["sacn_horizon_last_loss_weight"] == pytest.approx(1.0)
        assert metrics["sacn_horizon_last_loss_weight_share"] == pytest.approx(
            1.0 / 3.0
        )
        assert metrics["sacn_target_mode_is_fast_last"] == pytest.approx(0.0)
    finally:
        env.close()


@pytest.mark.parametrize("distributional", [False, True])
def test_sacn_no_importance_non_soft_update_smoke(distributional):
    env = make_env("Pendulum-v1", seed=0)
    try:
        cfg = SACConfig(
            buffer_size=64,
            learning_starts=4,
            batch_size=4,
            device="cpu",
            update_diagnostics=False,
            sacn_n_step=4,
            sacn_importance_mode="none",
            sacn_non_soft_targets=True,
            simba_backbone=distributional,
            simba_actor_hidden_dim=16,
            simba_critic_hidden_dim=16,
            simba_distributional_critic=distributional,
            simba_critic_num_bins=11,
            simba_reward_scaling=distributional,
        )
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        agent = SACAgent(obs_dim, env.action_space.low, env.action_space.high, cfg, device="cpu")
        replay = InstrumentedReplayBuffer(
            64,
            env.observation_space,
            env.action_space,
            device="cpu",
            n_envs=1,
            handle_timeout_termination=False,
        )
        uniform_log_prob = -float(np.sum(np.log(env.action_space.high - env.action_space.low)))

        obs, _ = env.reset(seed=0)
        for step in range(12):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.observe(obs)
            agent.observe_reward(float(reward), bool(terminated or truncated))
            replay.add(
                obs.reshape(1, obs_dim),
                next_obs.reshape(1, obs_dim),
                action.reshape(1, action_dim),
                np.asarray([reward], dtype=np.float32),
                np.asarray([terminated]),
                [{}],
                step=step + 1,
                episode_id=0,
                action_log_prob=np.asarray([uniform_log_prob], dtype=np.float32),
            )
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()

        batch = replay.sample_sacn(batch_size=4, n_step=4)
        metrics = agent.update(batch, update_step=2)

        assert metrics["q_loss"] >= 0.0
        assert metrics["sacn_weight_min"] == pytest.approx(1.0)
        assert metrics["sacn_weight_max"] == pytest.approx(1.0)
        assert metrics["sacn_log_omega_std"] == pytest.approx(0.0)
        assert metrics["sacn_horizon_active_count"] == pytest.approx(4.0)
        assert metrics["sacn_importance_is_density"] == pytest.approx(0.0)
        assert metrics["sacn_non_soft_targets"] == pytest.approx(1.0)
        assert metrics["sacn_horizon_loss_weight_sum"] == pytest.approx(4.0)
        assert metrics["sacn_horizon_last_loss_weight"] == pytest.approx(1.0)
        assert metrics["sacn_horizon_last_loss_weight_share"] == pytest.approx(0.25)
        assert metrics["sacn_horizon_lambda"] == pytest.approx(1.0)
    finally:
        env.close()


def test_sacn_horizon_ess_mask_preserves_loss_scale():
    cfg = SACConfig(
        buffer_size=16,
        batch_size=4,
        device="cpu",
        sacn_min_horizon_ess_fraction=0.75,
    )
    agent = SACAgent(3, np.asarray([-1.0], dtype=np.float32), np.asarray([1.0], dtype=np.float32), cfg, device="cpu")
    weights = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    masked_weights, horizon_mask, horizon_ess = agent._sacn_apply_horizon_support(weights)
    loss = agent._sacn_weighted_horizon_loss(torch.ones_like(weights), masked_weights, horizon_mask)

    torch.testing.assert_close(horizon_ess, torch.tensor([1.0, 0.25, 0.25]))
    torch.testing.assert_close(horizon_mask, torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_pendulum_symmetry_augmentation_mirrors_one_step_batch():
    cfg = SACConfig(batch_size=2, device="cpu", pendulum_symmetry_augmentation=True, update_diagnostics=False)
    agent = SACAgent(3, np.asarray([-2.0], dtype=np.float32), np.asarray([2.0], dtype=np.float32), cfg, device="cpu")
    observations = torch.tensor([[1.0, 0.2, 0.3], [0.5, -0.4, 0.6]])
    actions = torch.tensor([[0.7], [-0.8]])
    next_observations = torch.tensor([[0.9, 0.1, -0.2], [0.4, -0.5, 0.7]])
    data = ReplayBufferSamples(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        dones=torch.tensor([[0.0], [1.0]]),
        rewards=torch.tensor([[1.0], [2.0]]),
    )

    augmented, reference_actions, _reference_critic_actions, metrics = agent._augment_pendulum_symmetry_batch(
        data,
        reference_actions=actions,
        reference_critic_actions=None,
    )

    torch.testing.assert_close(augmented.observations[:2], observations)
    torch.testing.assert_close(augmented.observations[2:], torch.tensor([[1.0, -0.2, -0.3], [0.5, 0.4, -0.6]]))
    torch.testing.assert_close(augmented.actions[2:], -actions)
    torch.testing.assert_close(augmented.next_observations[2:], torch.tensor([[0.9, -0.1, 0.2], [0.4, 0.5, -0.7]]))
    torch.testing.assert_close(augmented.rewards[2:], data.rewards)
    torch.testing.assert_close(reference_actions[2:], -actions)
    assert metrics["pendulum_symmetry_batch_multiplier"] == pytest.approx(2.0)


def test_pendulum_symmetry_augmentation_mirrors_sacn_batch():
    cfg = SACConfig(
        batch_size=2,
        device="cpu",
        pendulum_symmetry_augmentation=True,
        sacn_n_step=3,
        sacn_importance_mode="none",
        update_diagnostics=False,
    )
    agent = SACAgent(3, np.asarray([-2.0], dtype=np.float32), np.asarray([2.0], dtype=np.float32), cfg, device="cpu")
    trajectory_observations = torch.tensor(
        [
            [[1.0, 0.1, 0.2], [0.9, 0.2, 0.3], [0.8, 0.3, 0.4]],
            [[0.7, -0.1, -0.2], [0.6, -0.2, -0.3], [0.5, -0.3, -0.4]],
        ]
    )
    trajectory_actions = torch.tensor([[[0.1], [0.2], [0.3]], [[-0.1], [-0.2], [-0.3]]])
    data = SACNReplaySamples(
        observations=trajectory_observations[:, 0],
        actions=trajectory_actions[:, 0],
        trajectory_observations=trajectory_observations,
        trajectory_actions=trajectory_actions,
        trajectory_next_observations=trajectory_observations + 0.01,
        trajectory_rewards=torch.ones((2, 3)),
        trajectory_dones=torch.zeros((2, 3)),
        trajectory_action_log_probs=torch.zeros((2, 3)),
    )

    augmented, _reference_actions, _reference_critic_actions, metrics = agent._augment_pendulum_symmetry_batch(
        data,
        reference_actions=None,
        reference_critic_actions=None,
    )

    assert augmented.observations.shape[0] == 4
    torch.testing.assert_close(augmented.trajectory_observations[2:, :, 0], trajectory_observations[:, :, 0])
    torch.testing.assert_close(augmented.trajectory_observations[2:, :, 1], -trajectory_observations[:, :, 1])
    torch.testing.assert_close(augmented.trajectory_observations[2:, :, 2], -trajectory_observations[:, :, 2])
    torch.testing.assert_close(augmented.trajectory_actions[2:], -trajectory_actions)
    torch.testing.assert_close(augmented.trajectory_rewards[2:], data.trajectory_rewards)
    assert metrics["pendulum_symmetry_augmented_batch_size"] == pytest.approx(4.0)


def test_pendulum_actor_symmetry_loss_enforces_mean_equivariance():
    cfg = SACConfig(
        device="cpu",
        update_diagnostics=False,
        pendulum_actor_symmetry_weight=0.25,
    )
    agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        cfg,
        device="cpu",
    )

    def odd_actor_forward(observations):
        latent_mean = observations[:, 1:2] + 0.25 * observations[:, 2:3]
        return latent_mean, torch.zeros_like(latent_mean)

    agent.actor.forward = odd_actor_forward
    observations = torch.tensor(
        [[1.0, 0.2, 0.4], [0.5, -0.3, 0.8]], dtype=torch.float32
    )
    mirrored_observations = agent._mirror_pendulum_observations(observations)
    latent_mean, _ = agent.actor(observations)
    policy_mean = (
        torch.tanh(latent_mean) * agent.actor.action_scale + agent.actor.action_bias
    )

    loss, metrics = agent._pendulum_actor_symmetry_loss(
        policy_mean=policy_mean,
        mirrored_observations=mirrored_observations,
    )

    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0.0)
    assert metrics["pendulum_actor_symmetry_abs_error_mean"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["pendulum_actor_symmetry_weight"] == pytest.approx(0.25)


def test_pendulum_critic_symmetry_loss_enforces_q_invariance():
    class InvariantQ(torch.nn.Module):
        def forward(self, observations, actions):
            return (
                observations[:, 0:1]
                + observations[:, 1:2].square()
                + observations[:, 2:3].square()
                + actions.square()
            )

    class AsymmetricQ(torch.nn.Module):
        def forward(self, observations, actions):
            return observations[:, 1:2] + observations[:, 2:3] + actions

    cfg = SACConfig(
        device="cpu",
        update_diagnostics=False,
        pendulum_critic_symmetry_weight=0.5,
    )
    agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        cfg,
        device="cpu",
    )
    agent.q_networks = torch.nn.ModuleList([InvariantQ(), AsymmetricQ()])
    observations = torch.tensor(
        [[1.0, 0.2, 0.4], [0.5, -0.3, 0.8]], dtype=torch.float32
    )
    actions = torch.tensor([[0.7], [-0.2]], dtype=torch.float32)
    mirrored_observations = agent._mirror_pendulum_observations(observations)
    original_q_values = [q(observations, actions).view(-1) for q in agent.q_networks]

    losses, metrics = agent._pendulum_critic_symmetry_losses(
        observations=observations,
        mirrored_observations=mirrored_observations,
        actions=actions,
        original_q_values=original_q_values,
    )

    torch.testing.assert_close(losses[0], torch.zeros_like(losses[0]), atol=1e-7, rtol=0.0)
    assert float(losses[1]) > 0.0
    assert metrics["pendulum_critic_symmetry_loss_sum"] == pytest.approx(float(losses[1]))
    assert metrics["pendulum_critic_symmetry_contribution"] == pytest.approx(
        0.5 * metrics["pendulum_critic_symmetry_loss_sum"]
    )


def test_pendulum_symmetry_consistency_update_logs_separate_losses():
    cfg = SACConfig(
        batch_size=4,
        device="cpu",
        update_diagnostics=False,
        pendulum_actor_symmetry_weight=0.25,
        pendulum_critic_symmetry_weight=0.5,
    )
    agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        cfg,
        device="cpu",
    )
    data = ReplayBufferSamples(
        observations=torch.tensor(
            [
                [1.0, 0.2, 0.4],
                [0.5, -0.3, 0.8],
                [-0.2, 0.6, -0.1],
                [0.7, -0.4, -0.5],
            ]
        ),
        actions=torch.tensor([[0.7], [-0.2], [0.1], [-0.8]]),
        next_observations=torch.tensor(
            [
                [0.9, 0.1, 0.3],
                [0.6, -0.2, 0.7],
                [-0.1, 0.5, -0.2],
                [0.8, -0.3, -0.4],
            ]
        ),
        dones=torch.zeros((4, 1)),
        rewards=torch.zeros((4, 1)),
    )

    metrics = agent.update(data, update_step=2)

    assert metrics["pendulum_actor_symmetry_contribution"] == pytest.approx(
        0.25 * metrics["pendulum_actor_symmetry_loss"]
    )
    assert metrics["actor_loss"] == pytest.approx(
        metrics["sac_actor_loss"]
        + metrics["pendulum_actor_symmetry_contribution"],
        rel=1e-5,
        abs=1e-5,
    )
    assert metrics["pendulum_critic_symmetry_contribution"] == pytest.approx(
        0.5 * metrics["pendulum_critic_symmetry_loss_sum"]
    )
    assert "pendulum_critic_1_symmetry_loss" in metrics
    assert "pendulum_critic_2_symmetry_loss" in metrics


def test_zero_symmetry_consistency_weights_leave_update_path_disabled():
    cfg = SACConfig(batch_size=2, device="cpu", update_diagnostics=False)
    agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        cfg,
        device="cpu",
    )
    data = ReplayBufferSamples(
        observations=torch.zeros((2, 3)),
        actions=torch.zeros((2, 1)),
        next_observations=torch.zeros((2, 3)),
        dones=torch.zeros((2, 1)),
        rewards=torch.zeros((2, 1)),
    )

    metrics = agent.update(data, update_step=2)

    assert "pendulum_actor_symmetry_loss" not in metrics
    assert "pendulum_critic_symmetry_loss" not in metrics


def test_sacn_fast_last_mode_selects_first_and_last_active_with_lambda():
    cfg = SACConfig(
        buffer_size=16,
        batch_size=4,
        device="cpu",
        sacn_target_mode="fast_last",
        sacn_horizon_lambda=0.5,
    )
    agent = SACAgent(3, np.asarray([-1.0], dtype=np.float32), np.asarray([1.0], dtype=np.float32), cfg, device="cpu")
    weights = torch.ones((4, 4))
    horizon_mask = torch.tensor([1.0, 1.0, 0.0, 1.0])

    selected_weights, horizon_loss_weights = agent._sacn_apply_target_mode_and_decay(weights, horizon_mask)
    loss = agent._sacn_weighted_horizon_loss(torch.ones_like(weights), selected_weights, horizon_loss_weights)

    torch.testing.assert_close(horizon_loss_weights, torch.tensor([1.0, 0.0, 0.0, 0.125]))
    torch.testing.assert_close(selected_weights[0], torch.tensor([1.0, 0.0, 0.0, 0.125]))
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_weight_projection_projects_hyperdense_rows_to_unit_norm():
    layer = SimbaHyperDense(3, 4)
    with torch.no_grad():
        layer.weight.mul_(5.0)

    project_simba_weights_to_unit_norm(layer)

    row_norms = torch.linalg.vector_norm(layer.weight, dim=1)
    assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-6)


def test_weight_projection_ignores_standard_linear_layers():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), SimbaHyperDense(4, 2))
    with torch.no_grad():
        model[0].weight.fill_(3.0)
        model[1].weight.mul_(5.0)
    linear_before = model[0].weight.detach().clone()

    project_simba_weights_to_unit_norm(model)

    assert torch.allclose(model[0].weight, linear_before)
    row_norms = torch.linalg.vector_norm(model[1].weight, dim=1)
    assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-6)


def test_agent_rejects_weight_projection_on_cleanrl_backbone():
    cfg = SACConfig(device="cpu", simba_weight_projection=True)

    with pytest.raises(ValueError, match="simba_weight_projection requires simba_backbone"):
        SACAgent(3, np.asarray([-2.0], dtype=np.float32), np.asarray([2.0], dtype=np.float32), cfg, device="cpu")


def test_agent_uses_configured_alpha_and_lr_schedule():
    cfg = SACConfig(
        total_steps=100,
        updates_per_step=1,
        device="cpu",
        policy_lr=1e-4,
        q_lr=1e-4,
        policy_lr_final=5e-5,
        q_lr_final=5e-5,
        alpha_initial_value=0.01,
        target_entropy_scale=-0.5,
    )

    agent = SACAgent(3, np.asarray([-2.0], dtype=np.float32), np.asarray([2.0], dtype=np.float32), cfg, device="cpu")
    q_lr, policy_lr = agent._apply_lr_schedule(update_step=50)

    assert agent.alpha == pytest.approx(0.01)
    assert agent.target_entropy == pytest.approx(-0.5)
    assert q_lr == pytest.approx(7.5e-5)
    assert policy_lr == pytest.approx(7.5e-5)
    assert agent._current_alpha_lr == pytest.approx(7.5e-5)


def test_agent_can_schedule_alpha_lr_independently():
    cfg = SACConfig(
        total_steps=100,
        updates_per_step=1,
        device="cpu",
        policy_lr=1e-4,
        q_lr=1e-4,
        alpha_lr=2e-5,
        alpha_lr_final=1e-5,
    )

    agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        cfg,
        device="cpu",
    )
    q_lr, policy_lr = agent._apply_lr_schedule(update_step=50)

    assert q_lr == pytest.approx(1e-4)
    assert policy_lr == pytest.approx(1e-4)
    assert agent._current_alpha_lr == pytest.approx(1.5e-5)
    assert agent.alpha_optimizer.param_groups[0]["lr"] == pytest.approx(1.5e-5)


def test_independent_alpha_lr_checkpoint_resume_preserves_optimizer_and_schedule(
    tmp_path,
):
    cfg = SACConfig(
        total_steps=100,
        updates_per_step=1,
        device="cpu",
        policy_lr=1e-4,
        q_lr=1e-4,
        alpha_lr=2e-5,
        alpha_lr_final=1e-5,
    )
    action_low = np.asarray([-2.0], dtype=np.float32)
    action_high = np.asarray([2.0], dtype=np.float32)
    agent = SACAgent(3, action_low, action_high, cfg, device="cpu")
    agent._apply_lr_schedule(update_step=40)
    checkpoint = tmp_path / "independent-alpha-lr.pt"
    agent.save_checkpoint(checkpoint)

    resumed = SACAgent(3, action_low, action_high, cfg, device="cpu")
    resumed.load_checkpoint(checkpoint, load_optimizers=True)

    assert resumed.alpha_optimizer.param_groups[0]["lr"] == pytest.approx(1.6e-5)
    resumed._apply_lr_schedule(update_step=40)
    assert resumed._current_alpha_lr == pytest.approx(1.6e-5)
    assert resumed.alpha_optimizer.param_groups[0]["lr"] == pytest.approx(1.6e-5)
    q_lr, policy_lr = resumed._apply_lr_schedule(update_step=75)
    assert resumed._current_alpha_lr == pytest.approx(1.25e-5)
    assert q_lr == pytest.approx(1e-4)
    assert policy_lr == pytest.approx(1e-4)


def test_agent_pendulum_potential_shapes_rewards(tmp_path):
    dp_path = tmp_path / "dp.csv"
    theta_values = [-math.pi, -math.pi / 2.0, 0.0, math.pi / 2.0]
    velocity_values = [-1.0, 0.0, 1.0]
    with dp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["theta", "theta_dot", "dp_policy_return"])
        writer.writeheader()
        for velocity_index, velocity in enumerate(velocity_values):
            for theta_index, theta in enumerate(theta_values):
                writer.writerow(
                    {
                        "theta": theta,
                        "theta_dot": velocity,
                        "dp_policy_return": float(10 * velocity_index + theta_index),
                    }
                )

    cfg = SACConfig(
        device="cpu",
        gamma=0.99,
        pendulum_potential_shaping_weight=0.1,
        pendulum_potential_shaping_start_update=3,
        pendulum_potential_shaping_abs_theta_low=2.0,
        pendulum_potential_shaping_velocity_limit=1.5,
        pendulum_potential_shaping_source="dp_policy",
        pendulum_potential_shaping_dp_grid_path=str(dp_path),
    )
    agent = SACAgent(3, np.asarray([-2.0], dtype=np.float32), np.asarray([2.0], dtype=np.float32), cfg, device="cpu")
    observations = torch.tensor([[math.cos(-math.pi), math.sin(-math.pi), -1.0]], dtype=torch.float32)
    next_observations = torch.tensor([[math.cos(0.0), math.sin(0.0), 0.0]], dtype=torch.float32)
    rewards = torch.tensor([[-1.0]], dtype=torch.float32)
    dones = torch.tensor([[0.0]], dtype=torch.float32)

    inactive_rewards, inactive_metrics = agent._shape_pendulum_rewards(
        observations,
        next_observations,
        rewards,
        dones,
        update_step=2,
    )
    torch.testing.assert_close(inactive_rewards, rewards)
    assert inactive_metrics["pendulum_potential_shaping_active"] == 0.0

    shaped_rewards, metrics = agent._shape_pendulum_rewards(
        observations,
        next_observations,
        rewards,
        dones,
        update_step=3,
    )

    expected = -1.0 + 0.1 * (0.99 * 12.0 - 0.0)
    torch.testing.assert_close(shaped_rewards, torch.tensor([[expected]], dtype=torch.float32))
    assert metrics["pendulum_potential_shaping_active"] == 1.0
    assert metrics["pendulum_potential_shaping_start_update"] == pytest.approx(3.0)
    assert metrics["pendulum_potential_shaping_weight"] == pytest.approx(0.1)
    assert metrics["pendulum_potential_shaping_gate_fraction"] == pytest.approx(1.0)
    assert metrics["pendulum_potential_shaping_abs_mean"] > 0.0

    agent.cfg.pendulum_potential_shaping_velocity_limit = 0.5
    masked_rewards, masked_metrics = agent._shape_pendulum_rewards(
        observations,
        next_observations,
        rewards,
        dones,
        update_step=3,
    )
    torch.testing.assert_close(masked_rewards, rewards)
    assert masked_metrics["pendulum_potential_shaping_gate_fraction"] == pytest.approx(0.0)


def test_reference_auxiliary_actor_loss_logs_metrics():
    env = make_env("Pendulum-v1", seed=0)
    try:
        cfg = SACConfig(
            buffer_size=32,
            learning_starts=4,
            batch_size=4,
            device="cpu",
            reference_auxiliary_mode="bc",
            reference_auxiliary_weight=0.5,
        )
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        agent = SACAgent(obs_dim, env.action_space.low, env.action_space.high, cfg, device="cpu")
        replay = InstrumentedReplayBuffer(
            32,
            env.observation_space,
            env.action_space,
            device="cpu",
            n_envs=1,
            handle_timeout_termination=False,
        )

        obs, _ = env.reset(seed=0)
        for step in range(8):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            replay.add(
                obs.reshape(1, obs_dim),
                next_obs.reshape(1, obs_dim),
                action.reshape(1, action_dim),
                np.asarray([reward], dtype=np.float32),
                np.asarray([terminated]),
                [{}],
                step=step,
                episode_id=0,
            )
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()

        batch = replay.sample(batch_size=4)
        reference_actions = np.zeros((4, action_dim), dtype=np.float32)
        anchor_observations = batch.observations.detach().cpu().numpy()
        anchor_actions = np.full((4, action_dim), 0.1, dtype=np.float32)
        metrics = agent.update(
            batch,
            update_step=2,
            reference_actions=reference_actions,
            reference_anchor_observations=anchor_observations,
            reference_anchor_actions=anchor_actions,
        )

        assert "reference_actor_bc_loss" in metrics
        assert "reference_actor_q_advantage_mean" in metrics
        assert metrics["reference_actor_bc_mask_fraction"] == pytest.approx(1.0)
        assert metrics["reference_actor_anchor_fraction"] == pytest.approx(0.5)
        assert "actor_sac_bc_gradient_cosine" in metrics
        assert metrics["actor_loss"] == pytest.approx(
            metrics["sac_actor_loss"] + 0.5 * metrics["reference_actor_bc_loss"],
            rel=1e-5,
            abs=1e-5,
        )
    finally:
        env.close()


def test_q_filtered_replay_bc_keeps_anchor_samples_unconditional():
    class ActionValue(torch.nn.Module):
        def forward(self, observations, actions):
            del observations
            return actions[:, :1]

    env = make_env("Pendulum-v1", seed=0)
    try:
        agent = SACAgent(
            3,
            env.action_space.low,
            env.action_space.high,
            SACConfig(
                buffer_size=16,
                learning_starts=1,
                batch_size=4,
                device="cpu",
                reference_auxiliary_mode="q_filtered_replay_bc",
                reference_auxiliary_margin=0.0,
            ),
            device="cpu",
        )
        agent.q1 = ActionValue()
        agent.q2 = ActionValue()
        observations = torch.zeros((4, 3), dtype=torch.float32)
        actor_actions = torch.zeros((4, 1), dtype=torch.float32)
        # Replay rows: one reference win and one actor win. Anchor rows both
        # look worse to the deliberately simple critic but must remain cloned.
        reference_actions = torch.tensor([[1.0], [-1.0], [-1.0], [-1.0]])

        loss, metrics = agent._reference_auxiliary_actor_loss(
            observations,
            actor_actions,
            reference_actions,
            replay_sample_count=2,
        )

        assert float(loss.detach()) == pytest.approx(0.25)
        assert metrics["reference_actor_bc_mask_fraction"] == pytest.approx(0.75)
        assert metrics["reference_actor_bc_replay_mask_fraction"] == pytest.approx(0.5)
        assert metrics["reference_actor_bc_anchor_mask_fraction"] == pytest.approx(1.0)
        assert metrics["reference_actor_bc_selected_count"] == pytest.approx(3.0)
        assert metrics["reference_actor_bc_replay_selected_count"] == pytest.approx(1.0)
        assert metrics["reference_actor_bc_normalizer_count"] == pytest.approx(3.0)
        assert metrics[
            "reference_actor_bc_replay_normalization_is_full_batch"
        ] == pytest.approx(0.0)
        assert metrics["reference_actor_bc_anchor_unconditional"] == pytest.approx(1.0)
        assert metrics["reference_actor_bc_q_filter_active"] == pytest.approx(1.0)

        agent.cfg.reference_auxiliary_replay_normalization = "full_batch_mean"
        full_batch_loss, full_batch_metrics = agent._reference_auxiliary_actor_loss(
            observations,
            actor_actions,
            reference_actions,
            replay_sample_count=2,
        )
        assert full_batch_loss == pytest.approx(0.1875)
        assert full_batch_metrics["reference_actor_bc_normalizer_count"] == pytest.approx(4.0)
        assert full_batch_metrics[
            "reference_actor_bc_replay_normalization_is_full_batch"
        ] == pytest.approx(1.0)
        # Restore the backwards-compatible default before checking warmup.
        agent.cfg.reference_auxiliary_replay_normalization = "selected_mean"

        warmup_loss, warmup_metrics = agent._reference_auxiliary_actor_loss(
            observations,
            actor_actions,
            reference_actions,
            replay_sample_count=2,
            q_filter_active=False,
        )
        assert warmup_loss == pytest.approx(0.25)
        assert warmup_metrics["reference_actor_bc_mask_fraction"] == pytest.approx(1.0)
        assert warmup_metrics["reference_actor_bc_replay_mask_fraction"] == pytest.approx(1.0)
        assert warmup_metrics["reference_actor_bc_q_filter_active"] == pytest.approx(0.0)

        with pytest.raises(ValueError, match="requires replay_sample_count"):
            agent._reference_auxiliary_actor_loss(
                observations,
                actor_actions,
                reference_actions,
            )
    finally:
        env.close()


def test_robust_reference_bc_filter_requires_every_online_and_target_critic():
    class SignedActionValue(torch.nn.Module):
        def __init__(self, sign: float):
            super().__init__()
            self.sign = torch.nn.Parameter(torch.tensor(sign))

        def forward(self, observations, actions):
            del observations
            return self.sign * actions[:, :1]

    env = make_env("Pendulum-v1", seed=0)
    try:
        agent = SACAgent(
            3,
            env.action_space.low,
            env.action_space.high,
            SACConfig(
                buffer_size=16,
                learning_starts=1,
                batch_size=3,
                device="cpu",
                reference_auxiliary_mode="q_filtered_replay_bc",
                reference_auxiliary_q_filter_mode="online_target_unanimous",
            ),
            device="cpu",
        )
        critics = [
            SignedActionValue(1.0),
            SignedActionValue(1.0),
            SignedActionValue(1.0),
            # This target critic vetoes the reference on both replay rows.
            SignedActionValue(-1.0),
        ]
        agent.q_networks = torch.nn.ModuleList(critics[:2])
        agent.q_target_networks = torch.nn.ModuleList(critics[2:])
        observations = torch.zeros((3, 3), dtype=torch.float32)
        actor_actions = torch.zeros((3, 1), dtype=torch.float32, requires_grad=True)
        reference_actions = torch.ones((3, 1), dtype=torch.float32)

        loss, metrics = agent._reference_auxiliary_actor_loss(
            observations,
            actor_actions,
            reference_actions,
            replay_sample_count=2,
        )
        loss.backward()

        assert float(loss.detach()) == pytest.approx(0.25)
        assert metrics["reference_actor_bc_replay_mask_fraction"] == pytest.approx(0.0)
        assert metrics["reference_actor_bc_anchor_mask_fraction"] == pytest.approx(1.0)
        assert metrics[
            "reference_actor_bc_q_filter_is_online_target_unanimous"
        ] == pytest.approx(1.0)
        assert metrics["reference_actor_bc_q_filter_critic_count"] == pytest.approx(4.0)
        # Q values decide a detached mask; BC must not update any critic.
        assert all(critic.sign.grad is None for critic in critics)
        assert actor_actions.grad is not None
        torch.testing.assert_close(
            actor_actions.grad[:2], torch.zeros_like(actor_actions.grad[:2])
        )
        assert torch.count_nonzero(actor_actions.grad[2:]).item() > 0
    finally:
        env.close()


def test_joint_actor_can_run_bc_before_delayed_sac_and_decay_bc_weight():
    env = make_env("Pendulum-v1", seed=0)
    try:
        cfg = SACConfig(
            buffer_size=16,
            learning_starts=1,
            batch_size=4,
            device="cpu",
            reference_auxiliary_mode="bc",
            reference_auxiliary_weight=2.0,
            reference_auxiliary_weight_final=0.5,
            reference_auxiliary_decay_updates=10,
            sac_actor_loss_start_step=10,
        )
        agent = SACAgent(3, env.action_space.low, env.action_space.high, cfg, device="cpu")
        data = ReplayBufferSamples(
            observations=torch.zeros((4, 3)),
            actions=torch.zeros((4, 1)),
            next_observations=torch.zeros((4, 3)),
            dones=torch.zeros((4, 1)),
            rewards=torch.zeros((4, 1)),
        )
        alpha_before = agent.alpha
        metrics = agent.update(
            data,
            update_step=2,
            reference_actions=np.ones((4, 1), dtype=np.float32),
        )
        expected_weight = 2.0 + 0.2 * (0.5 - 2.0)
        assert metrics["sac_actor_loss_active"] == 0.0
        assert metrics["reference_actor_loss_weight"] == pytest.approx(expected_weight)
        assert metrics["actor_loss"] == pytest.approx(
            expected_weight * metrics["reference_actor_bc_loss"], rel=1e-5, abs=1e-5
        )
        assert agent.alpha == pytest.approx(alpha_before)
    finally:
        env.close()


def test_reference_auxiliary_stop_update_is_exclusive_at_sac_boundary():
    env = make_env("Pendulum-v1", seed=0)
    try:
        cfg = SACConfig(
            buffer_size=16,
            learning_starts=1,
            batch_size=4,
            device="cpu",
            policy_frequency=1,
            actor_updates_per_trigger=1,
            reference_auxiliary_mode="bc",
            reference_auxiliary_weight=2.0,
            reference_auxiliary_weight_final=0.5,
            reference_auxiliary_decay_updates=10_000,
            reference_auxiliary_stop_update=6_000,
            sac_actor_loss_start_step=6_000,
        )
        agent = SACAgent(3, env.action_space.low, env.action_space.high, cfg, device="cpu")
        data = ReplayBufferSamples(
            observations=torch.zeros((4, 3)),
            actions=torch.zeros((4, 1)),
            next_observations=torch.zeros((4, 3)),
            dones=torch.zeros((4, 1)),
            rewards=torch.zeros((4, 1)),
        )
        reference_actions = np.ones((4, 1), dtype=np.float32)

        before = agent.update(
            data,
            update_step=5_999,
            reference_actions=reference_actions,
        )
        boundary = agent.update(
            data,
            update_step=6_000,
            reference_actions=reference_actions,
        )

        assert before["reference_actor_loss_weight"] > 0.0
        assert before["sac_actor_loss_active"] == pytest.approx(0.0)
        assert "reference_actor_bc_loss" in before
        assert boundary["reference_actor_loss_weight"] == pytest.approx(0.0)
        assert boundary["sac_actor_loss_active"] == pytest.approx(1.0)
        assert "reference_actor_bc_loss" not in boundary

        agent.cfg.reference_auxiliary_stop_update = 0
        assert agent._reference_auxiliary_weight(6_000) > 0.0
    finally:
        env.close()


def test_deterministic_mean_actor_objective_does_not_update_temperature_or_std_head():
    env = make_env("Pendulum-v1", seed=0)
    try:
        torch.manual_seed(19)
        cfg = SACConfig(
            buffer_size=16,
            learning_starts=1,
            batch_size=8,
            device="cpu",
            policy_frequency=1,
            actor_updates_per_trigger=1,
            sac_actor_objective_mode="deterministic_mean",
        )
        agent = SACAgent(3, env.action_space.low, env.action_space.high, cfg, device="cpu")
        observations = torch.randn((8, 3))
        data = ReplayBufferSamples(
            observations=observations,
            actions=torch.zeros((8, 1)),
            next_observations=observations,
            dones=torch.zeros((8, 1)),
            rewards=torch.zeros((8, 1)),
        )
        alpha_before = agent.alpha
        std_before = {
            name: parameter.detach().clone()
            for name, parameter in agent.actor.named_parameters()
            if name.startswith("std_") or name == "std_bias"
        }

        metrics = agent.update(data, update_step=1)

        assert metrics["sac_actor_objective_is_deterministic_mean"] == pytest.approx(1.0)
        assert agent.alpha == pytest.approx(alpha_before)
        for name, parameter in agent.actor.named_parameters():
            if name in std_before:
                torch.testing.assert_close(parameter, std_before[name])
    finally:
        env.close()


def test_actor_mean_logit_l2_regularizer_is_default_off_and_only_logs():
    cfg = SACConfig(
        batch_size=4,
        device="cpu",
        policy_frequency=1,
        actor_updates_per_trigger=1,
        update_diagnostics=False,
    )
    assert cfg.actor_mean_logit_l2_weight == 0.0
    agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        cfg,
        device="cpu",
    )
    data = ReplayBufferSamples(
        observations=torch.zeros((4, 3)),
        actions=torch.zeros((4, 1)),
        next_observations=torch.zeros((4, 3)),
        dones=torch.zeros((4, 1)),
        rewards=torch.zeros((4, 1)),
    )

    metrics = agent.update(data, update_step=1)

    assert metrics["actor_mean_logit_l2_weight"] == 0.0
    assert "actor_mean_logit_l2_penalty_raw" in metrics
    assert "actor_mean_logit_abs_mean" in metrics
    assert "actor_deterministic_action_saturation_fraction" in metrics
    assert "actor_mean_tanh_derivative_mean" in metrics
    assert metrics["actor_loss"] == pytest.approx(
        metrics["sac_actor_loss"], rel=1e-6, abs=1e-7
    )


def test_simba_actor_log_std_floor_is_configured_and_telemetried():
    cfg = SACConfig(
        batch_size=4,
        device="cpu",
        policy_frequency=1,
        actor_updates_per_trigger=1,
        update_diagnostics=False,
        sac_actor_loss_weight=0.0,
        simba_backbone=True,
        simba_actor_hidden_dim=16,
        simba_critic_hidden_dim=16,
        simba_actor_log_std_floor=-1.5,
    )
    agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        cfg,
        device="cpu",
    )
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.zero_()
        agent.actor.std_bias.fill_(-100.0)
        _mean, log_std = agent.actor(torch.zeros((4, 3)))
    torch.testing.assert_close(log_std, torch.full_like(log_std, -1.5))

    data = ReplayBufferSamples(
        observations=torch.zeros((4, 3)),
        actions=torch.zeros((4, 1)),
        next_observations=torch.zeros((4, 3)),
        dones=torch.zeros((4, 1)),
        rewards=torch.zeros((4, 1)),
    )
    metrics = agent.update(data, update_step=1)
    assert metrics["actor_log_std_effective_floor"] == pytest.approx(-1.5)
    assert metrics["actor_log_std_min"] == pytest.approx(-1.5)
    assert metrics["actor_log_std_below_minus_1_fraction"] == pytest.approx(1.0)
    assert metrics["actor_log_std_below_minus_1p5_fraction"] == pytest.approx(0.0)
    assert metrics["actor_unclamped_log_std_min"] == pytest.approx(-10.0)
    assert metrics[
        "actor_unclamped_log_std_below_minus_1p5_fraction"
    ] == pytest.approx(1.0)


def test_simba_actor_log_std_floor_clamps_after_unchanged_historical_mapping():
    common = dict(
        simba_backbone=True,
        simba_actor_hidden_dim=16,
        simba_critic_hidden_dim=16,
    )
    default_agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        SACConfig(**common),
        device="cpu",
    )
    floored_agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        SACConfig(**common, simba_actor_log_std_floor=-1.5),
        device="cpu",
    )
    floored_agent.actor.load_state_dict(default_agent.actor.state_dict())
    observations = torch.randn((32, 3))
    with torch.no_grad():
        default_mean, default_log_std = default_agent.actor(observations)
        floor_mean, floor_log_std, floor_unclamped = (
            floored_agent.actor.forward_with_unclamped_log_std(observations)
        )
    torch.testing.assert_close(floor_mean, default_mean)
    torch.testing.assert_close(floor_unclamped, default_log_std)
    torch.testing.assert_close(
        floor_log_std, torch.clamp(default_log_std, min=-1.5)
    )


@pytest.mark.parametrize("simba_backbone", [False, True])
def test_actor_mean_logit_l2_regularizer_restores_saturated_mean(simba_backbone):
    cfg = SACConfig(
        batch_size=4,
        device="cpu",
        policy_frequency=1,
        actor_updates_per_trigger=1,
        update_diagnostics=False,
        sac_actor_loss_weight=0.0,
        actor_mean_logit_l2_weight=0.1,
        simba_backbone=simba_backbone,
        simba_actor_hidden_dim=16,
        simba_critic_hidden_dim=16,
    )
    agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        cfg,
        device="cpu",
    )
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.zero_()
        mean_bias = agent.actor.mean_bias if simba_backbone else agent.actor.fc_mean.bias
        mean_bias.fill_(8.0)
    data = ReplayBufferSamples(
        observations=torch.zeros((4, 3)),
        actions=torch.zeros((4, 1)),
        next_observations=torch.zeros((4, 3)),
        dones=torch.zeros((4, 1)),
        rewards=torch.zeros((4, 1)),
    )
    bias_before = float(mean_bias.detach().item())

    metrics = agent.update(data, update_step=1)

    assert mean_bias.grad is not None
    assert float(mean_bias.grad.item()) > 0.0
    assert 0.0 < float(mean_bias.detach().item()) < bias_before
    assert metrics["actor_mean_logit_l2_penalty_raw"] == pytest.approx(64.0)
    assert metrics["actor_mean_logit_abs_mean"] == pytest.approx(8.0)
    assert metrics["actor_deterministic_action_saturation_fraction"] == pytest.approx(1.0)
    assert metrics["actor_mean_tanh_derivative_mean"] < 1e-5
    assert metrics["actor_loss"] == pytest.approx(6.4, rel=1e-6)


@pytest.mark.parametrize("simba_backbone", [False, True])
def test_actor_mean_logit_excess_regularizer_preserves_bounded_logits_and_caps_extremes(
    simba_backbone,
):
    cfg = SACConfig(
        batch_size=4,
        device="cpu",
        policy_frequency=1,
        actor_updates_per_trigger=1,
        update_diagnostics=False,
        sac_actor_loss_weight=0.0,
        actor_mean_logit_l2_weight=0.1,
        actor_mean_logit_excess_threshold=4.0,
        simba_backbone=simba_backbone,
        simba_actor_hidden_dim=16,
        simba_critic_hidden_dim=16,
    )
    agent = SACAgent(
        3,
        np.asarray([-2.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
        cfg,
        device="cpu",
    )
    data = ReplayBufferSamples(
        observations=torch.zeros((4, 3)),
        actions=torch.zeros((4, 1)),
        next_observations=torch.zeros((4, 3)),
        dones=torch.zeros((4, 1)),
        rewards=torch.zeros((4, 1)),
    )
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.zero_()
        mean_bias = agent.actor.mean_bias if simba_backbone else agent.actor.fc_mean.bias
        mean_bias.fill_(3.0)
    below_before = float(mean_bias.detach().item())
    below_metrics = agent.update(data, update_step=1)
    assert float(mean_bias.detach().item()) == pytest.approx(below_before)
    assert below_metrics["actor_mean_logit_l2_penalty_raw"] == pytest.approx(0.0)
    assert below_metrics["actor_mean_logit_excess_fraction"] == pytest.approx(0.0)

    with torch.no_grad():
        mean_bias.fill_(8.0)
    above_before = float(mean_bias.detach().item())
    above_metrics = agent.update(data, update_step=2)
    assert 4.0 < float(mean_bias.detach().item()) < above_before
    assert above_metrics["actor_mean_logit_l2_penalty_raw"] == pytest.approx(16.0)
    assert above_metrics["actor_mean_logit_excess_fraction"] == pytest.approx(1.0)
    assert above_metrics["actor_loss"] == pytest.approx(1.6, rel=1e-6)


def test_joint_actor_gradient_balance_scales_only_sac_term():
    env = make_env("Pendulum-v1", seed=0)
    try:
        cfg = SACConfig(
            buffer_size=16,
            learning_starts=1,
            batch_size=4,
            device="cpu",
            reference_auxiliary_mode="bc",
            reference_auxiliary_weight=0.5,
            sac_actor_loss_weight=2.0,
            sac_actor_gradient_balance_mode="match_reference",
            sac_actor_gradient_balance_min_multiplier=0.25,
            sac_actor_gradient_balance_max_multiplier=0.25,
        )
        agent = SACAgent(3, env.action_space.low, env.action_space.high, cfg, device="cpu")
        data = ReplayBufferSamples(
            observations=torch.zeros((4, 3)),
            actions=torch.zeros((4, 1)),
            next_observations=torch.zeros((4, 3)),
            dones=torch.zeros((4, 1)),
            rewards=torch.zeros((4, 1)),
        )
        metrics = agent.update(
            data,
            update_step=2,
            reference_actions=np.ones((4, 1), dtype=np.float32),
        )

        assert metrics["actor_sac_gradient_balance_enabled"] == pytest.approx(1.0)
        assert metrics["actor_sac_gradient_balance_active"] == pytest.approx(1.0)
        assert metrics["actor_sac_gradient_balance_raw_ratio"] > 0.0
        assert metrics["actor_sac_gradient_balance_multiplier"] == pytest.approx(0.25)
        assert metrics["sac_actor_loss_base_weight"] == pytest.approx(2.0)
        assert metrics["sac_actor_loss_weight"] == pytest.approx(0.5)
        assert "actor_sac_bc_gradient_cosine" in metrics
        assert metrics["actor_loss"] == pytest.approx(
            0.5 * metrics["sac_actor_loss"]
            + 0.5 * metrics["reference_actor_bc_loss"],
            rel=1e-5,
            abs=1e-5,
        )
    finally:
        env.close()


def test_joint_actor_gradient_balance_unclipped_matches_weighted_gradient_norms():
    env = make_env("Pendulum-v1", seed=0)
    try:
        torch.manual_seed(7)
        cfg = SACConfig(
            buffer_size=16,
            learning_starts=1,
            batch_size=8,
            device="cpu",
            reference_auxiliary_mode="bc",
            reference_auxiliary_weight=3.0,
            sac_actor_loss_weight=0.2,
            sac_actor_gradient_balance_mode="match_reference",
            sac_actor_gradient_balance_min_multiplier=0.0,
            sac_actor_gradient_balance_max_multiplier=1e6,
        )
        agent = SACAgent(3, env.action_space.low, env.action_space.high, cfg, device="cpu")
        observations = torch.randn((8, 3))
        data = ReplayBufferSamples(
            observations=observations,
            actions=torch.zeros((8, 1)),
            next_observations=observations,
            dones=torch.zeros((8, 1)),
            rewards=torch.zeros((8, 1)),
        )
        metrics = agent.update(
            data,
            update_step=2,
            reference_actions=np.linspace(-2.0, 2.0, 8, dtype=np.float32).reshape(-1, 1),
        )

        assert metrics["actor_sac_gradient_balance_active"] == pytest.approx(1.0)
        assert metrics["actor_sac_gradient_balance_multiplier"] == pytest.approx(
            metrics["actor_sac_gradient_balance_raw_ratio"]
        )
        assert metrics["sac_actor_loss_weight"] == pytest.approx(
            metrics["sac_actor_loss_base_weight"]
            * metrics["actor_sac_gradient_balance_raw_ratio"]
        )
        assert metrics["actor_weighted_sac_gradient_norm"] == pytest.approx(
            metrics["actor_weighted_bc_gradient_norm"], rel=1e-6, abs=1e-8
        )
    finally:
        env.close()


def test_asymmetric_actor_projection_is_identity_for_positive_cosine():
    sac_gradients = [torch.tensor([1.0, 2.0]), torch.tensor([-0.5])]
    bc_gradients = [torch.tensor([2.0, 1.0]), torch.tensor([-1.0])]

    projected, metrics = SACAgent._project_sac_gradient_off_bc(
        sac_gradients,
        bc_gradients,
    )

    for actual, expected in zip(projected, sac_gradients):
        assert actual is not None
        torch.testing.assert_close(actual, expected)
    assert metrics["actor_sac_bc_weighted_gradient_cosine_before"] > 0.0
    assert metrics["actor_sac_bc_projection_applied"] == pytest.approx(0.0)
    assert metrics["actor_sac_bc_projection_correction_norm"] == pytest.approx(0.0)


def test_asymmetric_actor_projection_removes_only_conflicting_sac_component():
    sac_gradients = [torch.tensor([1.0, -2.0]), None]
    bc_gradients = [torch.tensor([-1.0, 0.0]), torch.tensor([3.0])]
    bc_before = [gradient.clone() for gradient in bc_gradients]

    projected, metrics = SACAgent._project_sac_gradient_off_bc(
        sac_gradients,
        bc_gradients,
    )

    assert metrics["actor_sac_bc_weighted_gradient_dot_before"] == pytest.approx(-1.0)
    assert metrics["actor_sac_bc_projection_applied"] == pytest.approx(1.0)
    assert metrics["actor_sac_bc_weighted_gradient_dot_after"] == pytest.approx(
        0.0, abs=1e-7
    )
    flattened_projected = torch.cat(
        [gradient.reshape(-1) for gradient in projected if gradient is not None]
    )
    flattened_bc = torch.cat([gradient.reshape(-1) for gradient in bc_gradients])
    assert torch.dot(flattened_projected, flattened_bc) == pytest.approx(0.0, abs=1e-7)
    # The authoritative BC gradient is an input only and must never be mutated.
    for actual, expected in zip(bc_gradients, bc_before):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(projected[0], torch.tensor([0.9, -2.0]))
    torch.testing.assert_close(projected[1], torch.tensor([0.3]))


def test_asymmetric_actor_projection_is_identity_without_bc_gradient():
    sac_gradients = [torch.tensor([1.0, -2.0]), torch.tensor([0.25])]

    projected, metrics = SACAgent._project_sac_gradient_off_bc(
        sac_gradients,
        [None, None],
    )

    for actual, expected in zip(projected, sac_gradients):
        assert actual is not None
        torch.testing.assert_close(actual, expected)
    assert metrics["actor_sac_bc_projection_applied"] == pytest.approx(0.0)
    assert metrics["actor_sac_bc_weighted_gradient_dot_before"] == pytest.approx(0.0)
    assert metrics["actor_sac_bc_weighted_gradient_dot_after"] == pytest.approx(0.0)


def test_actor_projection_and_adaptive_gradient_balance_share_one_optimizer_step(monkeypatch):
    env = make_env("Pendulum-v1", seed=0)
    try:
        torch.manual_seed(29)
        cfg = SACConfig(
            buffer_size=16,
            learning_starts=1,
            batch_size=8,
            device="cpu",
            policy_frequency=1,
            actor_updates_per_trigger=1,
            reference_auxiliary_mode="bc",
            reference_auxiliary_weight=1.0,
            sac_actor_gradient_balance_mode="match_reference",
            sac_actor_gradient_balance_min_multiplier=0.1,
            sac_actor_gradient_balance_max_multiplier=10.0,
            sac_actor_gradient_conflict_mode="project_sac",
        )
        agent = SACAgent(3, env.action_space.low, env.action_space.high, cfg, device="cpu")
        observations = torch.randn((8, 3))
        data = ReplayBufferSamples(
            observations=observations,
            actions=torch.zeros((8, 1)),
            next_observations=observations,
            dones=torch.zeros((8, 1)),
            rewards=torch.zeros((8, 1)),
        )
        optimizer_steps = 0
        original_step = agent.actor_optimizer.step

        def counted_step(*args, **kwargs):
            nonlocal optimizer_steps
            optimizer_steps += 1
            return original_step(*args, **kwargs)

        monkeypatch.setattr(agent.actor_optimizer, "step", counted_step)
        metrics = agent.update(
            data,
            update_step=2,
            reference_actions=np.linspace(-2.0, 2.0, 8, dtype=np.float32).reshape(-1, 1),
        )

        assert optimizer_steps == 1
        assert metrics["actor_sac_gradient_balance_active"] == pytest.approx(1.0)
        assert metrics["actor_sac_bc_projection_enabled"] == pytest.approx(1.0)
        assert metrics["actor_sac_bc_projection_joint_active"] == pytest.approx(1.0)
        if metrics["actor_sac_bc_projection_applied"]:
            assert metrics["actor_sac_bc_weighted_gradient_dot_after"] == pytest.approx(
                0.0, abs=1e-5
            )
    finally:
        env.close()


def test_actor_gradient_balance_is_identity_for_zero_bc_gradient():
    env = make_env("Pendulum-v1", seed=0)
    try:
        cfg = SACConfig(
            buffer_size=16,
            learning_starts=1,
            batch_size=4,
            device="cpu",
            actor_updates_per_trigger=1,
            reference_auxiliary_mode="bc",
            reference_auxiliary_weight=1.0,
            sac_actor_gradient_balance_mode="match_reference",
            sac_actor_gradient_balance_min_multiplier=0.0,
            sac_actor_gradient_balance_max_multiplier=4.0,
        )
        agent = SACAgent(3, env.action_space.low, env.action_space.high, cfg, device="cpu")
        data = ReplayBufferSamples(
            observations=torch.zeros((4, 3)),
            actions=torch.zeros((4, 1)),
            next_observations=torch.zeros((4, 3)),
            dones=torch.zeros((4, 1)),
            rewards=torch.zeros((4, 1)),
        )
        with torch.no_grad():
            normalized_observations = agent._normalize_obs_tensor(data.observations)
            _, _, reference_actions = agent.actor.get_action(normalized_observations)
        metrics = agent.update(
            data,
            update_step=2,
            reference_actions=reference_actions.cpu().numpy(),
        )

        assert metrics["reference_actor_bc_loss"] == pytest.approx(0.0, abs=1e-12)
        assert metrics["actor_weighted_bc_gradient_norm"] == pytest.approx(0.0, abs=1e-12)
        assert metrics["actor_sac_gradient_balance_active"] == pytest.approx(0.0)
        assert metrics["actor_sac_gradient_balance_raw_ratio"] == pytest.approx(0.0)
        assert metrics["actor_sac_gradient_balance_multiplier"] == pytest.approx(1.0)
        assert metrics["sac_actor_loss_weight"] == pytest.approx(
            metrics["sac_actor_loss_base_weight"]
        )
    finally:
        env.close()


def test_actor_gradient_balance_is_identity_without_reference_loss():
    env = make_env("Pendulum-v1", seed=0)
    try:
        cfg = SACConfig(
            buffer_size=16,
            learning_starts=1,
            batch_size=4,
            device="cpu",
            sac_actor_gradient_balance_mode="match_reference",
            sac_actor_gradient_balance_min_multiplier=0.0,
            sac_actor_gradient_balance_max_multiplier=4.0,
        )
        agent = SACAgent(3, env.action_space.low, env.action_space.high, cfg, device="cpu")
        data = ReplayBufferSamples(
            observations=torch.zeros((4, 3)),
            actions=torch.zeros((4, 1)),
            next_observations=torch.zeros((4, 3)),
            dones=torch.zeros((4, 1)),
            rewards=torch.zeros((4, 1)),
        )
        metrics = agent.update(data, update_step=2)

        assert metrics["actor_sac_gradient_balance_active"] == pytest.approx(0.0)
        assert metrics["actor_sac_gradient_balance_raw_ratio"] == pytest.approx(0.0)
        assert metrics["actor_sac_gradient_balance_multiplier"] == pytest.approx(1.0)
        assert metrics["sac_actor_loss_weight"] == pytest.approx(1.0)
        assert metrics["actor_loss"] == pytest.approx(metrics["sac_actor_loss"])
    finally:
        env.close()


def test_increasing_reference_weight_schedule_enables_bc_after_zero_initial_weight():
    env = make_env("Pendulum-v1", seed=0)
    try:
        cfg = SACConfig(
            buffer_size=16,
            learning_starts=1,
            batch_size=4,
            device="cpu",
            reference_auxiliary_mode="bc",
            reference_auxiliary_weight=0.0,
            reference_auxiliary_weight_final=1.0,
            reference_auxiliary_decay_updates=10,
            sac_actor_loss_weight=0.0,
        )
        agent = SACAgent(3, env.action_space.low, env.action_space.high, cfg, device="cpu")
        data = ReplayBufferSamples(
            observations=torch.zeros((4, 3)),
            actions=torch.zeros((4, 1)),
            next_observations=torch.zeros((4, 3)),
            dones=torch.zeros((4, 1)),
            rewards=torch.zeros((4, 1)),
        )
        metrics = agent.update(
            data,
            update_step=4,
            reference_actions=np.ones((4, 1), dtype=np.float32),
        )

        assert metrics["reference_actor_loss_weight"] == pytest.approx(0.4)
        assert metrics["reference_actor_bc_loss"] > 0.0
        assert metrics["actor_loss"] == pytest.approx(
            0.4 * metrics["reference_actor_bc_loss"], rel=1e-5, abs=1e-5
        )
    finally:
        env.close()


def test_sac_actor_filter_can_use_reference_actions_without_bc_loss():
    env = make_env("Pendulum-v1", seed=0)
    try:
        cfg = SACConfig(
            buffer_size=16,
            learning_starts=1,
            batch_size=4,
            device="cpu",
            reference_auxiliary_mode="none",
            reference_auxiliary_weight=0.0,
            sac_actor_filter_mode="reference_online_unanimous",
        )
        agent = SACAgent(3, env.action_space.low, env.action_space.high, cfg, device="cpu")
        data = ReplayBufferSamples(
            observations=torch.zeros((4, 3)),
            actions=torch.zeros((4, 1)),
            next_observations=torch.zeros((4, 3)),
            dones=torch.zeros((4, 1)),
            rewards=torch.zeros((4, 1)),
        )
        metrics = agent.update(
            data,
            update_step=2,
            reference_actions=np.zeros((4, 1), dtype=np.float32),
        )

        assert "sac_actor_filter_selected_fraction" in metrics
        assert "reference_actor_bc_loss" not in metrics
        assert metrics["reference_actor_loss_weight"] == pytest.approx(0.0)
    finally:
        env.close()


def test_reference_unanimous_filter_vetoes_disagreeing_critic():
    env = make_env("Pendulum-v1", seed=0)
    try:
        cfg = SACConfig(
            buffer_size=16,
            learning_starts=1,
            batch_size=2,
            device="cpu",
            sac_actor_filter_mode="reference_online_unanimous",
        )
        agent = SACAgent(3, env.action_space.low, env.action_space.high, cfg, device="cpu")
        observations = torch.zeros((2, 3))
        policy_mean = torch.ones((2, 1))
        reference_actions = torch.zeros((2, 1))
        per_sample_loss = torch.tensor([[1.0], [2.0]], requires_grad=True)

        agent.q1.forward = lambda _obs, actions: actions[:, :1]
        agent.q2.forward = lambda _obs, actions: actions[:, :1]
        loss, selected, metrics = agent._filter_sac_actor_loss(
            observations=observations,
            policy_mean=policy_mean,
            reference_actions=reference_actions,
            per_sample_loss=per_sample_loss,
        )
        assert selected
        assert float(loss.detach()) == pytest.approx(1.5)
        assert metrics["sac_actor_filter_selected_fraction"] == pytest.approx(1.0)

        agent.q2.forward = lambda _obs, actions: -actions[:, :1]
        loss, selected, metrics = agent._filter_sac_actor_loss(
            observations=observations,
            policy_mean=policy_mean,
            reference_actions=reference_actions,
            per_sample_loss=per_sample_loss,
        )
        assert not selected
        assert float(loss.detach()) == pytest.approx(0.0)
        assert metrics["sac_actor_filter_selected_fraction"] == pytest.approx(0.0)
    finally:
        env.close()


def test_self_imitation_actor_loss_logs_metrics():
    env = make_env("Pendulum-v1", seed=0)
    try:
        cfg = SACConfig(
            buffer_size=32,
            learning_starts=4,
            batch_size=4,
            device="cpu",
            self_imitation_weight=0.25,
        )
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        agent = SACAgent(obs_dim, env.action_space.low, env.action_space.high, cfg, device="cpu")
        replay = InstrumentedReplayBuffer(
            32,
            env.observation_space,
            env.action_space,
            device="cpu",
            n_envs=1,
            handle_timeout_termination=False,
        )

        obs, _ = env.reset(seed=0)
        for step in range(8):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            replay.add(
                obs.reshape(1, obs_dim),
                next_obs.reshape(1, obs_dim),
                action.reshape(1, action_dim),
                np.asarray([reward], dtype=np.float32),
                np.asarray([terminated]),
                [{}],
                step=step,
                episode_id=0,
            )
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()

        batch = replay.sample(batch_size=4)
        metrics = agent.update(batch, update_step=2)

        assert "self_imitation_actor_loss" in metrics
        assert "self_imitation_q_advantage_mean" in metrics
        assert metrics["self_imitation_loss_type_is_mse"] == pytest.approx(1.0)
        assert 0.0 <= metrics["self_imitation_mask_fraction"] <= 1.0
        assert metrics["self_imitation_weight_max"] <= cfg.self_imitation_max_weight
        assert metrics["actor_loss"] == pytest.approx(
            metrics["sac_actor_loss"] + 0.25 * metrics["self_imitation_actor_loss"],
            rel=1e-5,
            abs=1e-5,
        )
    finally:
        env.close()


def test_reference_critic_margin_loss_logs_metrics():
    env = make_env("Pendulum-v1", seed=0)
    try:
        cfg = SACConfig(
            buffer_size=32,
            learning_starts=4,
            batch_size=4,
            device="cpu",
            reference_critic_mode="margin",
            reference_critic_weight=0.5,
            reference_critic_margin=0.02,
            update_diagnostics=False,
        )
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        agent = SACAgent(obs_dim, env.action_space.low, env.action_space.high, cfg, device="cpu")
        replay = InstrumentedReplayBuffer(
            32,
            env.observation_space,
            env.action_space,
            device="cpu",
            n_envs=1,
            handle_timeout_termination=False,
        )

        obs, _ = env.reset(seed=0)
        for step in range(8):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            replay.add(
                obs.reshape(1, obs_dim),
                next_obs.reshape(1, obs_dim),
                action.reshape(1, action_dim),
                np.asarray([reward], dtype=np.float32),
                np.asarray([terminated]),
                [{}],
                step=step,
                episode_id=0,
            )
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()

        batch = replay.sample(batch_size=4)
        reference_actions = np.zeros((4, action_dim), dtype=np.float32)
        metrics = agent.update(batch, update_step=2, reference_critic_actions=reference_actions)

        assert "reference_critic_margin_loss" in metrics
        assert "reference_critic_q_advantage_mean" in metrics
        assert "reference_critic_margin_violation_fraction" in metrics
        assert 0.0 <= metrics["reference_critic_margin_violation_fraction"] <= 1.0
        assert metrics["q_loss"] == pytest.approx(
            metrics["q1_loss"] + metrics["q2_loss"],
            rel=1e-6,
            abs=1e-6,
        )
    finally:
        env.close()


def test_simba_scalers_match_official_reference_wrapper():
    cfg = SACConfig(
        device="cpu",
        simba_backbone=True,
        simba_actor_hidden_dim=32,
        simba_critic_hidden_dim=64,
        simba_distributional_critic=True,
        simba_critic_num_bins=11,
    )

    agent = SACAgent(3, np.asarray([-2.0], dtype=np.float32), np.asarray([2.0], dtype=np.float32), cfg, device="cpu")

    actor_expected = np.sqrt(2.0 / cfg.simba_actor_hidden_dim)
    critic_expected = np.sqrt(2.0 / cfg.simba_critic_hidden_dim)
    assert torch.allclose(agent.actor.backbone.embedder.scaler.scaler, torch.full((32,), actor_expected))
    assert torch.allclose(agent.q1.backbone.embedder.scaler.scaler, torch.full((64,), critic_expected))
    assert torch.allclose(agent.actor.mean_scaler.scaler, torch.ones(32))
    assert torch.allclose(agent.actor.std_scaler.scaler, torch.ones(32))
    assert agent.actor.mean_scaler.forward_scaler == pytest.approx(1.0)
    assert agent.actor.std_scaler.forward_scaler == pytest.approx(1.0)
    assert torch.allclose(agent.q1.value_scaler.scaler, torch.ones(64))
    assert torch.allclose(agent.q2.value_scaler.scaler, torch.ones(64))
    assert agent.q1.value_scaler.forward_scaler == pytest.approx(1.0)
    assert agent.q2.value_scaler.forward_scaler == pytest.approx(1.0)


def test_redo_recycles_dormant_cleanrl_critic_units():
    cfg = SACConfig(device="cpu", redo_interval_updates=1, redo_dormant_threshold=0.1)
    agent = SACAgent(3, np.asarray([-2.0], dtype=np.float32), np.asarray([2.0], dtype=np.float32), cfg, device="cpu")
    obs = torch.zeros(8, 3)
    action = torch.zeros(8, 1)
    with torch.no_grad():
        agent.q1.fc1.weight.zero_()
        agent.q1.fc1.bias.zero_()
        agent.q1_target.fc1.weight.fill_(7.0)
        agent.q1_target.fc1.bias.fill_(7.0)

    metrics = agent._redo_softqnetwork(agent.q1, agent.q1_target, obs, action, prefix="q1")

    assert metrics["redo_q1_fc1_units"] == 256.0
    assert not torch.allclose(agent.q1.fc1.weight, torch.zeros_like(agent.q1.fc1.weight))
    assert torch.allclose(agent.q1_target.fc1.weight, agent.q1.fc1.weight)
    assert torch.allclose(agent.q1_target.fc1.bias, agent.q1.fc1.bias)
    assert torch.allclose(agent.q1.fc2.weight, torch.zeros_like(agent.q1.fc2.weight))


def test_redo_recycles_dormant_simba_critic_units():
    cfg = SACConfig(
        device="cpu",
        simba_backbone=True,
        simba_actor_hidden_dim=16,
        simba_critic_hidden_dim=16,
        redo_interval_updates=1,
        redo_dormant_threshold=0.1,
    )
    agent = SACAgent(3, np.asarray([-2.0], dtype=np.float32), np.asarray([2.0], dtype=np.float32), cfg, device="cpu")
    obs = torch.zeros(8, 3)
    action = torch.zeros(8, 1)
    with torch.no_grad():
        agent.q1.value_w1.weight.zero_()
        agent.q1.value_scaler.scaler.fill_(3.0)
        agent.q1_target.value_w1.weight.fill_(7.0)
        agent.q1_target.value_w2.weight.fill_(7.0)

    metrics = agent._redo_simba_qnetwork(agent.q1, agent.q1_target, obs, action, prefix="q1")

    assert metrics["redo_q1_value_units"] == 16.0
    assert not torch.allclose(agent.q1.value_w1.weight, torch.zeros_like(agent.q1.value_w1.weight))
    assert torch.allclose(agent.q1_target.value_w1.weight, agent.q1.value_w1.weight)
    assert torch.allclose(agent.q1.value_w2.weight, torch.zeros_like(agent.q1.value_w2.weight))
    assert torch.allclose(agent.q1_target.value_w2.weight, torch.zeros_like(agent.q1_target.value_w2.weight))
    assert torch.allclose(agent.q1.value_scaler.scaler, torch.ones_like(agent.q1.value_scaler.scaler))


def test_simba_observation_normalizer_updates_and_roundtrips(tmp_path):
    cfg = SACConfig(
        device="cpu",
        simba_backbone=True,
        simba_actor_hidden_dim=16,
        simba_critic_hidden_dim=16,
        simba_reward_scaling=True,
    )
    agent = SACAgent(3, np.asarray([-2.0], dtype=np.float32), np.asarray([2.0], dtype=np.float32), cfg, device="cpu")
    agent.observe(np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    agent.observe_reward(-1.0, False)
    path = tmp_path / "agent.pt"
    agent.save_checkpoint(path)

    loaded = SACAgent(3, np.asarray([-2.0], dtype=np.float32), np.asarray([2.0], dtype=np.float32), cfg, device="cpu")
    loaded.load_checkpoint(path)

    assert loaded.obs_rms is not None
    assert agent.obs_rms is not None
    assert np.allclose(loaded.obs_rms.mean, agent.obs_rms.mean)
    assert np.allclose(loaded.obs_rms.var, agent.obs_rms.var)
    assert loaded.obs_rms.count == pytest.approx(agent.obs_rms.count)
    assert loaded.reward_scaler is not None
    assert agent.reward_scaler is not None
    assert loaded.reward_scaler.discounted_return == pytest.approx(agent.reward_scaler.discounted_return)


def test_l2_feature_norm_agent_uses_archived_hidden_scale():
    env = make_env("Pendulum-v1", seed=0)
    try:
        cfg = SACConfig(l2_feature_norm=True, device="cpu")
        agent = SACAgent(3, env.action_space.low, env.action_space.high, cfg, device="cpu")
        observation = torch.tensor([[1.0, -0.25, 0.5], [-0.5, 0.75, -1.0]])

        first = torch.relu(agent.actor.fc1(observation))
        normalized = torch.nn.functional.normalize(first, p=2.0, dim=1) * math.sqrt(first.shape[1])

        assert torch.allclose(torch.linalg.vector_norm(normalized, dim=1), torch.full((2,), 16.0))
        action = agent.act(observation[0].numpy(), deterministic=True)
        assert action.shape == env.action_space.shape
    finally:
        env.close()
