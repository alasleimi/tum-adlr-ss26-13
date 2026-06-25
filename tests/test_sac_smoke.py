import numpy as np
import pytest
import torch

from last_nine_rl.config import SACConfig
from last_nine_rl.envs import make_env
from last_nine_rl.replay import InstrumentedReplayBuffer
from last_nine_rl.sac import SACAgent
from last_nine_rl.simba_v2 import SimbaHyperDense, project_simba_weights_to_unit_norm


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
    finally:
        env.close()


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
        metrics = agent.update(batch, update_step=2, reference_actions=reference_actions)

        assert "reference_actor_bc_loss" in metrics
        assert "reference_actor_q_advantage_mean" in metrics
        assert metrics["reference_actor_bc_mask_fraction"] == pytest.approx(1.0)
        assert metrics["actor_loss"] == pytest.approx(
            metrics["sac_actor_loss"] + 0.5 * metrics["reference_actor_bc_loss"],
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
