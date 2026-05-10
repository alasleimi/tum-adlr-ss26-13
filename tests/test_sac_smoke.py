import numpy as np

from last_nine_rl.config import SACConfig
from last_nine_rl.envs import make_env
from last_nine_rl.replay import InstrumentedReplayBuffer
from last_nine_rl.sac import SACAgent


def test_sac_action_bounds_and_single_update():
    env = make_env("Pendulum-v1", seed=0)
    try:
        cfg = SACConfig(
            buffer_size=32,
            learning_starts=4,
            batch_size=4,
            device="cpu",
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
        action = agent.act(obs, deterministic=False)
        assert action.shape == env.action_space.shape
        assert np.all(action <= env.action_space.high + 1e-5)
        assert np.all(action >= env.action_space.low - 1e-5)

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
        assert metrics["q_loss"] >= 0.0
        assert "actor_loss" in metrics
        assert metrics["alpha"] > 0.0
        assert metrics["q_update_norm_ratio"] > 0.0
    finally:
        env.close()
