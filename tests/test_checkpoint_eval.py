import numpy as np
import gymnasium as gym

from last_nine_rl.checkpoints import load_agent_from_run
from last_nine_rl.config import EnvConfig, EvalConfig, ExperimentConfig, SACConfig, TelemetryConfig
from last_nine_rl.pendulum_grid import pendulum_step_batch, reset_pendulum_to_state, summarize_cells
from last_nine_rl.train import train


def test_checkpoint_roundtrip_from_run(tmp_path):
    run_dir = tmp_path / "run"
    config = ExperimentConfig(
        name="checkpoint_roundtrip",
        seed=0,
        env=EnvConfig(env_id="Pendulum-v1", max_episode_steps=3),
        sac=SACConfig(total_steps=8, buffer_size=16, learning_starts=2, batch_size=4, device="cpu"),
        eval=EvalConfig(every_steps=4, episodes=1, seed_base=321),
        telemetry=TelemetryConfig(
            run_root=str(tmp_path),
            tensorboard=False,
            log_interval_steps=4,
            replay_inspection_interval_steps=0,
            diagnostics_interval_steps=0,
            save_model=True,
        ),
    )
    train(config, run_dir)
    agent, loaded_config, payload = load_agent_from_run(run_dir, device="cpu")

    assert loaded_config.seed == 0
    assert payload["extra"]["global_step"] == 8
    action = agent.act(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), deterministic=True)
    assert action.shape == (1,)


def test_pendulum_grid_cell_summary():
    rows = [
        {"theta": 0.0, "theta_dot": 0.0, "return": -10.0, "return_success": 1.0, "strict_success": 1.0, "near_upright_fraction": 1.0, "not_near_upright_streak": 0.0},
        {"theta": 0.0, "theta_dot": 0.0, "return": -300.0, "return_success": 0.0, "strict_success": 0.0, "near_upright_fraction": 0.1, "not_near_upright_streak": 100.0},
    ]

    summary = summarize_cells(rows, np.asarray([0.0]), np.asarray([0.0]))
    assert summary[0]["return_success_rate"] == 0.5
    assert summary[0]["strict_success_rate"] == 0.5
    assert summary[0]["worst_return"] == -300.0


def test_vectorized_pendulum_step_matches_gym():
    theta = np.asarray([-2.0, -0.25, 1.5], dtype=np.float64)
    theta_dot = np.asarray([-0.5, 0.0, 0.75], dtype=np.float64)
    action = np.asarray([-2.0, 0.4, 1.75], dtype=np.float64)

    next_theta, next_theta_dot, rewards = pendulum_step_batch(theta, theta_dot, action)

    env = gym.make("Pendulum-v1")
    try:
        for idx in range(len(theta)):
            reset_pendulum_to_state(env, float(theta[idx]), float(theta_dot[idx]))
            obs, reward, _terminated, _truncated, _info = env.step(np.asarray([action[idx]], dtype=np.float32))
            np.testing.assert_allclose(obs, [np.cos(next_theta[idx]), np.sin(next_theta[idx]), next_theta_dot[idx]])
            np.testing.assert_allclose(reward, rewards[idx])
    finally:
        env.close()
