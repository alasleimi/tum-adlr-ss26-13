import numpy as np

from last_nine_rl.config import EnvConfig, EvalConfig, ExperimentConfig, SACConfig, TelemetryConfig
from last_nine_rl.reference import PendulumEnergySwingupController, evaluate_pendulum_reference


def test_pendulum_reference_controller_is_bounded_and_evaluable(tmp_path):
    controller = PendulumEnergySwingupController()
    action = controller.act(np.asarray([-1.0, 0.0, 0.0], dtype=np.float32))
    assert action.shape == (1,)
    assert -2.0 <= action[0] <= 2.0

    config = ExperimentConfig(
        env=EnvConfig(env_id="Pendulum-v1", max_episode_steps=5),
        sac=SACConfig(total_steps=10, learning_starts=1, batch_size=4, buffer_size=16),
        eval=EvalConfig(episodes=2, seed_base=123),
        telemetry=TelemetryConfig(run_root=str(tmp_path), tensorboard=False),
    )
    result = evaluate_pendulum_reference(config)

    assert result["controller"] == "pendulum_energy_swingup_pd"
    assert result["eval_seeds"] == [123, 124]
    assert "strict_success_rate" in result["metrics"]
    assert "return_reliability_nines_wilson95_low" in result["metrics"]
