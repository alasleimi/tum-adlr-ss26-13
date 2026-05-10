import pytest

from last_nine_rl.config import ExperimentConfig, SACConfig


def test_config_validation_rejects_late_failures():
    config = ExperimentConfig(
        sac=SACConfig(
            total_steps=10,
            learning_starts=10,
            buffer_size=4,
            batch_size=8,
            updates_per_step=0,
        )
    )

    with pytest.raises(ValueError) as exc_info:
        config.validate()

    message = str(exc_info.value)
    assert "sac.batch_size" in message
    assert "sac.learning_starts" in message
    assert "sac.updates_per_step" in message
