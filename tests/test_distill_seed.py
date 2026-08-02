from __future__ import annotations

import random

import numpy as np
import torch

from last_nine_rl.distill_reference import seed_training_rngs


def test_distilled_actor_seed_covers_python_numpy_and_torch() -> None:
    seed_training_rngs(731)
    first = (random.random(), np.random.random(), torch.rand(4))
    seed_training_rngs(731)
    second = (random.random(), np.random.random(), torch.rand(4))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
