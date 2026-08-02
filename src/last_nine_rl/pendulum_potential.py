from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_DP_GRID_PATH = Path("data/reference/pendulum_dp_grid.csv")
DEFAULT_CONTROLLER_GRID_PATH = Path("data/reference/controller_grid.csv")


class PendulumPotential:
    """State-only Pendulum value potential loaded from the DP/controller evaluation grids."""

    def __init__(
        self,
        source: str = "best",
        dp_grid_path: str | Path | None = None,
        controller_grid_path: str | Path | None = None,
        device: torch.device | str = "cpu",
    ):
        if source not in {"best", "dp_policy", "dp_value", "controller"}:
            raise ValueError("source must be one of best, dp_policy, dp_value, controller")
        self.source = source
        self.dp_grid_path = Path(dp_grid_path) if dp_grid_path else DEFAULT_DP_GRID_PATH
        self.controller_grid_path = Path(controller_grid_path) if controller_grid_path else DEFAULT_CONTROLLER_GRID_PATH
        values, theta_values, velocity_values = self._load_values()
        self.theta_bins = int(theta_values.shape[0])
        self.velocity_bins = int(velocity_values.shape[0])
        self.max_speed = float(max(abs(float(velocity_values[0])), abs(float(velocity_values[-1]))))
        self.values = torch.as_tensor(values, dtype=torch.float32, device=device)

    def to(self, device: torch.device | str) -> "PendulumPotential":
        self.values = self.values.to(device)
        return self

    def query(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] < 3:
            raise ValueError("Pendulum potential expects observations [cos(theta), sin(theta), theta_dot].")
        flat = observations.reshape(-1, observations.shape[-1])
        theta = torch.atan2(flat[:, 1], flat[:, 0])
        theta = torch.remainder(theta + math.pi, 2.0 * math.pi) - math.pi
        velocity = flat[:, 2].clamp(-self.max_speed, self.max_speed)

        theta_step = 2.0 * math.pi / float(self.theta_bins)
        theta_pos = (theta + math.pi) / theta_step
        theta_floor = torch.floor(theta_pos)
        theta0 = torch.remainder(theta_floor.to(torch.long), self.theta_bins)
        theta1 = torch.remainder(theta0 + 1, self.theta_bins)
        theta_frac = (theta_pos - theta_floor).to(self.values.dtype)

        velocity_step = 2.0 * self.max_speed / float(self.velocity_bins - 1)
        velocity_pos = (velocity + self.max_speed) / velocity_step
        velocity0 = torch.floor(velocity_pos).to(torch.long).clamp(0, self.velocity_bins - 2)
        velocity1 = velocity0 + 1
        velocity_frac = (velocity_pos - velocity0.to(velocity_pos.dtype)).to(self.values.dtype).clamp(0.0, 1.0)

        v00 = self.values[velocity0, theta0]
        v01 = self.values[velocity0, theta1]
        v10 = self.values[velocity1, theta0]
        v11 = self.values[velocity1, theta1]
        interpolated = (
            (1.0 - velocity_frac) * (1.0 - theta_frac) * v00
            + (1.0 - velocity_frac) * theta_frac * v01
            + velocity_frac * (1.0 - theta_frac) * v10
            + velocity_frac * theta_frac * v11
        )
        return interpolated.reshape(observations.shape[:-1])

    def _load_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dp_rows = _read_rows(self.dp_grid_path)
        theta_values = np.asarray(sorted({float(row["theta"]) for row in dp_rows}), dtype=np.float64)
        velocity_values = np.asarray(sorted({float(row["theta_dot"]) for row in dp_rows}), dtype=np.float64)
        expected = len(theta_values) * len(velocity_values)
        if expected != len(dp_rows):
            raise ValueError(f"DP grid has {len(dp_rows)} rows but {expected} unique theta/velocity cells.")

        dp_grid = _grid_from_rows(dp_rows, theta_values, velocity_values, "dp_policy_return")
        if self.source == "dp_policy":
            return dp_grid, theta_values, velocity_values
        if self.source == "dp_value":
            return _grid_from_rows(dp_rows, theta_values, velocity_values, "dp_value"), theta_values, velocity_values

        controller_rows = _read_rows(self.controller_grid_path)
        controller_grid = _grid_from_rows(controller_rows, theta_values, velocity_values, "controller_return")
        if self.source == "controller":
            return controller_grid, theta_values, velocity_values
        return np.maximum(dp_grid, controller_grid), theta_values, velocity_values


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Pendulum potential grid not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _grid_from_rows(
    rows: list[dict[str, Any]],
    theta_values: np.ndarray,
    velocity_values: np.ndarray,
    value_key: str,
) -> np.ndarray:
    theta_index = {float(value): idx for idx, value in enumerate(theta_values)}
    velocity_index = {float(value): idx for idx, value in enumerate(velocity_values)}
    grid = np.full((len(velocity_values), len(theta_values)), np.nan, dtype=np.float32)
    for row in rows:
        try:
            v_idx = velocity_index[float(row["theta_dot"])]
            t_idx = theta_index[float(row["theta"])]
            grid[v_idx, t_idx] = float(row[value_key])
        except KeyError as exc:
            raise ValueError(f"Missing {value_key!r} or state column in potential grid row.") from exc
    if not np.isfinite(grid).all():
        raise ValueError(f"Potential grid for {value_key!r} is incomplete or non-finite.")
    return grid
