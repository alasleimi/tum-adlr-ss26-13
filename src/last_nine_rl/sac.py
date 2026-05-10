from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from cleanrl.sac_continuous_action import Actor, SoftQNetwork

from last_nine_rl.config import SACConfig
from last_nine_rl.metrics import (
    activation_statistics,
    gradient_list_norm,
    parameter_delta_norm,
    parameter_list_norm,
    gradient_norm,
    parameter_norm,
    snapshot_parameters,
)


@dataclass
class _CleanRLEnvSpec:
    single_observation_space: gym.spaces.Box
    single_action_space: gym.spaces.Box


class SACAgent:
    """Thin telemetry wrapper around the vendored CleanRL SAC networks/update."""

    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        cfg: SACConfig,
        device: str,
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self.env_spec = _CleanRLEnvSpec(
            single_observation_space=gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            ),
            single_action_space=gym.spaces.Box(
                low=np.asarray(action_low, dtype=np.float32),
                high=np.asarray(action_high, dtype=np.float32),
                dtype=np.float32,
            ),
        )

        self.actor = Actor(self.env_spec).to(self.device)
        self.q1 = SoftQNetwork(self.env_spec).to(self.device)
        self.q2 = SoftQNetwork(self.env_spec).to(self.device)
        self.q1_target = SoftQNetwork(self.env_spec).to(self.device)
        self.q2_target = SoftQNetwork(self.env_spec).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=cfg.policy_lr)

        self.target_entropy = -torch.prod(torch.Tensor(self.env_spec.single_action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.q_lr)
        self.last_actor_grad_norm = 0.0
        self.last_q_grad_norm = 0.0

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, mean = self.actor.get_action(obs)
        selected = mean if deterministic else action
        return selected.squeeze(0).cpu().numpy()

    def update(self, data: Any, update_step: int) -> dict[str, float]:
        if update_step <= 0:
            raise ValueError("update_step is one-indexed and must be positive.")

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations)
            qf1_next_target = self.q1_target(data.next_observations, next_state_actions)
            qf2_next_target = self.q2_target(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.cfg.gamma * min_qf_next_target.view(-1)

        qf1_a_values = self.q1(data.observations, data.actions).view(-1)
        qf2_a_values = self.q2(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        q_params = list(self.q1.parameters()) + list(self.q2.parameters())
        q_params_before = snapshot_parameters(q_params)
        q_param_norm_before = parameter_list_norm(q_params)
        self.last_q_grad_norm = gradient_list_norm(q_params)
        self.q_optimizer.step()
        q_update_norm = parameter_delta_norm(q_params_before, q_params)

        metrics: dict[str, float] = {
            "q1_loss": float(qf1_loss.detach().cpu()),
            "q2_loss": float(qf2_loss.detach().cpu()),
            "q_loss": float(qf_loss.detach().cpu()),
            "q_loss_cleanrl_logged": float((qf_loss.detach() / 2.0).cpu()),
            "q1_mean": float(qf1_a_values.detach().mean().cpu()),
            "q2_mean": float(qf2_a_values.detach().mean().cpu()),
            "target_q_mean": float(next_q_value.detach().mean().cpu()),
            "alpha": float(self.alpha),
            "q_grad_norm": self.last_q_grad_norm,
        }

        if update_step % self.cfg.policy_frequency == 0:
            for _ in range(self.cfg.policy_frequency):
                pi, log_pi, _ = self.actor.get_action(data.observations)
                qf1_pi = self.q1(data.observations, pi)
                qf2_pi = self.q2(data.observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_params = list(self.actor.parameters())
                actor_params_before = snapshot_parameters(actor_params)
                actor_param_norm_before = parameter_list_norm(actor_params)
                self.last_actor_grad_norm = gradient_norm(self.actor)
                self.actor_optimizer.step()
                actor_update_norm = parameter_delta_norm(actor_params_before, actor_params)

                with torch.no_grad():
                    _, log_pi, _ = self.actor.get_action(data.observations)
                alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()

            metrics.update(
                {
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "alpha_loss": float(alpha_loss.detach().cpu()),
                    "policy_log_prob_mean": float(log_pi.detach().mean().cpu()),
                    "policy_entropy_estimate": float((-log_pi.detach()).mean().cpu()),
                    "actor_grad_norm": self.last_actor_grad_norm,
                    "actor_update_norm": actor_update_norm,
                    "actor_update_norm_ratio": actor_update_norm / max(actor_param_norm_before, 1e-12),
                    "alpha": float(self.alpha),
                }
            )

        if update_step % self.cfg.target_network_frequency == 0:
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)
            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)

        actor_norm = parameter_norm(self.actor)
        q_norm = parameter_norm(self.q1) + parameter_norm(self.q2)
        metrics.update(
            {
                "actor_param_norm": actor_norm,
                "q_param_norm": q_norm,
                "q_update_norm": q_update_norm,
                "q_update_norm_ratio": q_update_norm / max(q_param_norm_before, 1e-12),
            }
        )
        return metrics

    def diagnostics(self, data: Any, dormant_relative_threshold: float) -> dict[str, float]:
        with torch.no_grad():
            actor_acts = self._actor_hidden_activations(data.observations)
            q1_acts = self._q_hidden_activations(self.q1, data.observations, data.actions)
            q2_acts = self._q_hidden_activations(self.q2, data.observations, data.actions)
        out: dict[str, float] = {
            "actor_param_norm": parameter_norm(self.actor),
            "q1_param_norm": parameter_norm(self.q1),
            "q2_param_norm": parameter_norm(self.q2),
            "alpha": float(self.alpha),
        }
        out.update(activation_statistics("actor", actor_acts, dormant_relative_threshold))
        out.update(activation_statistics("q1", q1_acts, dormant_relative_threshold))
        out.update(activation_statistics("q2", q2_acts, dormant_relative_threshold))
        return out

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": "CleanRL sac_continuous_action.py",
            "actor": self.actor.state_dict(),
            "qf1": self.q1.state_dict(),
            "qf2": self.q2.state_dict(),
            "qf1_target": self.q1_target.state_dict(),
            "qf2_target": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha": self.alpha,
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path, load_optimizers: bool = False) -> dict[str, Any]:
        try:
            payload = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older PyTorch.
            payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
        self.q1.load_state_dict(payload["qf1"])
        self.q2.load_state_dict(payload["qf2"])
        self.q1_target.load_state_dict(payload["qf1_target"])
        self.q2_target.load_state_dict(payload["qf2_target"])
        self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
        self.alpha = float(self.log_alpha.exp().item())

        if load_optimizers:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            self.q_optimizer.load_state_dict(payload["q_optimizer"])
            self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])

        self.actor.eval()
        self.q1.eval()
        self.q2.eval()
        self.q1_target.eval()
        self.q2_target.eval()
        return payload

    def _actor_hidden_activations(self, obs: torch.Tensor) -> list[torch.Tensor]:
        first = F.relu(self.actor.fc1(obs))
        second = F.relu(self.actor.fc2(first))
        return [first, second]

    @staticmethod
    def _q_hidden_activations(network: SoftQNetwork, obs: torch.Tensor, action: torch.Tensor) -> list[torch.Tensor]:
        x = torch.cat([obs, action], 1)
        first = F.relu(network.fc1(x))
        second = F.relu(network.fc2(first))
        return [first, second]
