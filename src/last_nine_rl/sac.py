from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cleanrl.sac_continuous_action import Actor, SoftQNetwork
from cleanrl_utils.buffers import ReplayBufferSamples

from last_nine_rl.config import SACConfig, needs_actor_reference_actions
from last_nine_rl.metrics import (
    activation_statistics,
    gradient_list_norm,
    parameter_delta_norm,
    parameter_list_norm,
    gradient_norm,
    parameter_norm,
    snapshot_parameters,
)
from last_nine_rl.pendulum_potential import PendulumPotential
from last_nine_rl.replay import SACNReplaySamples
from last_nine_rl.simba_v2 import (
    SimbaCategoricalQNetwork,
    SimbaHyperDense,
    SimbaNormalTanhActor,
    SimbaScalarQNetwork,
    project_simba_weights_to_unit_norm,
)


@dataclass
class _CleanRLEnvSpec:
    single_observation_space: gym.spaces.Box
    single_action_space: gym.spaces.Box


def _l2_normalize_hidden(features: torch.Tensor) -> torch.Tensor:
    """Archived L2-feature transform: unit norm, then restore sqrt(width) scale."""

    return F.normalize(features, p=2.0, dim=1) * math.sqrt(float(features.shape[1]))


class _L2FeatureNormActor(Actor):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = _l2_normalize_hidden(F.relu(self.fc1(x)))
        x = _l2_normalize_hidden(F.relu(self.fc2(x)))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = -5.0 + 0.5 * (2.0 - (-5.0)) * (log_std + 1.0)
        return mean, log_std


class _L2FeatureNormSoftQNetwork(SoftQNetwork):
    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, a], dim=1)
        x = _l2_normalize_hidden(F.relu(self.fc1(x)))
        x = _l2_normalize_hidden(F.relu(self.fc2(x)))
        return self.fc3(x)


class _RunningMeanStd:
    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, value: np.ndarray) -> None:
        batch = np.asarray(value, dtype=np.float64).reshape(-1, *self.mean.shape)
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = int(batch.shape[0])
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        ratio = batch_count / total_count
        new_mean = self.mean + delta * ratio
        old_m2 = self.var * self.count
        batch_m2 = batch_var * batch_count
        new_m2 = old_m2 + batch_m2 + np.square(delta) * self.count * ratio
        self.mean = new_mean
        self.var = new_m2 / total_count
        self.count = float(total_count)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "mean": torch.as_tensor(self.mean, dtype=torch.float64),
            "var": torch.as_tensor(self.var, dtype=torch.float64),
            "count": torch.as_tensor(self.count, dtype=torch.float64),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        mean = state["mean"].detach().cpu().numpy() if isinstance(state["mean"], torch.Tensor) else state["mean"]
        var = state["var"].detach().cpu().numpy() if isinstance(state["var"], torch.Tensor) else state["var"]
        count = state["count"].detach().cpu().item() if isinstance(state["count"], torch.Tensor) else state["count"]
        self.mean = np.asarray(mean, dtype=np.float64)
        self.var = np.asarray(var, dtype=np.float64)
        self.count = float(count)


class _RewardScaler:
    def __init__(self, gamma: float, g_max: float, epsilon: float = 1e-8):
        self.gamma = float(gamma)
        self.g_max = float(g_max)
        self.epsilon = float(epsilon)
        self.discounted_return = 0.0
        self.return_rms = _RunningMeanStd(())
        self.return_abs_max = 0.0

    def update(self, reward: float, done: bool) -> None:
        self.discounted_return = self.gamma * (1.0 - float(done)) * self.discounted_return + float(reward)
        value = np.asarray(self.discounted_return, dtype=np.float64)
        self.return_rms.update(value)
        self.return_abs_max = max(self.return_abs_max, abs(self.discounted_return))

    def scale_tensor(self, rewards: torch.Tensor) -> torch.Tensor:
        var_denominator = float(np.sqrt(self.return_rms.var + self.epsilon))
        max_denominator = self.return_abs_max / self.g_max if self.g_max > 0.0 else 0.0
        denominator = max(var_denominator, max_denominator, self.epsilon)
        return rewards / denominator

    def state_dict(self) -> dict[str, Any]:
        return {
            "discounted_return": torch.as_tensor(self.discounted_return, dtype=torch.float64),
            "return_rms": self.return_rms.state_dict(),
            "return_abs_max": torch.as_tensor(self.return_abs_max, dtype=torch.float64),
            "gamma": torch.as_tensor(self.gamma, dtype=torch.float64),
            "g_max": torch.as_tensor(self.g_max, dtype=torch.float64),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        discounted_return = state["discounted_return"]
        return_abs_max = state["return_abs_max"]
        self.discounted_return = float(
            discounted_return.detach().cpu().item() if isinstance(discounted_return, torch.Tensor) else discounted_return
        )
        self.return_rms.load_state_dict(state["return_rms"])
        self.return_abs_max = float(
            return_abs_max.detach().cpu().item() if isinstance(return_abs_max, torch.Tensor) else return_abs_max
        )


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
        self._collect_update_metrics = True
        if cfg.simba_weight_projection and not cfg.simba_backbone:
            raise ValueError(
                "simba_weight_projection requires simba_backbone because SimbaV2 projection is defined on "
                "bias-free HyperDense weights, not CleanRL Linear layers."
            )
        if cfg.sac_actor_gradient_conflict_mode not in {"none", "project_sac"}:
            raise ValueError(
                "sac_actor_gradient_conflict_mode must be one of none, project_sac"
            )
        if cfg.sac_actor_gradient_conflict_mode != "none":
            incompatible_actor_auxiliaries = {
                "critic_search_actor_weight": cfg.critic_search_actor_weight,
                "self_imitation_weight": cfg.self_imitation_weight,
                "pendulum_actor_symmetry_weight": cfg.pendulum_actor_symmetry_weight,
                "actor_mean_logit_l2_weight": cfg.actor_mean_logit_l2_weight,
            }
            active_auxiliaries = [
                name
                for name, weight in incompatible_actor_auxiliaries.items()
                if float(weight) > 0.0
            ]
            if active_auxiliaries:
                raise ValueError(
                    "sac_actor_gradient_conflict_mode supports exactly SAC and reference BC; "
                    "disable incompatible actor auxiliaries: " + ", ".join(active_auxiliaries)
                )
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
        self._normalize_observations = bool(cfg.simba_backbone and cfg.simba_observation_norm)
        self.obs_rms = _RunningMeanStd((obs_dim,)) if self._normalize_observations else None
        self.reward_scaler = (
            _RewardScaler(cfg.gamma, cfg.simba_reward_scale_g_max) if cfg.simba_reward_scaling else None
        )
        self.pendulum_potential = (
            PendulumPotential(
                source=cfg.pendulum_potential_shaping_source,
                dp_grid_path=cfg.pendulum_potential_shaping_dp_grid_path,
                controller_grid_path=cfg.pendulum_potential_shaping_controller_grid_path,
                device=self.device,
            )
            if cfg.pendulum_potential_shaping_weight > 0.0
            else None
        )

        action_dim = int(np.prod(self.env_spec.single_action_space.shape))
        critic_cls: type[nn.Module]
        if cfg.simba_backbone:
            self.actor = SimbaNormalTanhActor(
                obs_dim,
                self.env_spec.single_action_space.low,
                self.env_spec.single_action_space.high,
                cfg,
            ).to(self.device)
            critic_cls = SimbaCategoricalQNetwork if cfg.simba_distributional_critic else SimbaScalarQNetwork
            self.q1 = critic_cls(obs_dim, action_dim, cfg).to(self.device)
            self.q2 = critic_cls(obs_dim, action_dim, cfg).to(self.device)
            self.q1_target = critic_cls(obs_dim, action_dim, cfg).to(self.device)
            self.q2_target = critic_cls(obs_dim, action_dim, cfg).to(self.device)
        else:
            actor_cls = _L2FeatureNormActor if cfg.l2_feature_norm else Actor
            critic_cls = _L2FeatureNormSoftQNetwork if cfg.l2_feature_norm else SoftQNetwork
            self.actor = actor_cls(self.env_spec).to(self.device)
            self.q1 = critic_cls(self.env_spec).to(self.device)
            self.q2 = critic_cls(self.env_spec).to(self.device)
            self.q1_target = critic_cls(self.env_spec).to(self.device)
            self.q2_target = critic_cls(self.env_spec).to(self.device)
        self.q_networks = nn.ModuleList([self.q1, self.q2])
        self.q_target_networks = nn.ModuleList([self.q1_target, self.q2_target])
        for _critic_index in range(2, int(cfg.redq_num_critics)):
            if cfg.simba_backbone:
                self.q_networks.append(critic_cls(obs_dim, action_dim, cfg).to(self.device))
                self.q_target_networks.append(critic_cls(obs_dim, action_dim, cfg).to(self.device))
            else:
                self.q_networks.append(critic_cls(self.env_spec).to(self.device))
                self.q_target_networks.append(critic_cls(self.env_spec).to(self.device))
        if cfg.simba_weight_projection:
            self._project_weights()
        for q_network, q_target_network in zip(self.q_networks, self.q_target_networks):
            q_target_network.load_state_dict(q_network.state_dict())

        q_params = [param for q_network in self.q_networks for param in q_network.parameters()]
        self.q_optimizer = optim.Adam(q_params, lr=cfg.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=cfg.policy_lr)
        self._q_lr_initial = float(cfg.q_lr)
        self._policy_lr_initial = float(cfg.policy_lr)
        self._alpha_lr_initial = float(cfg.q_lr if cfg.alpha_lr is None else cfg.alpha_lr)
        self._alpha_lr_final = cfg.q_lr_final if cfg.alpha_lr is None else cfg.alpha_lr_final
        self._current_alpha_lr = self._alpha_lr_initial

        action_dim_tensor = torch.prod(torch.Tensor(self.env_spec.single_action_space.shape).to(self.device)).item()
        self.target_entropy = float(cfg.target_entropy_scale) * float(action_dim_tensor)
        self.alpha_min_value = max(0.0, float(cfg.alpha_min_value))
        initial_alpha = max(float(cfg.alpha_initial_value), self.alpha_min_value)
        self._log_alpha_min = math.log(self.alpha_min_value) if self.alpha_min_value > 0.0 else None
        self.log_alpha = torch.full(
            (1,),
            math.log(initial_alpha),
            requires_grad=True,
            device=self.device,
        )
        self._alpha_tensor = self.log_alpha.exp().detach()
        self.alpha = float(self._alpha_tensor.cpu())
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self._alpha_lr_initial)
        self.last_actor_grad_norm = 0.0
        self.last_q_grad_norm = 0.0
        self.last_replay_priority_values: np.ndarray | None = None
        self.feature_norms: dict[str, float] = {}
        self._feature_hook_handles: list[Any] = []
        if cfg.update_diagnostics:
            self._register_feature_hooks()

    def observe(self, observation: np.ndarray) -> None:
        if self.obs_rms is not None and self.cfg.obs_rms_update_enabled:
            self.obs_rms.update(np.asarray(observation, dtype=np.float32))

    def observe_reward(self, reward: float, done: bool) -> None:
        if self.reward_scaler is not None:
            self.reward_scaler.update(reward, done)

    def _normalize_obs_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_rms is None:
            return obs
        mean = torch.as_tensor(self.obs_rms.mean, dtype=obs.dtype, device=obs.device)
        var = torch.as_tensor(self.obs_rms.var, dtype=obs.dtype, device=obs.device)
        return (obs - mean) / torch.sqrt(var + 1e-8)

    def _register_feature_hooks(self) -> None:
        def hook_for(metric_name: str):
            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                if isinstance(tensor, torch.Tensor):
                    self.feature_norms[metric_name] = float(torch.linalg.vector_norm(tensor.detach()).cpu())

            return hook

        modules: list[tuple[str, nn.Module]] = [("actor", self.actor)]
        modules.extend((f"q{idx + 1}", module) for idx, module in enumerate(self.q_networks))
        for prefix, module in modules:
            for name, child in module.named_modules():
                if name and not list(child.children()):
                    metric_name = f"feat_{prefix}_{name}"
                    self._feature_hook_handles.append(child.register_forward_hook(hook_for(metric_name)))

    @staticmethod
    def _record_parameter_norms(metrics: dict[str, float], prefix: str, module: nn.Module) -> None:
        for name, param in module.named_parameters():
            metrics[f"param_{prefix}_{name}"] = float(torch.linalg.vector_norm(param.detach()).cpu())

    @staticmethod
    def _record_gradient_norms(metrics: dict[str, float], prefix: str, module: nn.Module) -> None:
        for name, param in module.named_parameters():
            if param.grad is not None:
                metrics[f"grad_{prefix}_{name}"] = float(torch.linalg.vector_norm(param.grad.detach()).cpu())

    def _project_weights(self) -> None:
        for module in [self.actor, *self.q_networks]:
            project_simba_weights_to_unit_norm(module)

    def _maybe_redo_critics(self, obs: torch.Tensor, action: torch.Tensor, update_step: int) -> dict[str, float]:
        metrics = {
            "redo_event": 0.0,
            "redo_total_units": 0.0,
            "redo_q1_fc1_units": 0.0,
            "redo_q1_fc2_units": 0.0,
            "redo_q2_fc1_units": 0.0,
            "redo_q2_fc2_units": 0.0,
            "redo_q1_value_units": 0.0,
            "redo_q2_value_units": 0.0,
        }
        if self.cfg.redo_interval_updates <= 0 or update_step % self.cfg.redo_interval_updates != 0:
            return metrics
        if self.cfg.simba_backbone:
            q1_metrics = self._redo_simba_qnetwork(self.q1, self.q1_target, obs, action, prefix="q1")
            q2_metrics = self._redo_simba_qnetwork(self.q2, self.q2_target, obs, action, prefix="q2")
        else:
            q1_metrics = self._redo_softqnetwork(self.q1, self.q1_target, obs, action, prefix="q1")
            q2_metrics = self._redo_softqnetwork(self.q2, self.q2_target, obs, action, prefix="q2")
        metrics.update(q1_metrics)
        metrics.update(q2_metrics)
        metrics["redo_total_units"] = (
            metrics["redo_q1_fc1_units"]
            + metrics["redo_q1_fc2_units"]
            + metrics["redo_q2_fc1_units"]
            + metrics["redo_q2_fc2_units"]
            + metrics["redo_q1_value_units"]
            + metrics["redo_q2_value_units"]
        )
        metrics["redo_event"] = float(metrics["redo_total_units"] > 0.0)
        return metrics

    def _redo_softqnetwork(
        self,
        network: nn.Module,
        target_network: nn.Module,
        obs: torch.Tensor,
        action: torch.Tensor,
        prefix: str,
    ) -> dict[str, float]:
        if not all(hasattr(network, name) for name in ("fc1", "fc2", "fc3")):
            return {f"redo_{prefix}_fc1_units": 0.0, f"redo_{prefix}_fc2_units": 0.0}

        with torch.no_grad():
            x = torch.cat([obs, action], dim=1)
            first = F.relu(network.fc1(x))
            second = F.relu(network.fc2(first))
            fc1_mask = self._redo_dormant_mask(first)
            fc2_mask = self._redo_dormant_mask(second)
            self._reset_linear_rows(network.fc1, fc1_mask)
            self._reset_linear_rows(network.fc2, fc2_mask)
            network.fc2.weight[:, fc1_mask] = 0.0
            network.fc3.weight[:, fc2_mask] = 0.0
            self._copy_masked_linear(network.fc1, target_network.fc1, row_mask=fc1_mask)
            self._copy_masked_linear(network.fc2, target_network.fc2, row_mask=fc2_mask, col_mask=fc1_mask)
            self._copy_masked_linear(network.fc3, target_network.fc3, col_mask=fc2_mask)

        self._zero_optimizer_state(network.fc1.weight, row_mask=fc1_mask)
        self._zero_optimizer_state(network.fc1.bias, row_mask=fc1_mask)
        self._zero_optimizer_state(network.fc2.weight, row_mask=fc2_mask, col_mask=fc1_mask)
        self._zero_optimizer_state(network.fc2.bias, row_mask=fc2_mask)
        self._zero_optimizer_state(network.fc3.weight, col_mask=fc2_mask)
        return {
            f"redo_{prefix}_fc1_units": float(fc1_mask.sum().item()),
            f"redo_{prefix}_fc2_units": float(fc2_mask.sum().item()),
            f"redo_{prefix}_fc1_fraction": float(fc1_mask.float().mean().item()),
            f"redo_{prefix}_fc2_fraction": float(fc2_mask.float().mean().item()),
        }

    def _redo_simba_qnetwork(
        self,
        network: nn.Module,
        target_network: nn.Module,
        obs: torch.Tensor,
        action: torch.Tensor,
        prefix: str,
    ) -> dict[str, float]:
        required = ("backbone", "value_w1", "value_scaler", "value_w2")
        if not all(hasattr(network, name) and hasattr(target_network, name) for name in required):
            return {f"redo_{prefix}_value_units": 0.0}

        with torch.no_grad():
            x = torch.cat([obs, action], dim=1)
            hidden = network.backbone(x)
            value_hidden = network.value_scaler(network.value_w1(hidden))
            value_mask = self._redo_dormant_mask(value_hidden)
            self._reset_hyperdense_rows(network.value_w1, value_mask)
            network.value_w2.weight[:, value_mask] = 0.0
            network.value_scaler.scaler[value_mask] = 1.0
            self._copy_masked_hyperdense(network.value_w1, target_network.value_w1, row_mask=value_mask)
            self._copy_masked_hyperdense(network.value_w2, target_network.value_w2, col_mask=value_mask)
            target_network.value_scaler.scaler[value_mask] = network.value_scaler.scaler[value_mask]

        self._zero_optimizer_state(network.value_w1.weight, row_mask=value_mask)
        self._zero_optimizer_state(network.value_w2.weight, col_mask=value_mask)
        self._zero_optimizer_state(network.value_scaler.scaler, row_mask=value_mask)
        return {
            f"redo_{prefix}_value_units": float(value_mask.sum().item()),
            f"redo_{prefix}_value_fraction": float(value_mask.float().mean().item()),
        }

    def _redo_dormant_mask(self, activation: torch.Tensor) -> torch.Tensor:
        if activation.ndim > 2:
            activation = activation.flatten(start_dim=1)
        mean_abs_by_unit = activation.detach().abs().mean(dim=0)
        layer_mean = mean_abs_by_unit.mean()
        if float(layer_mean.cpu()) <= 0.0:
            return torch.ones_like(mean_abs_by_unit, dtype=torch.bool)
        score = mean_abs_by_unit / (layer_mean + 1e-12)
        return score <= float(self.cfg.redo_dormant_threshold)

    @staticmethod
    def _reset_linear_rows(layer: nn.Linear, row_mask: torch.Tensor) -> None:
        num_rows = int(row_mask.sum().item())
        if num_rows <= 0:
            return
        weight = torch.empty(
            (num_rows, layer.in_features),
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        layer.weight[row_mask] = weight
        if layer.bias is not None:
            bound = 1.0 / math.sqrt(layer.in_features)
            bias = torch.empty(num_rows, device=layer.bias.device, dtype=layer.bias.dtype).uniform_(-bound, bound)
            layer.bias[row_mask] = bias

    @staticmethod
    def _reset_hyperdense_rows(layer: SimbaHyperDense, row_mask: torch.Tensor) -> None:
        num_rows = int(row_mask.sum().item())
        if num_rows <= 0:
            return
        weight = torch.empty(
            (num_rows, layer.linear.in_features),
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        nn.init.orthogonal_(weight, gain=1.0)
        layer.weight[row_mask] = weight

    @staticmethod
    def _copy_masked_linear(
        source: nn.Linear,
        target: nn.Linear,
        row_mask: torch.Tensor | None = None,
        col_mask: torch.Tensor | None = None,
    ) -> None:
        if row_mask is not None and bool(row_mask.any().item()):
            target.weight[row_mask] = source.weight[row_mask]
            if source.bias is not None and target.bias is not None:
                target.bias[row_mask] = source.bias[row_mask]
        if col_mask is not None and bool(col_mask.any().item()):
            target.weight[:, col_mask] = source.weight[:, col_mask]

    @staticmethod
    def _copy_masked_hyperdense(
        source: SimbaHyperDense,
        target: SimbaHyperDense,
        row_mask: torch.Tensor | None = None,
        col_mask: torch.Tensor | None = None,
    ) -> None:
        if row_mask is not None and bool(row_mask.any().item()):
            target.weight[row_mask] = source.weight[row_mask]
        if col_mask is not None and bool(col_mask.any().item()):
            target.weight[:, col_mask] = source.weight[:, col_mask]

    def _zero_optimizer_state(
        self,
        param: torch.nn.Parameter | None,
        row_mask: torch.Tensor | None = None,
        col_mask: torch.Tensor | None = None,
    ) -> None:
        if param is None:
            return
        state = self.q_optimizer.state.get(param)
        if not state:
            return
        for value in state.values():
            if not isinstance(value, torch.Tensor) or value.shape != param.shape:
                continue
            if row_mask is not None and bool(row_mask.any().item()):
                if value.ndim == 1:
                    value[row_mask] = 0.0
                else:
                    value[row_mask, :] = 0.0
            if col_mask is not None and value.ndim == 2 and bool(col_mask.any().item()):
                value[:, col_mask] = 0.0

    def _scheduled_lr(self, initial: float, final: float | None, update_step: int) -> float:
        if final is None:
            return initial
        total_updates = max(1, int(self.cfg.total_steps) * int(self.cfg.updates_per_step))
        progress = min(max(update_step, 0) / total_updates, 1.0)
        return float(initial + progress * (final - initial))

    @staticmethod
    def _set_optimizer_lr(optimizer: optim.Optimizer, lr: float) -> None:
        for group in optimizer.param_groups:
            group["lr"] = lr

    def _apply_lr_schedule(self, update_step: int) -> tuple[float, float]:
        q_lr = self._scheduled_lr(self._q_lr_initial, self.cfg.q_lr_final, update_step)
        policy_lr = self._scheduled_lr(self._policy_lr_initial, self.cfg.policy_lr_final, update_step)
        alpha_lr = self._scheduled_lr(
            self._alpha_lr_initial,
            self._alpha_lr_final,
            update_step,
        )
        self._set_optimizer_lr(self.q_optimizer, q_lr)
        self._set_optimizer_lr(self.alpha_optimizer, alpha_lr)
        self._set_optimizer_lr(self.actor_optimizer, policy_lr)
        self._current_alpha_lr = alpha_lr
        return q_lr, policy_lr

    def _actor_q_aggregation_for_step(self, update_step: int) -> str:
        if (
            self.cfg.actor_q_aggregation_late is not None
            and self.cfg.actor_q_aggregation_switch_step > 0
            and update_step >= self.cfg.actor_q_aggregation_switch_step
        ):
            return self.cfg.actor_q_aggregation_late
        return self.cfg.actor_q_aggregation

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.act_batch(np.asarray(observation, dtype=np.float32)[None, :], deterministic=deterministic)[0]

    def act_batch(self, observations: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        obs = self._normalize_obs_tensor(obs)
        with torch.no_grad():
            action, _, mean = self.actor.get_action(obs)
        selected = mean if deterministic else action
        return selected.cpu().numpy()

    def act_with_log_prob(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        actions, log_probs = self.act_batch_with_log_prob(np.asarray(observation, dtype=np.float32)[None, :])
        return actions[0], float(log_probs[0])

    def act_batch_with_log_prob(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        obs = self._normalize_obs_tensor(obs)
        with torch.no_grad():
            action, log_prob, _mean = self.actor.get_action(obs)
        return action.cpu().numpy(), log_prob.view(-1).cpu().numpy()

    def act_batch_critic_search(
        self,
        observations: np.ndarray,
        num_actions: int,
        margin: float,
        batch_size: int = 512,
        filter_mode: str = "clipped_value",
        blend_fraction: float = 1.0,
    ) -> np.ndarray:
        if int(num_actions) <= 0:
            raise ValueError("num_actions must be positive")
        if float(margin) < 0.0:
            raise ValueError("margin must be nonnegative")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 < float(blend_fraction) <= 1.0:
            raise ValueError("blend_fraction must be in (0, 1]")
        if filter_mode not in {
            "always",
            "clipped_value",
            "unanimous_advantage",
            "online_target_unanimous_advantage",
            "online_target_joint_unanimous_advantage",
            "mean_proposal_unanimous_advantage",
            "mid0125_proposal_unanimous_advantage",
            "mid025_proposal_unanimous_advantage",
            "mid0375_proposal_unanimous_advantage",
            "unc025_increase_unanimous_advantage",
            "unc05_increase_unanimous_advantage",
            "unc1_increase_unanimous_advantage",
            "unc2_increase_unanimous_advantage",
            "target_unanimous_advantage",
            "target_proposal_online_unanimous_advantage",
            "target_proposal_online_target_unanimous_advantage",
            "lcb025_proposal_unanimous_advantage",
            "lcb05_proposal_unanimous_advantage",
            "lcb1_proposal_unanimous_advantage",
            "symmetric_actor_unanimous_advantage",
            "symmetric_critic_unanimous_advantage",
            "symmetric_actor_critic_unanimous_advantage",
        }:
            raise ValueError(
                "filter_mode must be one of: always, clipped_value, unanimous_advantage, "
                "online_target_unanimous_advantage, "
                "online_target_joint_unanimous_advantage, "
                "mean_proposal_unanimous_advantage, target_unanimous_advantage, "
                "mid0125_proposal_unanimous_advantage, "
                "mid025_proposal_unanimous_advantage, "
                "mid0375_proposal_unanimous_advantage, "
                "unc025_increase_unanimous_advantage, "
                "unc05_increase_unanimous_advantage, "
                "unc1_increase_unanimous_advantage, "
                "unc2_increase_unanimous_advantage, "
                "target_proposal_online_unanimous_advantage, "
                "target_proposal_online_target_unanimous_advantage, "
                "lcb025_proposal_unanimous_advantage, "
                "lcb05_proposal_unanimous_advantage, "
                "lcb1_proposal_unanimous_advantage, "
                "symmetric_actor_unanimous_advantage, "
                "symmetric_critic_unanimous_advantage, "
                "symmetric_actor_critic_unanimous_advantage"
            )
        raw_observations = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        normalized = self._normalize_obs_tensor(raw_observations)
        symmetric_modes = {
            "symmetric_actor_unanimous_advantage",
            "symmetric_critic_unanimous_advantage",
            "symmetric_actor_critic_unanimous_advantage",
        }
        mirrored_normalized = None
        if filter_mode in symmetric_modes:
            mirrored_normalized = self._normalize_obs_tensor(
                self._mirror_pendulum_observations(raw_observations)
            )
        selected_actions: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(normalized), int(batch_size)):
                obs_batch = normalized[start : start + int(batch_size)]
                _sampled, _log_prob, actor_actions = self.actor.get_action(obs_batch)
                mirrored_obs_batch = None
                if mirrored_normalized is not None:
                    mirrored_obs_batch = mirrored_normalized[start : start + int(batch_size)]
                if filter_mode in {
                    "symmetric_actor_unanimous_advantage",
                    "symmetric_actor_critic_unanimous_advantage",
                }:
                    assert mirrored_obs_batch is not None
                    _mirror_sampled, _mirror_log_prob, mirrored_actor_actions = (
                        self.actor.get_action(mirrored_obs_batch)
                    )
                    actor_actions = 0.5 * (actor_actions - mirrored_actor_actions)
                if filter_mode in {
                    "symmetric_critic_unanimous_advantage",
                    "symmetric_actor_critic_unanimous_advantage",
                }:
                    assert mirrored_obs_batch is not None
                    best_actions, advantage = self._critic_search_symmetric_best_actions(
                        observations=obs_batch,
                        mirrored_observations=mirrored_obs_batch,
                        actor_actions=actor_actions,
                        num_actions=int(num_actions),
                    )
                    use_search = advantage > float(margin)
                    blended_actions = actor_actions + float(blend_fraction) * (
                        best_actions - actor_actions
                    )
                    selected_actions.append(
                        torch.where(use_search, blended_actions, actor_actions)
                    )
                    continue
                search_critics = None
                if filter_mode == "online_target_joint_unanimous_advantage":
                    search_critics = list(self.q_networks) + list(self.q_target_networks)
                elif filter_mode in {
                    "target_unanimous_advantage",
                    "target_proposal_online_unanimous_advantage",
                    "target_proposal_online_target_unanimous_advantage",
                }:
                    search_critics = list(self.q_target_networks)
                candidate_aggregation = {
                    "mean_proposal_unanimous_advantage": "mean",
                    "mid0125_proposal_unanimous_advantage": "mid_0.125",
                    "mid025_proposal_unanimous_advantage": "mid_0.25",
                    "mid0375_proposal_unanimous_advantage": "mid_0.375",
                    "lcb025_proposal_unanimous_advantage": "lcb_0.25",
                    "lcb05_proposal_unanimous_advantage": "lcb_0.5",
                    "lcb1_proposal_unanimous_advantage": "lcb_1.0",
                }.get(filter_mode, "min")
                best_actions, best_q, actor_q = self._critic_search_best_actions(
                    observations=obs_batch,
                    actor_actions=actor_actions,
                    num_actions=int(num_actions),
                    critic_networks=search_critics,
                    candidate_aggregation=candidate_aggregation,
                )
                advantage = self._critic_search_filter_advantage(
                    observations=obs_batch,
                    actor_actions=actor_actions,
                    best_actions=best_actions,
                    best_q=best_q,
                    actor_q=actor_q,
                    filter_mode=filter_mode,
                )
                use_search = advantage > float(margin)
                blended_actions = actor_actions + float(blend_fraction) * (best_actions - actor_actions)
                selected_actions.append(torch.where(use_search, blended_actions, actor_actions))
        return torch.cat(selected_actions, dim=0).cpu().numpy()

    def _critic_search_filter_advantage(
        self,
        observations: torch.Tensor,
        actor_actions: torch.Tensor,
        best_actions: torch.Tensor,
        best_q: torch.Tensor,
        actor_q: torch.Tensor,
        filter_mode: str,
    ) -> torch.Tensor:
        if filter_mode == "always":
            return torch.full_like(best_q, float("inf"))
        if filter_mode == "clipped_value":
            return best_q - actor_q
        if filter_mode not in {
            "unanimous_advantage",
            "online_target_unanimous_advantage",
            "online_target_joint_unanimous_advantage",
            "mean_proposal_unanimous_advantage",
            "mid0125_proposal_unanimous_advantage",
            "mid025_proposal_unanimous_advantage",
            "mid0375_proposal_unanimous_advantage",
            "unc025_increase_unanimous_advantage",
            "unc05_increase_unanimous_advantage",
            "unc1_increase_unanimous_advantage",
            "unc2_increase_unanimous_advantage",
            "target_unanimous_advantage",
            "target_proposal_online_unanimous_advantage",
            "target_proposal_online_target_unanimous_advantage",
            "lcb025_proposal_unanimous_advantage",
            "lcb05_proposal_unanimous_advantage",
            "lcb1_proposal_unanimous_advantage",
            "symmetric_actor_unanimous_advantage",
            "symmetric_critic_unanimous_advantage",
            "symmetric_actor_critic_unanimous_advantage",
        }:
            raise ValueError(f"Unknown critic search filter mode: {filter_mode}")
        critic_networks = (
            list(self.q_target_networks)
            if filter_mode == "target_unanimous_advantage"
            else list(self.q_networks)
        )
        if filter_mode in {
            "online_target_unanimous_advantage",
            "online_target_joint_unanimous_advantage",
            "target_proposal_online_target_unanimous_advantage",
        }:
            critic_networks.extend(self.q_target_networks)
        with torch.no_grad():
            candidate_values = torch.stack(
                [
                    q_network(observations, best_actions).view(-1, 1)
                    for q_network in critic_networks
                ],
                dim=0,
            )
            actor_values = torch.stack(
                [
                    q_network(observations, actor_actions.detach()).view(-1, 1)
                    for q_network in critic_networks
                ],
                dim=0,
            )
            critic_advantages = candidate_values - actor_values
        robust_advantage = critic_advantages.min(dim=0).values
        uncertainty_coefficients = {
            "unc025_increase_unanimous_advantage": 0.25,
            "unc05_increase_unanimous_advantage": 0.5,
            "unc1_increase_unanimous_advantage": 1.0,
            "unc2_increase_unanimous_advantage": 2.0,
        }
        coefficient = uncertainty_coefficients.get(filter_mode)
        if coefficient is None:
            return robust_advantage
        candidate_disagreement = (
            candidate_values.max(dim=0).values - candidate_values.min(dim=0).values
        )
        actor_disagreement = actor_values.max(dim=0).values - actor_values.min(dim=0).values
        disagreement_increase = torch.clamp(
            candidate_disagreement - actor_disagreement, min=0.0
        )
        return robust_advantage - coefficient * disagreement_increase

    def _replay_importance_weights(
        self,
        data: Any,
        batch_size: int,
    ) -> torch.Tensor | None:
        if self.cfg.replay_priority_mode == "none":
            return None
        raw_weights = getattr(data, "importance_weights", None)
        if raw_weights is None:
            return torch.ones(batch_size, dtype=data.actions.dtype, device=self.device)
        weights = torch.as_tensor(
            raw_weights,
            dtype=data.actions.dtype,
            device=self.device,
        ).reshape(-1)
        if weights.shape[0] != batch_size:
            raise ValueError("Replay importance weights must align with the sampled batch.")
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0.0).any()):
            raise ValueError("Replay importance weights must be finite and positive.")
        return weights

    def _prepare_replay_priority_update(
        self,
        q_values: list[torch.Tensor],
        targets: torch.Tensor,
        replay_importance_weights: torch.Tensor | None,
    ) -> dict[str, float]:
        mode = self.cfg.replay_priority_mode
        if mode == "none":
            self.last_replay_priority_values = None
            return {"replay_priority_enabled": 0.0}

        with torch.no_grad():
            q_stack = torch.stack([value.reshape(-1) for value in q_values], dim=0)
            resolved_targets = targets.detach()
            if resolved_targets.ndim > 1:
                # A SAC-N sample is one replay transition with several valid
                # backup horizons. The longest available backup supplies one
                # unambiguous per-transition residual for replay prioritization.
                resolved_targets = resolved_targets.reshape(q_stack.shape[1], -1)[:, -1]
            else:
                resolved_targets = resolved_targets.reshape(-1)
            if resolved_targets.shape[0] != q_stack.shape[1]:
                raise ValueError("Replay priority targets must align with critic predictions.")
            bellman_residual = (q_stack - resolved_targets.unsqueeze(0)).abs().mean(dim=0)
            critic_disagreement = q_stack.max(dim=0).values - q_stack.min(dim=0).values
            if mode == "bellman_residual":
                raw_priority = bellman_residual
            elif mode == "critic_disagreement":
                raw_priority = critic_disagreement
            elif mode == "max":
                raw_priority = torch.maximum(bellman_residual, critic_disagreement)
            else:  # Configuration validation should make this unreachable.
                raise ValueError(f"Unknown replay_priority_mode: {mode}")
            clipped_priority = raw_priority.clamp(min=0.0, max=float(self.cfg.replay_priority_clip))
            self.last_replay_priority_values = clipped_priority.cpu().numpy().astype(np.float32, copy=False)

        metrics = {
            "replay_priority_enabled": 1.0,
            "replay_priority_mode_is_bellman_residual": float(mode == "bellman_residual"),
            "replay_priority_mode_is_critic_disagreement": float(mode == "critic_disagreement"),
            "replay_priority_mode_is_max": float(mode == "max"),
            "replay_priority_bellman_residual_mean": float(bellman_residual.mean().cpu()),
            "replay_priority_bellman_residual_max": float(bellman_residual.max().cpu()),
            "replay_priority_critic_disagreement_mean": float(critic_disagreement.mean().cpu()),
            "replay_priority_critic_disagreement_max": float(critic_disagreement.max().cpu()),
            "replay_priority_value_mean": float(clipped_priority.mean().cpu()),
            "replay_priority_value_max": float(clipped_priority.max().cpu()),
            "replay_priority_clip_fraction": float(
                (raw_priority > float(self.cfg.replay_priority_clip)).float().mean().cpu()
            ),
            "replay_priority_importance_correction_applied_to_critic": 1.0,
        }
        if replay_importance_weights is not None:
            metrics.update(
                {
                    "replay_priority_importance_weight_mean": float(
                        replay_importance_weights.detach().mean().cpu()
                    ),
                    "replay_priority_importance_weight_min": float(
                        replay_importance_weights.detach().min().cpu()
                    ),
                    "replay_priority_importance_weight_max": float(
                        replay_importance_weights.detach().max().cpu()
                    ),
                }
            )
        return metrics

    def update(
        self,
        data: Any,
        update_step: int,
        reference_actions: Any | None = None,
        reference_critic_actions: Any | None = None,
        reference_anchor_observations: Any | None = None,
        reference_anchor_actions: Any | None = None,
        collect_metrics: bool = True,
    ) -> dict[str, float]:
        if update_step <= 0:
            raise ValueError("update_step is one-indexed and must be positive.")
        self._collect_update_metrics = bool(collect_metrics)
        self.last_replay_priority_values = None
        if self.cfg.update_diagnostics:
            self.feature_norms.clear()
        q_lr, policy_lr = self._apply_lr_schedule(update_step)
        symmetry_metrics: dict[str, float] = {}
        if self.cfg.pendulum_symmetry_augmentation:
            data, reference_actions, reference_critic_actions, symmetry_metrics = (
                self._augment_pendulum_symmetry_batch(data, reference_actions, reference_critic_actions)
            )

        observations = self._normalize_obs_tensor(data.observations)
        replay_importance_weights = self._replay_importance_weights(data, observations.shape[0])
        mirrored_observations = None
        if (
            self.cfg.pendulum_actor_symmetry_weight > 0.0
            or self.cfg.pendulum_critic_symmetry_weight > 0.0
        ):
            # Mirror in physical observation coordinates, then apply the same
            # running-statistics transform as the original observation.
            mirrored_observations = self._normalize_obs_tensor(
                self._mirror_pendulum_observations(data.observations)
            )
        reference_anchor_observations_tensor = None
        reference_anchor_actions_tensor = None
        if reference_anchor_observations is not None or reference_anchor_actions is not None:
            if reference_anchor_observations is None or reference_anchor_actions is None:
                raise ValueError("reference anchor observations and actions must be provided together")
            raw_anchor_observations = torch.as_tensor(
                reference_anchor_observations,
                dtype=observations.dtype,
                device=self.device,
            ).reshape(-1, observations.shape[1])
            reference_anchor_observations_tensor = self._normalize_obs_tensor(
                raw_anchor_observations
            )
            reference_anchor_actions_tensor = torch.as_tensor(
                reference_anchor_actions,
                dtype=data.actions.dtype,
                device=self.device,
            ).reshape(-1, data.actions.shape[1])
            if reference_anchor_observations_tensor.shape[0] != reference_anchor_actions_tensor.shape[0]:
                raise ValueError("reference anchor observations/actions must have the same batch size")
        reference_actions_tensor = self._prepare_reference_actions(reference_actions, data.actions, observations)
        reference_critic_actions_tensor = self._prepare_reference_critic_actions(
            reference_critic_actions,
            data.actions,
            observations,
        )
        q_a_values = [q_network(observations, data.actions).view(-1) for q_network in self.q_networks]
        qf1_a_values = q_a_values[0]
        qf2_a_values = q_a_values[1]

        sacn_metrics: dict[str, float] = {}
        shaping_metrics: dict[str, float] = {}
        if self._uses_sacn_batch(data):
            if self.cfg.simba_distributional_critic:
                q_losses, next_q_value, sacn_metrics = self._distributional_sacn_critic_losses(
                    data=data,
                    observations=observations,
                    actions=data.actions,
                    update_step=update_step,
                    replay_importance_weights=replay_importance_weights,
                )
            else:
                q_losses, next_q_value, sacn_metrics = self._scalar_sacn_critic_losses(
                    data=data,
                    observations=observations,
                    actions=data.actions,
                    update_step=update_step,
                    replay_importance_weights=replay_importance_weights,
                )
        else:
            next_observations = self._normalize_obs_tensor(data.next_observations)
            rewards = data.rewards
            rewards, shaping_metrics = self._shape_pendulum_rewards(
                observations=data.observations,
                next_observations=data.next_observations,
                rewards=rewards,
                dones=data.dones,
                update_step=update_step,
            )
            if self.reward_scaler is not None:
                rewards = self.reward_scaler.scale_tensor(rewards)

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = self.actor.get_action(next_observations)
                qf_next_target = (
                    self._target_q_aggregate(next_observations, next_state_actions, update_step)
                    - self._alpha_tensor * next_state_log_pi
                )
                next_q_value = rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * self.cfg.gamma * qf_next_target.view(-1)

            if self.cfg.simba_distributional_critic:
                q_losses = self._distributional_critic_losses(
                    observations=observations,
                    actions=data.actions,
                    next_observations=next_observations,
                    rewards=rewards,
                    dones=data.dones,
                    next_actions=next_state_actions,
                    next_log_pi=next_state_log_pi,
                    update_step=update_step,
                    replay_importance_weights=replay_importance_weights,
                )
            else:
                if replay_importance_weights is None:
                    q_losses = [F.mse_loss(q_values, next_q_value) for q_values in q_a_values]
                else:
                    q_losses = [
                        ((q_values - next_q_value).pow(2) * replay_importance_weights).mean()
                        for q_values in q_a_values
                    ]

        priority_metrics = self._prepare_replay_priority_update(
            q_values=q_a_values,
            targets=next_q_value,
            replay_importance_weights=replay_importance_weights,
        )

        reference_critic_metrics: dict[str, float] = {}
        if reference_critic_actions_tensor is not None:
            ref_qf1_loss, ref_qf2_loss, reference_critic_metrics = self._reference_critic_margin_losses(
                observations=observations,
                reference_actions=reference_critic_actions_tensor,
            )
            q_losses[0] = q_losses[0] + float(self.cfg.reference_critic_weight) * ref_qf1_loss
            q_losses[1] = q_losses[1] + float(self.cfg.reference_critic_weight) * ref_qf2_loss
        critic_symmetry_metrics: dict[str, float] = {}
        if mirrored_observations is not None and self.cfg.pendulum_critic_symmetry_weight > 0.0:
            critic_symmetry_losses, critic_symmetry_metrics = (
                self._pendulum_critic_symmetry_losses(
                    observations=observations,
                    mirrored_observations=mirrored_observations,
                    actions=data.actions,
                    original_q_values=q_a_values,
                )
            )
            critic_symmetry_weight = float(self.cfg.pendulum_critic_symmetry_weight)
            q_losses = [
                q_loss + critic_symmetry_weight * symmetry_loss
                for q_loss, symmetry_loss in zip(q_losses, critic_symmetry_losses)
            ]
        cql_metrics: dict[str, float] = {}
        apply_cql = self.cfg.cql_alpha > 0.0 and update_step % int(self.cfg.cql_interval_updates) == 0
        if apply_cql:
            cql_qf1_loss, cql_qf2_loss, cql_metrics = self._conservative_critic_losses(
                observations=observations,
                data_actions=data.actions,
                qf1_data_values=qf1_a_values,
                qf2_data_values=qf2_a_values,
            )
            q_losses[0] = q_losses[0] + float(self.cfg.cql_alpha) * cql_qf1_loss
            q_losses[1] = q_losses[1] + float(self.cfg.cql_alpha) * cql_qf2_loss
        qf1_loss = q_losses[0]
        qf2_loss = q_losses[1]
        qf_loss = torch.stack(q_losses).sum()

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        q_params = [param for q_network in self.q_networks for param in q_network.parameters()]
        if self.cfg.update_diagnostics:
            q_params_before = snapshot_parameters(q_params)
            q_param_norm_before = parameter_list_norm(q_params)
            self.last_q_grad_norm = gradient_list_norm(q_params)
        else:
            q_params_before = []
            q_param_norm_before = 0.0
            self.last_q_grad_norm = 0.0
        self.q_optimizer.step()
        if self.cfg.simba_weight_projection:
            for q_network in self.q_networks:
                project_simba_weights_to_unit_norm(q_network)
        redo_metrics = self._maybe_redo_critics(observations, data.actions, update_step)
        q_update_norm = parameter_delta_norm(q_params_before, q_params) if self.cfg.update_diagnostics else 0.0

        metrics: dict[str, float] = {}
        if collect_metrics:
            metrics.update(
                {
                    "q1_loss": float(qf1_loss.detach().cpu()),
                    "q2_loss": float(qf2_loss.detach().cpu()),
                    "q_loss": float(qf_loss.detach().cpu()),
                    "q_loss_cleanrl_logged": float((qf_loss.detach() / float(len(q_losses))).cpu()),
                    "q1_mean": float(qf1_a_values.detach().mean().cpu()),
                    "q2_mean": float(qf2_a_values.detach().mean().cpu()),
                    "target_q_mean": float(next_q_value.detach().mean().cpu()),
                    "alpha": float(self._alpha_tensor.cpu()),
                    "q_lr": float(q_lr),
                    "policy_lr": float(policy_lr),
                    "alpha_lr": float(self._current_alpha_lr),
                    "redq_num_critics": float(len(self.q_networks)),
                    "redq_target_subset_size": float(self._redq_target_subset_size()),
                    "target_q_aggregation_is_min": 1.0 if self.cfg.target_q_aggregation == "min" else 0.0,
                    "target_q_aggregation_is_mean": 1.0 if self.cfg.target_q_aggregation == "mean" else 0.0,
                    "target_q_aggregation_is_max": 1.0 if self.cfg.target_q_aggregation == "max" else 0.0,
                    "cql_interval_updates": float(self.cfg.cql_interval_updates),
                    "cql_applied": float(apply_cql),
                }
            )
            metrics.update(shaping_metrics)
            metrics.update(sacn_metrics)
            metrics.update(symmetry_metrics)
            metrics.update(reference_critic_metrics)
            metrics.update(critic_symmetry_metrics)
            metrics.update(cql_metrics)
            metrics.update(redo_metrics)
            metrics.update(priority_metrics)
        if self.cfg.update_diagnostics:
            metrics["q_grad_norm"] = self.last_q_grad_norm
            self._record_gradient_norms(metrics, "q1", self.q1)
            self._record_gradient_norms(metrics, "q2", self.q2)

        actor_updates_enabled = update_step >= int(self.cfg.actor_update_start_step) and (
            int(self.cfg.actor_update_stop_step) <= 0 or update_step <= int(self.cfg.actor_update_stop_step)
        )
        actor_updates_per_trigger = int(self.cfg.actor_updates_per_trigger)
        if actor_updates_per_trigger <= 0:
            actor_updates_per_trigger = int(self.cfg.policy_frequency)
        if collect_metrics:
            metrics["actor_updates_enabled"] = float(actor_updates_enabled)
            metrics["actor_updates_per_trigger"] = float(actor_updates_per_trigger)
            metrics["actor_updates_executed"] = 0.0
        if actor_updates_enabled and update_step % self.cfg.policy_frequency == 0:
            for _ in range(actor_updates_per_trigger):
                pi, log_pi, policy_mean = self.actor.get_action(observations)
                deterministic_mean_objective = (
                    self.cfg.sac_actor_objective_mode == "deterministic_mean"
                )
                actor_objective_actions = policy_mean if deterministic_mean_objective else pi
                qf_pi_values = torch.stack(
                    [
                        q_network(observations, actor_objective_actions)
                        for q_network in self.q_networks
                    ],
                    dim=0,
                )
                actor_q_aggregation = self._actor_q_aggregation_for_step(update_step)
                if actor_q_aggregation == "min":
                    actor_qf_pi = qf_pi_values.min(dim=0).values
                elif actor_q_aggregation == "mean":
                    actor_qf_pi = qf_pi_values.mean(dim=0)
                elif actor_q_aggregation == "max":
                    actor_qf_pi = qf_pi_values.max(dim=0).values
                else:
                    raise ValueError(f"Unknown actor_q_aggregation: {actor_q_aggregation}")
                sac_actor_per_sample = (
                    -actor_qf_pi
                    if deterministic_mean_objective
                    else (self._alpha_tensor * log_pi) - actor_qf_pi
                )
                sac_actor_loss_unfiltered = sac_actor_per_sample.mean()
                (
                    sac_actor_loss,
                    sac_actor_filter_has_samples,
                    sac_actor_filter_metrics,
                ) = self._filter_sac_actor_loss(
                    observations=observations,
                    policy_mean=policy_mean,
                    reference_actions=reference_actions_tensor,
                    per_sample_loss=sac_actor_per_sample,
                )
                sac_actor_loss_active = (
                    update_step >= int(self.cfg.sac_actor_loss_start_step)
                    and float(self.cfg.sac_actor_loss_weight) > 0.0
                    and sac_actor_filter_has_samples
                )
                sac_actor_weight = (
                    float(self.cfg.sac_actor_loss_weight) if sac_actor_loss_active else 0.0
                )
                base_sac_actor_weight = sac_actor_weight
                actor_loss = sac_actor_weight * sac_actor_loss
                actor_mean_logit_metrics: dict[str, float] = {}
                actor_mean_logit_l2_penalty: torch.Tensor | None = None
                actor_mean_logit_l2_weight = float(
                    self.cfg.actor_mean_logit_l2_weight
                )
                if actor_mean_logit_l2_weight > 0.0 or collect_metrics:
                    if isinstance(self.actor, SimbaNormalTanhActor):
                        (
                            actor_mean_logits,
                            actor_log_std,
                            actor_unclamped_log_std,
                        ) = self.actor.forward_with_unclamped_log_std(observations)
                    else:
                        actor_mean_logits, actor_log_std = self.actor(observations)
                        actor_unclamped_log_std = actor_log_std
                    (
                        actor_mean_logit_l2_penalty,
                        actor_mean_logit_metrics,
                    ) = self._actor_mean_logit_l2_penalty(
                        actor_mean_logits,
                        excess_threshold=float(
                            self.cfg.actor_mean_logit_excess_threshold
                        ),
                        collect_metrics=collect_metrics,
                    )
                    if actor_mean_logit_l2_weight > 0.0:
                        actor_loss = (
                            actor_loss
                            + actor_mean_logit_l2_weight
                            * actor_mean_logit_l2_penalty
                        )
                    if collect_metrics:
                        detached_log_std = actor_log_std.detach()
                        detached_unclamped_log_std = actor_unclamped_log_std.detach()
                        configured_floor = getattr(
                            self.actor, "log_std_floor", None
                        )
                        actor_mean_logit_metrics.update(
                            {
                                "actor_log_std_mean": float(
                                    detached_log_std.mean().cpu()
                                ),
                                "actor_log_std_min": float(
                                    detached_log_std.min().cpu()
                                ),
                                "actor_log_std_below_minus_1_fraction": float(
                                    (detached_log_std < -1.0).float().mean().cpu()
                                ),
                                "actor_log_std_below_minus_1p5_fraction": float(
                                    (detached_log_std < -1.5).float().mean().cpu()
                                ),
                                "actor_log_std_below_minus_2_fraction": float(
                                    (detached_log_std < -2.0).float().mean().cpu()
                                ),
                                "actor_log_std_below_minus_3_fraction": float(
                                    (detached_log_std < -3.0).float().mean().cpu()
                                ),
                                "actor_unclamped_log_std_mean": float(
                                    detached_unclamped_log_std.mean().cpu()
                                ),
                                "actor_unclamped_log_std_min": float(
                                    detached_unclamped_log_std.min().cpu()
                                ),
                                "actor_unclamped_log_std_below_minus_1_fraction": float(
                                    (detached_unclamped_log_std < -1.0)
                                    .float()
                                    .mean()
                                    .cpu()
                                ),
                                "actor_unclamped_log_std_below_minus_1p5_fraction": float(
                                    (detached_unclamped_log_std < -1.5)
                                    .float()
                                    .mean()
                                    .cpu()
                                ),
                                "actor_unclamped_log_std_below_minus_2_fraction": float(
                                    (detached_unclamped_log_std < -2.0)
                                    .float()
                                    .mean()
                                    .cpu()
                                ),
                                "actor_unclamped_log_std_below_minus_3_fraction": float(
                                    (detached_unclamped_log_std < -3.0)
                                    .float()
                                    .mean()
                                    .cpu()
                                ),
                                "actor_log_std_effective_floor": float(
                                    configured_floor
                                    if configured_floor is not None
                                    else (-10.0 if isinstance(self.actor, SimbaNormalTanhActor) else -5.0)
                                ),
                            }
                        )
                reference_actor_metrics: dict[str, float] = {}
                critic_search_metrics: dict[str, float] = {}
                self_imitation_metrics: dict[str, float] = {}
                actor_symmetry_metrics: dict[str, float] = {}
                actor_gradient_alignment_metrics: dict[str, float] = {}
                actor_gradient_balance_metrics: dict[str, float] = {
                    "actor_sac_gradient_balance_enabled": float(
                        self.cfg.sac_actor_gradient_balance_mode != "none"
                    ),
                    "actor_sac_gradient_balance_active": 0.0,
                    "actor_sac_gradient_balance_raw_ratio": 0.0,
                    "actor_sac_gradient_balance_multiplier": 1.0,
                }
                actor_gradient_projection_metrics: dict[str, float] = {
                    "actor_sac_bc_projection_enabled": float(
                        self.cfg.sac_actor_gradient_conflict_mode != "none"
                    ),
                    "actor_sac_bc_projection_joint_active": 0.0,
                    "actor_sac_bc_projection_applied": 0.0,
                }
                reference_actor_loss: torch.Tensor | None = None
                reference_actor_weight = self._reference_auxiliary_weight(update_step)
                if (
                    self.cfg.critic_search_actor_weight > 0.0
                    and update_step >= int(self.cfg.critic_search_start_update)
                ):
                    critic_search_loss, critic_search_metrics = self._critic_search_actor_loss(
                        observations=observations,
                        actor_actions=policy_mean,
                    )
                    actor_loss = actor_loss + float(self.cfg.critic_search_actor_weight) * critic_search_loss
                if (
                    self.cfg.self_imitation_weight > 0.0
                    and update_step >= int(self.cfg.self_imitation_start_step)
                ):
                    self_imitation_loss, self_imitation_metrics = self._self_imitation_actor_loss(
                        observations=observations,
                        replay_actions=data.actions,
                        actor_actions=policy_mean,
                    )
                    actor_loss = actor_loss + float(self.cfg.self_imitation_weight) * self_imitation_loss
                if mirrored_observations is not None and self.cfg.pendulum_actor_symmetry_weight > 0.0:
                    actor_symmetry_loss, actor_symmetry_metrics = (
                        self._pendulum_actor_symmetry_loss(
                            policy_mean=policy_mean,
                            mirrored_observations=mirrored_observations,
                        )
                    )
                    actor_loss = (
                        actor_loss
                        + float(self.cfg.pendulum_actor_symmetry_weight) * actor_symmetry_loss
                    )
                if (
                    reference_actions_tensor is not None
                    and self.cfg.reference_auxiliary_mode != "none"
                    and reference_actor_weight > 0.0
                ):
                    reference_replay_sample_count = int(observations.shape[0])
                    reference_loss_observations = observations
                    reference_loss_actor_actions = policy_mean
                    reference_loss_actions = reference_actions_tensor
                    if (
                        reference_anchor_observations_tensor is not None
                        and reference_anchor_actions_tensor is not None
                    ):
                        _, _, anchor_policy_mean = self.actor.get_action(
                            reference_anchor_observations_tensor
                        )
                        reference_loss_observations = torch.cat(
                            [observations, reference_anchor_observations_tensor], dim=0
                        )
                        reference_loss_actor_actions = torch.cat(
                            [policy_mean, anchor_policy_mean], dim=0
                        )
                        reference_loss_actions = torch.cat(
                            [reference_actions_tensor, reference_anchor_actions_tensor], dim=0
                        )
                    reference_actor_loss, reference_actor_metrics = self._reference_auxiliary_actor_loss(
                        observations=reference_loss_observations,
                        actor_actions=reference_loss_actor_actions,
                        reference_actions=reference_loss_actions,
                        replay_sample_count=reference_replay_sample_count,
                        q_filter_active=(
                            update_step
                            >= int(self.cfg.reference_auxiliary_filter_start_update)
                        ),
                    )
                    reference_actor_metrics["reference_actor_anchor_fraction"] = float(
                        0.0
                        if reference_anchor_actions_tensor is None
                        else reference_anchor_actions_tensor.shape[0]
                        / float(reference_loss_actions.shape[0])
                    )
                    actor_loss = actor_loss + reference_actor_weight * reference_actor_loss

                gradient_balance_enabled = (
                    self.cfg.sac_actor_gradient_balance_mode == "match_reference"
                )
                if (
                    (collect_metrics or gradient_balance_enabled)
                    and reference_actor_loss is not None
                    and base_sac_actor_weight > 0.0
                    and reference_actor_weight > 0.0
                ):
                    actor_gradient_alignment_metrics = self._actor_loss_gradient_alignment(
                        sac_actor_loss=sac_actor_loss,
                        reference_actor_loss=reference_actor_loss,
                        sac_weight=base_sac_actor_weight,
                        reference_weight=reference_actor_weight,
                    )
                    if gradient_balance_enabled:
                        raw_ratio = self._safe_actor_gradient_balance_ratio(
                            actor_gradient_alignment_metrics
                        )
                        if raw_ratio is not None:
                            gradient_balance_multiplier = min(
                                max(
                                    raw_ratio,
                                    float(self.cfg.sac_actor_gradient_balance_min_multiplier),
                                ),
                                float(self.cfg.sac_actor_gradient_balance_max_multiplier),
                            )
                            sac_actor_weight = base_sac_actor_weight * gradient_balance_multiplier
                            actor_loss = actor_loss + (
                                sac_actor_weight - base_sac_actor_weight
                            ) * sac_actor_loss
                            actor_gradient_balance_metrics.update(
                                {
                                    "actor_sac_gradient_balance_active": 1.0,
                                    "actor_sac_gradient_balance_raw_ratio": raw_ratio,
                                    "actor_sac_gradient_balance_multiplier": gradient_balance_multiplier,
                                }
                            )
                            actor_gradient_alignment_metrics[
                                "actor_weighted_sac_gradient_norm"
                            ] = (
                                actor_gradient_alignment_metrics["actor_sac_gradient_norm"]
                                * sac_actor_weight
                            )

                actor_params = list(self.actor.parameters())
                self.actor_optimizer.zero_grad()
                project_sac_gradient = (
                    self.cfg.sac_actor_gradient_conflict_mode == "project_sac"
                    and reference_actor_loss is not None
                    and sac_actor_weight > 0.0
                    and reference_actor_weight > 0.0
                )
                if project_sac_gradient:
                    weighted_sac_gradients = torch.autograd.grad(
                        sac_actor_weight * sac_actor_loss,
                        actor_params,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    weighted_bc_gradients = torch.autograd.grad(
                        reference_actor_weight * reference_actor_loss,
                        actor_params,
                        allow_unused=True,
                    )
                    projected_sac_gradients, projection_metrics = (
                        self._project_sac_gradient_off_bc(
                            weighted_sac_gradients,
                            weighted_bc_gradients,
                        )
                    )
                    for parameter, sac_gradient, bc_gradient in zip(
                        actor_params,
                        projected_sac_gradients,
                        weighted_bc_gradients,
                    ):
                        if sac_gradient is None and bc_gradient is None:
                            parameter.grad = None
                            continue
                        if sac_gradient is None:
                            combined_gradient = bc_gradient
                        elif bc_gradient is None:
                            combined_gradient = sac_gradient
                        else:
                            combined_gradient = sac_gradient + bc_gradient
                        assert combined_gradient is not None
                        parameter.grad = combined_gradient.detach()
                    actor_gradient_projection_metrics.update(projection_metrics)
                    actor_gradient_projection_metrics[
                        "actor_sac_bc_projection_joint_active"
                    ] = 1.0
                else:
                    actor_loss.backward()
                if self.cfg.update_diagnostics:
                    actor_params_before = snapshot_parameters(actor_params)
                    actor_param_norm_before = parameter_list_norm(actor_params)
                    self.last_actor_grad_norm = gradient_norm(self.actor)
                    self._record_gradient_norms(metrics, "actor", self.actor)
                else:
                    actor_params_before = []
                    actor_param_norm_before = 0.0
                    self.last_actor_grad_norm = 0.0
                self.actor_optimizer.step()
                if self.cfg.simba_weight_projection:
                    project_simba_weights_to_unit_norm(self.actor)
                actor_update_norm = (
                    parameter_delta_norm(actor_params_before, actor_params) if self.cfg.update_diagnostics else 0.0
                )

                with torch.no_grad():
                    _, log_pi, _ = self.actor.get_action(observations)
                alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                if sac_actor_loss_active and not deterministic_mean_objective:
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    if self._log_alpha_min is not None:
                        with torch.no_grad():
                            self.log_alpha.clamp_(min=self._log_alpha_min)
                self._alpha_tensor = self.log_alpha.exp().detach()
                if collect_metrics:
                    self.alpha = float(self._alpha_tensor.cpu())

            if collect_metrics:
                metrics.update(
                    {
                    "actor_updates_executed": float(actor_updates_per_trigger),
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "sac_actor_loss": float(sac_actor_loss.detach().cpu()),
                    "sac_actor_loss_unfiltered": float(sac_actor_loss_unfiltered.detach().cpu()),
                    "sac_actor_loss_weight": float(sac_actor_weight),
                    "sac_actor_loss_base_weight": float(base_sac_actor_weight),
                    "sac_actor_loss_active": float(sac_actor_loss_active),
                    "sac_actor_objective_is_deterministic_mean": float(
                        deterministic_mean_objective
                    ),
                    "reference_actor_loss_weight": float(reference_actor_weight),
                    "alpha_loss": float(alpha_loss.detach().cpu()),
                    "policy_log_prob_mean": float(log_pi.detach().mean().cpu()),
                    "policy_entropy_estimate": float((-log_pi.detach()).mean().cpu()),
                    "actor_q_mean": float(actor_qf_pi.detach().mean().cpu()),
                    "actor_q_min_mean": float(qf_pi_values.detach().min(dim=0).values.mean().cpu()),
                    "actor_q_max_mean": float(qf_pi_values.detach().max(dim=0).values.mean().cpu()),
                    "actor_q_aggregation_is_min": 1.0 if actor_q_aggregation == "min" else 0.0,
                    "actor_q_aggregation_is_mean": 1.0 if actor_q_aggregation == "mean" else 0.0,
                    "actor_q_aggregation_is_max": 1.0 if actor_q_aggregation == "max" else 0.0,
                    "actor_q_aggregation_switch_active": float(
                        self.cfg.actor_q_aggregation_late is not None
                        and self.cfg.actor_q_aggregation_switch_step > 0
                        and update_step >= self.cfg.actor_q_aggregation_switch_step
                    ),
                    "alpha": float(self.alpha),
                    "alpha_min_value": float(self.alpha_min_value),
                    "alpha_floor_active": float(
                        self.alpha_min_value > 0.0 and self.alpha <= self.alpha_min_value * (1.0 + 1e-6)
                    ),
                    }
                )
                metrics.update(critic_search_metrics)
                metrics.update(self_imitation_metrics)
                metrics.update(actor_symmetry_metrics)
                metrics.update(reference_actor_metrics)
                metrics.update(actor_gradient_alignment_metrics)
                metrics.update(actor_gradient_balance_metrics)
                metrics.update(actor_gradient_projection_metrics)
                metrics.update(sac_actor_filter_metrics)
                metrics.update(actor_mean_logit_metrics)
                metrics["actor_mean_logit_l2_weight"] = actor_mean_logit_l2_weight
                metrics["actor_mean_logit_excess_threshold"] = float(
                    self.cfg.actor_mean_logit_excess_threshold
                )
            if self.cfg.update_diagnostics:
                metrics.update(
                    {
                        "actor_grad_norm": self.last_actor_grad_norm,
                        "actor_update_norm": actor_update_norm,
                        "actor_update_norm_ratio": actor_update_norm / max(actor_param_norm_before, 1e-12),
                    }
                )

        if update_step % self.cfg.target_network_frequency == 0:
            for q_network, q_target_network in zip(self.q_networks, self.q_target_networks):
                for param, target_param in zip(q_network.parameters(), q_target_network.parameters()):
                    target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)

        if self.cfg.update_diagnostics:
            actor_norm = parameter_norm(self.actor)
            q_norm = sum(parameter_norm(q_network) for q_network in self.q_networks)
            self._record_parameter_norms(metrics, "actor", self.actor)
            self._record_parameter_norms(metrics, "q1", self.q1)
            self._record_parameter_norms(metrics, "q2", self.q2)
            metrics.update(self.feature_norms)
            metrics.update(
                {
                    "actor_param_norm": actor_norm,
                    "q_param_norm": q_norm,
                    "q_update_norm": q_update_norm,
                    "q_update_norm_ratio": q_update_norm / max(q_param_norm_before, 1e-12),
                }
            )
        return metrics

    def _uses_sacn_batch(self, data: Any) -> bool:
        return int(self.cfg.sacn_n_step) > 1 and hasattr(data, "trajectory_rewards")

    def _augment_pendulum_symmetry_batch(
        self,
        data: Any,
        reference_actions: Any | None,
        reference_critic_actions: Any | None,
    ) -> tuple[Any, Any | None, Any | None, dict[str, float]]:
        original_size = int(data.observations.shape[0])
        if self._uses_sacn_batch(data):
            augmented = SACNReplaySamples(
                observations=torch.cat(
                    [data.observations, self._mirror_pendulum_observations(data.observations)], dim=0
                ),
                actions=torch.cat([data.actions, self._mirror_pendulum_actions(data.actions)], dim=0),
                trajectory_observations=torch.cat(
                    [
                        data.trajectory_observations,
                        self._mirror_pendulum_observations(data.trajectory_observations),
                    ],
                    dim=0,
                ),
                trajectory_actions=torch.cat(
                    [data.trajectory_actions, self._mirror_pendulum_actions(data.trajectory_actions)], dim=0
                ),
                trajectory_next_observations=torch.cat(
                    [
                        data.trajectory_next_observations,
                        self._mirror_pendulum_observations(data.trajectory_next_observations),
                    ],
                    dim=0,
                ),
                trajectory_rewards=torch.cat([data.trajectory_rewards, data.trajectory_rewards.clone()], dim=0),
                trajectory_dones=torch.cat([data.trajectory_dones, data.trajectory_dones.clone()], dim=0),
                trajectory_action_log_probs=torch.cat(
                    [data.trajectory_action_log_probs, data.trajectory_action_log_probs.clone()], dim=0
                ),
            )
        elif isinstance(data, ReplayBufferSamples):
            augmented = ReplayBufferSamples(
                observations=torch.cat(
                    [data.observations, self._mirror_pendulum_observations(data.observations)], dim=0
                ),
                actions=torch.cat([data.actions, self._mirror_pendulum_actions(data.actions)], dim=0),
                next_observations=torch.cat(
                    [data.next_observations, self._mirror_pendulum_observations(data.next_observations)], dim=0
                ),
                dones=torch.cat([data.dones, data.dones.clone()], dim=0),
                rewards=torch.cat([data.rewards, data.rewards.clone()], dim=0),
            )
        elif hasattr(data, "_replace"):
            augmented = data._replace(
                observations=torch.cat(
                    [data.observations, self._mirror_pendulum_observations(data.observations)], dim=0
                ),
                actions=torch.cat([data.actions, self._mirror_pendulum_actions(data.actions)], dim=0),
                next_observations=torch.cat(
                    [data.next_observations, self._mirror_pendulum_observations(data.next_observations)], dim=0
                ),
                dones=torch.cat([data.dones, data.dones.clone()], dim=0),
                rewards=torch.cat([data.rewards, data.rewards.clone()], dim=0),
            )
        else:
            raise TypeError(f"Unsupported replay batch type for Pendulum symmetry augmentation: {type(data)!r}")

        augmented_size = int(augmented.observations.shape[0])
        return (
            augmented,
            self._augment_optional_pendulum_actions(reference_actions),
            self._augment_optional_pendulum_actions(reference_critic_actions),
            {
                "pendulum_symmetry_augmentation": 1.0,
                "pendulum_symmetry_original_batch_size": float(original_size),
                "pendulum_symmetry_augmented_batch_size": float(augmented_size),
                "pendulum_symmetry_batch_multiplier": float(augmented_size) / float(max(original_size, 1)),
            },
        )

    @staticmethod
    def _mirror_pendulum_observations(observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] < 3:
            raise ValueError("Pendulum symmetry requires observations shaped as [cos, sin, theta_dot].")
        mirrored = observations.clone()
        mirrored[..., 1] = -mirrored[..., 1]
        mirrored[..., 2] = -mirrored[..., 2]
        return mirrored

    @staticmethod
    def _mirror_pendulum_actions(actions: torch.Tensor) -> torch.Tensor:
        return -actions.clone()

    def _pendulum_actor_symmetry_loss(
        self,
        policy_mean: torch.Tensor,
        mirrored_observations: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mirrored_latent_mean, _ = self.actor(mirrored_observations)
        mirrored_policy_mean = (
            torch.tanh(mirrored_latent_mean) * self.actor.action_scale
            + self.actor.action_bias
        )
        equivariance_error = mirrored_policy_mean + policy_mean
        loss = equivariance_error.pow(2).mean()
        metrics = {
            "pendulum_actor_symmetry_loss": float(loss.detach().cpu()),
            "pendulum_actor_symmetry_contribution": float(
                (float(self.cfg.pendulum_actor_symmetry_weight) * loss.detach()).cpu()
            ),
            "pendulum_actor_symmetry_abs_error_mean": float(
                equivariance_error.detach().abs().mean().cpu()
            ),
            "pendulum_actor_symmetry_weight": float(
                self.cfg.pendulum_actor_symmetry_weight
            ),
        }
        return loss, metrics

    def _pendulum_critic_symmetry_losses(
        self,
        observations: torch.Tensor,
        mirrored_observations: torch.Tensor,
        actions: torch.Tensor,
        original_q_values: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], dict[str, float]]:
        if len(original_q_values) != len(self.q_networks):
            raise ValueError("original_q_values must contain one tensor per online critic")
        if observations.shape != mirrored_observations.shape:
            raise ValueError("original and mirrored observations must have identical shapes")
        mirrored_actions = self._mirror_pendulum_actions(actions)
        losses = [
            F.mse_loss(
                original_q_value.view(-1),
                q_network(mirrored_observations, mirrored_actions).view(-1),
            )
            for q_network, original_q_value in zip(self.q_networks, original_q_values)
        ]
        detached_losses = torch.stack([loss.detach() for loss in losses])
        metrics = {
            "pendulum_critic_symmetry_loss": float(detached_losses.mean().cpu()),
            "pendulum_critic_symmetry_loss_sum": float(detached_losses.sum().cpu()),
            "pendulum_critic_symmetry_contribution": float(
                (
                    float(self.cfg.pendulum_critic_symmetry_weight)
                    * detached_losses.sum()
                ).cpu()
            ),
            "pendulum_critic_symmetry_weight": float(
                self.cfg.pendulum_critic_symmetry_weight
            ),
        }
        for index, loss in enumerate(detached_losses, start=1):
            metrics[f"pendulum_critic_{index}_symmetry_loss"] = float(loss.cpu())
        return losses, metrics

    @classmethod
    def _augment_optional_pendulum_actions(cls, actions: Any | None) -> Any | None:
        if actions is None:
            return None
        if isinstance(actions, torch.Tensor):
            return torch.cat([actions, cls._mirror_pendulum_actions(actions)], dim=0)
        array = np.asarray(actions)
        return np.concatenate([array, -array], axis=0)

    def _redq_target_subset_size(self) -> int:
        return min(int(self.cfg.redq_target_subset_size), len(self.q_target_networks))

    def _redq_target_indices(self, update_step: int) -> list[int]:
        subset_size = self._redq_target_subset_size()
        if subset_size >= len(self.q_target_networks):
            return list(range(len(self.q_target_networks)))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(update_step) * 1_000_003 + len(self.q_target_networks) * 97 + subset_size)
        indices = torch.randperm(len(self.q_target_networks), generator=generator)[:subset_size]
        return [int(index.item()) for index in indices]

    def _target_q_aggregate(self, observations: torch.Tensor, actions: torch.Tensor, update_step: int) -> torch.Tensor:
        target_values = [
            self.q_target_networks[index](observations, actions)
            for index in self._redq_target_indices(update_step)
        ]
        target_stack = torch.stack(target_values, dim=0)
        if self.cfg.target_q_aggregation == "min":
            return target_stack.min(dim=0).values
        if self.cfg.target_q_aggregation == "mean":
            return target_stack.mean(dim=0)
        if self.cfg.target_q_aggregation == "max":
            return target_stack.max(dim=0).values
        raise ValueError(f"Unknown target_q_aggregation: {self.cfg.target_q_aggregation}")

    def _scalar_sacn_critic_losses(
        self,
        data: Any,
        observations: torch.Tensor,
        actions: torch.Tensor,
        update_step: int,
        replay_importance_weights: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, float]]:
        with torch.no_grad():
            common = self._sacn_common_target_inputs(data, update_step=update_step)
            bootstrap_values = self._target_q_aggregate(
                common["successor_observations"],
                common["successor_actions"],
                update_step=update_step,
            ).view(common["batch_size"], common["n_step"])
            targets = common["offsets"] + common["discounts"] * bootstrap_values

        weights = common["weights"]
        losses = []
        for q_network in self.q_networks:
            q_pred = q_network(observations, actions).view(-1, 1)
            losses.append(
                self._sacn_weighted_horizon_loss(
                    (q_pred - targets).pow(2),
                    weights,
                    common["horizon_loss_weights"],
                    replay_importance_weights,
                )
            )
        metrics = self._sacn_metrics(
            targets=targets,
            weights=weights,
            log_omega=common["log_omega"],
            log_importance_clip=common["log_importance_clip"],
            entropy_sample_counts=common["entropy_sample_counts"],
            horizon_ess_fraction=common["horizon_ess_fraction"],
            horizon_mask=common["horizon_mask"],
            horizon_loss_weights=common["horizon_loss_weights"],
        )
        metrics.update(common["shaping_metrics"])
        return losses, targets, metrics

    def _distributional_sacn_critic_losses(
        self,
        data: Any,
        observations: torch.Tensor,
        actions: torch.Tensor,
        update_step: int,
        replay_importance_weights: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, float]]:
        if not all(isinstance(q_network, SimbaCategoricalQNetwork) for q_network in self.q_networks):
            raise TypeError("Distributional SACn critic loss requires SimbaCategoricalQNetwork critics.")
        if not all(isinstance(q_network, SimbaCategoricalQNetwork) for q_network in self.q_target_networks):
            raise TypeError("Distributional SACn critic loss requires categorical target critics.")

        with torch.no_grad():
            common = self._sacn_common_target_inputs(data, update_step=update_step)
            target_qs = []
            target_log_probs = []
            for index in self._redq_target_indices(update_step):
                q_network = self.q_target_networks[index]
                next_q, next_log_prob = q_network.distribution(
                    common["successor_observations"], common["successor_actions"]
                )
                target_qs.append(next_q.view(-1))
                target_log_probs.append(next_log_prob)
            next_q_stack = torch.stack(target_qs, dim=0)
            next_log_probs_stack = torch.stack(target_log_probs, dim=0)
            flat_indices = torch.arange(next_q_stack.shape[1], device=next_q_stack.device)
            if self.cfg.target_q_aggregation == "min":
                selected_indices = next_q_stack.argmin(dim=0)
                next_log_probs = next_log_probs_stack[selected_indices, flat_indices]
                bootstrap_values = next_q_stack.min(dim=0).values
            elif self.cfg.target_q_aggregation == "mean":
                next_probs = next_log_probs_stack.exp().mean(dim=0).clamp_min(1e-12)
                next_log_probs = next_probs.log()
                bootstrap_values = next_q_stack.mean(dim=0)
            elif self.cfg.target_q_aggregation == "max":
                selected_indices = next_q_stack.argmax(dim=0)
                next_log_probs = next_log_probs_stack[selected_indices, flat_indices]
                bootstrap_values = next_q_stack.max(dim=0).values
            else:
                raise ValueError(f"Unknown target_q_aggregation: {self.cfg.target_q_aggregation}")
            next_log_probs = next_log_probs.view(
                common["batch_size"],
                common["n_step"],
                -1,
            )
            bootstrap_values = bootstrap_values.view(common["batch_size"], common["n_step"])
            target_values = common["offsets"] + common["discounts"] * bootstrap_values

        target_probs = self._categorical_project_distribution(
            target_log_probs=next_log_probs.reshape(common["batch_size"] * common["n_step"], -1),
            offsets=common["offsets"].reshape(-1),
            discounts=common["discounts"].reshape(-1),
            support=self.q1.bin_values,
        ).reshape(common["batch_size"], common["n_step"], -1)
        losses = []
        for q_network in self.q_networks:
            _q, pred_log_probs = q_network.distribution(observations, actions)
            ce = -(target_probs * pred_log_probs[:, None, :]).sum(dim=2)
            losses.append(
                self._sacn_weighted_horizon_loss(
                    ce,
                    common["weights"],
                    common["horizon_loss_weights"],
                    replay_importance_weights,
                )
            )

        metrics = self._sacn_metrics(
            targets=target_values,
            weights=common["weights"],
            log_omega=common["log_omega"],
            log_importance_clip=common["log_importance_clip"],
            entropy_sample_counts=common["entropy_sample_counts"],
            horizon_ess_fraction=common["horizon_ess_fraction"],
            horizon_mask=common["horizon_mask"],
            horizon_loss_weights=common["horizon_loss_weights"],
        )
        metrics.update(common["shaping_metrics"])
        return losses, target_values, metrics

    def _shape_pendulum_rewards(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        update_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.pendulum_potential is None:
            return rewards, {}
        start_update = int(self.cfg.pendulum_potential_shaping_start_update)
        if start_update > 0 and update_step < start_update:
            return rewards, {
                "pendulum_potential_shaping_active": 0.0,
                "pendulum_potential_shaping_start_update": float(start_update),
                "pendulum_potential_shaping_weight": float(self.cfg.pendulum_potential_shaping_weight),
                "pendulum_potential_shaping_abs_theta_low": float(
                    self.cfg.pendulum_potential_shaping_abs_theta_low
                ),
                "pendulum_potential_shaping_abs_theta_high": float(
                    self.cfg.pendulum_potential_shaping_abs_theta_high
                ),
                "pendulum_potential_shaping_velocity_limit": float(
                    self.cfg.pendulum_potential_shaping_velocity_limit
                ),
            }
        with torch.no_grad():
            obs_for_query = observations.to(device=self.device, dtype=torch.float32)
            next_obs_for_query = next_observations.to(device=self.device, dtype=torch.float32)
            potential = self.pendulum_potential.query(obs_for_query).to(device=rewards.device, dtype=rewards.dtype)
            next_potential = self.pendulum_potential.query(next_obs_for_query).to(
                device=rewards.device,
                dtype=rewards.dtype,
            )
            potential = potential.reshape_as(rewards)
            next_potential = next_potential.reshape_as(rewards)
            not_done = 1.0 - dones.to(device=rewards.device, dtype=rewards.dtype).reshape_as(rewards)
            shaping = float(self.cfg.pendulum_potential_shaping_weight) * (
                float(self.cfg.gamma) * next_potential * not_done - potential
            )
            abs_theta = torch.atan2(obs_for_query[..., 1], obs_for_query[..., 0]).abs()
            gate = (
                (abs_theta >= float(self.cfg.pendulum_potential_shaping_abs_theta_low))
                & (abs_theta <= float(self.cfg.pendulum_potential_shaping_abs_theta_high))
            )
            velocity_limit = float(self.cfg.pendulum_potential_shaping_velocity_limit)
            if velocity_limit > 0.0:
                gate = gate & (obs_for_query[..., 2].abs() <= velocity_limit)
            gate_tensor = gate.to(device=rewards.device, dtype=rewards.dtype).reshape_as(rewards)
            shaping = shaping * gate_tensor
            shaped_rewards = rewards + shaping
            detached = shaping.detach()
            metrics = {
                "pendulum_potential_shaping_active": 1.0,
                "pendulum_potential_shaping_start_update": float(start_update),
                "pendulum_potential_shaping_weight": float(self.cfg.pendulum_potential_shaping_weight),
                "pendulum_potential_shaping_abs_theta_low": float(
                    self.cfg.pendulum_potential_shaping_abs_theta_low
                ),
                "pendulum_potential_shaping_abs_theta_high": float(
                    self.cfg.pendulum_potential_shaping_abs_theta_high
                ),
                "pendulum_potential_shaping_velocity_limit": float(
                    self.cfg.pendulum_potential_shaping_velocity_limit
                ),
                "pendulum_potential_shaping_gate_fraction": float(gate_tensor.detach().mean().cpu()),
                "pendulum_potential_shaping_mean": float(detached.mean().cpu()),
                "pendulum_potential_shaping_abs_mean": float(detached.abs().mean().cpu()),
                "pendulum_potential_shaping_min": float(detached.min().cpu()),
                "pendulum_potential_shaping_max": float(detached.max().cpu()),
                "pendulum_potential_value_mean": float(potential.detach().mean().cpu()),
                "pendulum_potential_next_value_mean": float(next_potential.detach().mean().cpu()),
            }
            return shaped_rewards, metrics

    def _sacn_common_target_inputs(self, data: Any, update_step: int) -> dict[str, Any]:
        rewards = data.trajectory_rewards
        rewards, shaping_metrics = self._shape_pendulum_rewards(
            observations=data.trajectory_observations,
            next_observations=data.trajectory_next_observations,
            rewards=rewards,
            dones=data.trajectory_dones,
            update_step=update_step,
        )
        if self.reward_scaler is not None:
            rewards = self.reward_scaler.scale_tensor(rewards)
        rewards = rewards.squeeze(-1)
        dones = data.trajectory_dones.squeeze(-1).to(dtype=rewards.dtype)
        batch_size, n_step = rewards.shape

        weights, log_omega, log_importance_clip = self._sacn_importance_weights(data)
        weights, horizon_mask, horizon_ess_fraction = self._sacn_apply_horizon_support(weights)
        weights, horizon_loss_weights = self._sacn_apply_target_mode_and_decay(weights, horizon_mask)
        successor_observations = self._normalize_batched_observations(data.trajectory_next_observations)
        successor_actions, entropy_by_tau, entropy_sample_counts = self._sacn_successor_policy_samples(
            successor_observations
        )
        offsets, discounts = self._sacn_offsets_and_discounts(rewards, dones, entropy_by_tau)

        return {
            "batch_size": int(batch_size),
            "n_step": int(n_step),
            "weights": weights,
            "horizon_mask": horizon_mask,
            "horizon_loss_weights": horizon_loss_weights,
            "horizon_ess_fraction": horizon_ess_fraction,
            "log_omega": log_omega,
            "log_importance_clip": log_importance_clip,
            "offsets": offsets,
            "discounts": discounts,
            "successor_observations": successor_observations.reshape(batch_size * n_step, -1),
            "successor_actions": successor_actions.reshape(batch_size * n_step, -1),
            "entropy_sample_counts": entropy_sample_counts,
            "shaping_metrics": shaping_metrics,
        }

    def _normalize_batched_observations(self, observations: torch.Tensor) -> torch.Tensor:
        original_shape = observations.shape
        flat = observations.reshape(-1, original_shape[-1])
        return self._normalize_obs_tensor(flat).reshape(original_shape)

    def _sacn_apply_horizon_support(
        self,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight_sum = weights.sum(dim=0)
        weight_square_sum = weights.pow(2).sum(dim=0).clamp_min(1e-12)
        horizon_ess_fraction = (weight_sum.pow(2) / weight_square_sum) / float(weights.shape[0])
        horizon_mask = torch.ones_like(horizon_ess_fraction, dtype=weights.dtype)
        min_ess_fraction = float(self.cfg.sacn_min_horizon_ess_fraction)
        if min_ess_fraction > 0.0:
            horizon_mask_bool = horizon_ess_fraction >= min_ess_fraction
            horizon_mask_bool[0] = True
            horizon_mask = horizon_mask_bool.to(dtype=weights.dtype)
            weights = weights * horizon_mask.view(1, -1)
        return weights, horizon_mask, horizon_ess_fraction

    @staticmethod
    def _sacn_weighted_horizon_loss(
        per_horizon_loss: torch.Tensor,
        weights: torch.Tensor,
        horizon_loss_weights: torch.Tensor,
        replay_importance_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        horizon_weight_sum = horizon_loss_weights.to(dtype=per_horizon_loss.dtype).sum().clamp_min(1.0)
        normalizer = float(per_horizon_loss.shape[0]) * horizon_weight_sum
        weighted_loss = per_horizon_loss * weights
        if replay_importance_weights is not None:
            weighted_loss = weighted_loss * replay_importance_weights.reshape(-1, 1)
        return weighted_loss.sum() / normalizer

    def _sacn_apply_target_mode_and_decay(
        self,
        weights: torch.Tensor,
        horizon_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        horizon_loss_weights = horizon_mask.to(dtype=weights.dtype)
        if self.cfg.sacn_target_mode == "fast_last":
            active_indices = torch.nonzero(horizon_mask > 0.0, as_tuple=False).flatten()
            selected = torch.zeros_like(horizon_loss_weights)
            selected[0] = 1.0
            if active_indices.numel() > 0:
                selected[active_indices[-1]] = 1.0
            horizon_loss_weights = horizon_loss_weights * selected

        horizon_lambda = float(self.cfg.sacn_horizon_lambda)
        if horizon_lambda < 1.0:
            powers = torch.arange(weights.shape[1], dtype=weights.dtype, device=weights.device)
            decay = torch.pow(torch.full((), horizon_lambda, dtype=weights.dtype, device=weights.device), powers)
            horizon_loss_weights = horizon_loss_weights * decay

        weights = weights * horizon_loss_weights.view(1, -1)
        return weights, horizon_loss_weights

    def _sacn_successor_policy_samples(
        self,
        successor_observations: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[int]]:
        batch_size, n_step, obs_dim = successor_observations.shape
        flat_observations = successor_observations.reshape(batch_size * n_step, obs_dim)
        entropy_sample_counts = self._sacn_entropy_sample_counts(n_step)
        max_samples = max(entropy_sample_counts)
        with torch.no_grad():
            expanded_observations = flat_observations.repeat(max_samples, 1)
            sampled_actions, sampled_log_probs, _mean = self.actor.get_action(expanded_observations)

        action_stack = sampled_actions.reshape(max_samples, batch_size, n_step, -1)
        entropy_stack = (-sampled_log_probs).reshape(max_samples, batch_size, n_step)
        entropy_by_tau = [entropy_stack[:sample_count].mean(dim=0) for sample_count in entropy_sample_counts]
        return action_stack[0], entropy_by_tau, entropy_sample_counts

    def _sacn_offsets_and_discounts(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        entropy_by_tau: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_step = rewards.shape[1]
        dtype = rewards.dtype
        device = rewards.device
        powers = torch.arange(n_step, dtype=dtype, device=device)
        reward_discounts = torch.pow(torch.full((), float(self.cfg.gamma), dtype=dtype, device=device), powers)
        bootstrap_discounts = torch.pow(
            torch.full((), float(self.cfg.gamma), dtype=dtype, device=device),
            powers + 1.0,
        )

        discounted_rewards = torch.cumsum(rewards * reward_discounts.view(1, -1), dim=1)
        not_done = 1.0 - dones
        offsets = []
        for tau_index, entropy_estimate in enumerate(entropy_by_tau):
            if self.cfg.sacn_non_soft_targets:
                entropy_terms = torch.zeros_like(rewards)
            else:
                entropy_terms = (
                    self._alpha_tensor
                    * entropy_estimate
                    * bootstrap_discounts.view(1, -1)
                    * not_done
                )
            entropy_prefix = torch.cumsum(entropy_terms, dim=1)
            offsets.append(discounted_rewards[:, tau_index] + entropy_prefix[:, tau_index])
        offsets_tensor = torch.stack(offsets, dim=1)
        discounts = bootstrap_discounts.view(1, -1) * not_done
        return offsets_tensor, discounts

    def _sacn_importance_weights(self, data: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if self.cfg.sacn_importance_mode == "none":
                actions = data.trajectory_actions
                batch_size, n_step = actions.shape[:2]
                weights = torch.ones(
                    (batch_size, n_step),
                    dtype=actions.dtype,
                    device=actions.device,
                )
                log_omega = torch.zeros_like(weights)
                log_importance_clip = torch.zeros((), dtype=actions.dtype, device=actions.device)
                return weights, log_omega, log_importance_clip

            trajectory_observations = self._normalize_batched_observations(data.trajectory_observations)
            batch_size, n_step, obs_dim = trajectory_observations.shape
            flat_observations = trajectory_observations.reshape(batch_size * n_step, obs_dim)
            flat_actions = data.trajectory_actions.reshape(batch_size * n_step, -1)
            current_log_probs = self._actor_log_prob_from_normalized_obs(flat_observations, flat_actions).view(
                batch_size,
                n_step,
            )
            behavior_log_probs = data.trajectory_action_log_probs.squeeze(-1).to(
                dtype=current_log_probs.dtype,
                device=current_log_probs.device,
            )
            if not bool(torch.isfinite(behavior_log_probs).all().item()):
                raise ValueError("SACn sequence batch contains non-finite behavior action log-probabilities.")

            log_ratios = current_log_probs - behavior_log_probs
            log_omega = torch.zeros_like(log_ratios)
            if n_step > 1:
                log_omega[:, 1:] = torch.cumsum(log_ratios[:, 1:], dim=1)

            log_importance_clip = self._log_quantile_from_log_values(
                log_omega.reshape(-1),
                float(self.cfg.sacn_importance_quantile),
            )
            clipped_log_omega = torch.minimum(log_omega, log_importance_clip)
            denominator = clipped_log_omega.max(dim=0, keepdim=True).values
            weights = torch.exp(torch.clamp(clipped_log_omega - denominator, min=-80.0, max=0.0))
            return weights, log_omega, log_importance_clip

    def _actor_log_prob_from_normalized_obs(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.actor(observations)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        action_scale = torch.clamp(self.actor.action_scale.abs(), min=1e-6).view(1, -1)
        action_bias = self.actor.action_bias.view(1, -1)
        squashed = ((actions - action_bias) / action_scale).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = 0.5 * (torch.log1p(squashed) - torch.log1p(-squashed))
        log_prob = normal.log_prob(pre_tanh)
        log_prob -= torch.log(action_scale * (1.0 - squashed.pow(2)) + 1e-6)
        return log_prob.sum(dim=1, keepdim=True)

    def _sacn_entropy_sample_counts(self, n_step: int) -> list[int]:
        if not self.cfg.sacn_tau_entropy:
            return [1 for _tau in range(n_step)]
        max_samples = int(self.cfg.sacn_max_entropy_samples)
        gamma = float(self.cfg.gamma)
        counts: list[int] = []
        for tau in range(1, n_step + 1):
            if abs(gamma - 1.0) < 1e-8:
                estimate = float(tau)
            else:
                estimate = (1.0 - gamma ** (2 * tau)) / (1.0 - gamma**2)
            counts.append(max(1, min(max_samples, int(round(estimate)))))
        return counts

    @staticmethod
    def _log_quantile_from_log_values(log_values: torch.Tensor, quantile: float) -> torch.Tensor:
        sorted_log_values = torch.sort(log_values.reshape(-1)).values
        count = int(sorted_log_values.numel())
        if count == 0:
            raise ValueError("Cannot compute quantile of an empty tensor.")
        if count == 1:
            return sorted_log_values[0]
        position = (count - 1) * float(quantile)
        lower_index = int(math.floor(position))
        upper_index = int(math.ceil(position))
        fraction = float(position - lower_index)
        if lower_index == upper_index or fraction <= 0.0:
            return sorted_log_values[lower_index]
        if fraction >= 1.0:
            return sorted_log_values[upper_index]
        return torch.logaddexp(
            sorted_log_values[lower_index] + math.log(1.0 - fraction),
            sorted_log_values[upper_index] + math.log(fraction),
        )

    def _sacn_metrics(
        self,
        targets: torch.Tensor,
        weights: torch.Tensor,
        log_omega: torch.Tensor,
        log_importance_clip: torch.Tensor,
        entropy_sample_counts: list[int],
        horizon_ess_fraction: torch.Tensor,
        horizon_mask: torch.Tensor,
        horizon_loss_weights: torch.Tensor,
    ) -> dict[str, float]:
        if not self._collect_update_metrics:
            return {}
        detached_targets = targets.detach()
        detached_weights = weights.detach()
        detached_log_omega = log_omega.detach()
        detached_ess_fraction = horizon_ess_fraction.detach()
        detached_horizon_mask = horizon_mask.detach()
        detached_horizon_loss_weights = horizon_loss_weights.detach()
        horizon_loss_weight_sum = detached_horizon_loss_weights.sum()
        last_weights = detached_weights[:, -1]
        return {
            "sacn_n_step": float(weights.shape[1]),
            "sacn_target_first_mean": float(detached_targets[:, 0].mean().cpu()),
            "sacn_target_last_mean": float(detached_targets[:, -1].mean().cpu()),
            "sacn_weight_mean": float(detached_weights.mean().cpu()),
            "sacn_weight_min": float(detached_weights.min().cpu()),
            "sacn_weight_max": float(detached_weights.max().cpu()),
            "sacn_weight_last_mean": float(last_weights.mean().cpu()),
            "sacn_weight_last_collapse_fraction": float((last_weights <= 1e-4).to(torch.float32).mean().cpu()),
            "sacn_horizon_ess_first_fraction": float(detached_ess_fraction[0].cpu()),
            "sacn_horizon_ess_last_fraction": float(detached_ess_fraction[-1].cpu()),
            "sacn_horizon_ess_min_fraction": float(detached_ess_fraction.min().cpu()),
            "sacn_horizon_active_count": float(detached_horizon_mask.sum().cpu()),
            "sacn_horizon_loss_weight_sum": float(horizon_loss_weight_sum.cpu()),
            "sacn_horizon_last_loss_weight": float(
                detached_horizon_loss_weights[-1].cpu()
            ),
            "sacn_horizon_last_loss_weight_share": float(
                (
                    detached_horizon_loss_weights[-1]
                    / horizon_loss_weight_sum.clamp_min(1e-12)
                ).cpu()
            ),
            "sacn_horizon_last_active": float(detached_horizon_mask[-1].cpu()),
            "sacn_log_omega_mean": float(detached_log_omega.mean().cpu()),
            "sacn_log_omega_std": float(detached_log_omega.std(unbiased=False).cpu()),
            "sacn_log_importance_clip": float(log_importance_clip.detach().cpu()),
            "sacn_entropy_samples_max": float(max(entropy_sample_counts)),
            "sacn_importance_is_density": 1.0 if self.cfg.sacn_importance_mode == "density" else 0.0,
            "sacn_non_soft_targets": 1.0 if self.cfg.sacn_non_soft_targets else 0.0,
            "sacn_target_mode_is_fast_last": 1.0 if self.cfg.sacn_target_mode == "fast_last" else 0.0,
            "sacn_horizon_lambda": float(self.cfg.sacn_horizon_lambda),
        }

    def _prepare_reference_actions(
        self,
        reference_actions: Any | None,
        sampled_actions: torch.Tensor,
        observations: torch.Tensor,
    ) -> torch.Tensor | None:
        if (
            reference_actions is None
            or not needs_actor_reference_actions(self.cfg)
        ):
            return None
        actions = torch.as_tensor(reference_actions, dtype=observations.dtype, device=self.device)
        return actions.reshape_as(sampled_actions).clamp(
            self.actor.action_bias - self.actor.action_scale,
            self.actor.action_bias + self.actor.action_scale,
        )

    def _prepare_reference_critic_actions(
        self,
        reference_actions: Any | None,
        sampled_actions: torch.Tensor,
        observations: torch.Tensor,
    ) -> torch.Tensor | None:
        if (
            reference_actions is None
            or self.cfg.reference_critic_mode == "none"
            or self.cfg.reference_critic_weight <= 0.0
        ):
            return None
        actions = torch.as_tensor(reference_actions, dtype=observations.dtype, device=self.device)
        return actions.reshape_as(sampled_actions).clamp(
            self.actor.action_bias - self.actor.action_scale,
            self.actor.action_bias + self.actor.action_scale,
        )

    def _reference_critic_margin_losses(
        self,
        observations: torch.Tensor,
        reference_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        if self.cfg.reference_critic_mode != "margin":
            raise ValueError(f"Unknown reference critic mode: {self.cfg.reference_critic_mode}")

        with torch.no_grad():
            _sampled_actions, _log_pi, actor_mean = self.actor.get_action(observations)
            actor_mean = actor_mean.detach()

        q1_reference = self.q1(observations, reference_actions).view(-1, 1)
        q2_reference = self.q2(observations, reference_actions).view(-1, 1)
        q1_actor = self.q1(observations, actor_mean).view(-1, 1)
        q2_actor = self.q2(observations, actor_mean).view(-1, 1)
        q1_advantage = q1_reference - q1_actor
        q2_advantage = q2_reference - q2_actor
        margin = float(self.cfg.reference_critic_margin)
        loss1 = F.relu(margin - q1_advantage).mean()
        loss2 = F.relu(margin - q2_advantage).mean()
        min_reference = torch.min(q1_reference, q2_reference)
        min_actor = torch.min(q1_actor, q2_actor)
        min_advantage = min_reference - min_actor

        metrics = {
            "reference_critic_margin_loss": float((0.5 * (loss1 + loss2)).detach().cpu()),
            "reference_critic_q_advantage_mean": float(min_advantage.detach().mean().cpu()),
            "reference_critic_q_advantage_positive_fraction": float(
                (min_advantage.detach() > 0.0).float().mean().cpu()
            ),
            "reference_critic_margin_violation_fraction": float(
                (min_advantage.detach() < margin).float().mean().cpu()
            ),
            "reference_critic_q_actor_mean": float(min_actor.detach().mean().cpu()),
            "reference_critic_q_reference_mean": float(min_reference.detach().mean().cpu()),
            "reference_critic_action_abs_error_mean": float(
                (actor_mean - reference_actions).detach().abs().mean().cpu()
            ),
        }
        return loss1, loss2, metrics

    def _conservative_critic_losses(
        self,
        observations: torch.Tensor,
        data_actions: torch.Tensor,
        qf1_data_values: torch.Tensor,
        qf2_data_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        batch_size = observations.shape[0]
        num_random = int(self.cfg.cql_num_random_actions)
        action_low = (self.actor.action_bias - self.actor.action_scale).view(1, 1, -1)
        action_high = (self.actor.action_bias + self.actor.action_scale).view(1, 1, -1)
        random_actions = action_low + torch.rand(
            batch_size,
            num_random,
            data_actions.shape[1],
            device=observations.device,
            dtype=observations.dtype,
        ) * (action_high - action_low)
        expanded_observations = observations.unsqueeze(1).expand(-1, num_random, -1).reshape(batch_size * num_random, -1)
        random_actions_flat = random_actions.reshape(batch_size * num_random, -1)
        q1_candidates = [self.q1(expanded_observations, random_actions_flat).view(batch_size, num_random)]
        q2_candidates = [self.q2(expanded_observations, random_actions_flat).view(batch_size, num_random)]

        if self.cfg.cql_include_policy_actions:
            with torch.no_grad():
                policy_actions, policy_log_probs, _mean = self.actor.get_action(observations)
            policy_q1 = self.q1(observations, policy_actions.detach()).view(batch_size, 1)
            policy_q2 = self.q2(observations, policy_actions.detach()).view(batch_size, 1)
            log_probs = policy_log_probs.detach().view(batch_size, 1)
            q1_candidates.append(policy_q1 - log_probs)
            q2_candidates.append(policy_q2 - log_probs)

        temperature = float(self.cfg.cql_temperature)
        q1_cat = torch.cat(q1_candidates, dim=1)
        q2_cat = torch.cat(q2_candidates, dim=1)
        conservative_q1 = torch.logsumexp(q1_cat / temperature, dim=1).mean() * temperature
        conservative_q2 = torch.logsumexp(q2_cat / temperature, dim=1).mean() * temperature
        data_q1 = qf1_data_values.view(batch_size).mean()
        data_q2 = qf2_data_values.view(batch_size).mean()
        loss1 = conservative_q1 - data_q1
        loss2 = conservative_q2 - data_q2
        metrics = {
            "cql_q1_loss": float(loss1.detach().cpu()),
            "cql_q2_loss": float(loss2.detach().cpu()),
            "cql_q1_conservative_mean": float(conservative_q1.detach().cpu()),
            "cql_q2_conservative_mean": float(conservative_q2.detach().cpu()),
            "cql_q1_data_mean": float(data_q1.detach().cpu()),
            "cql_q2_data_mean": float(data_q2.detach().cpu()),
            "cql_num_candidates": float(q1_cat.shape[1]),
        }
        return loss1, loss2, metrics

    @staticmethod
    def _actor_mean_logit_l2_penalty(
        mean_logits: torch.Tensor,
        *,
        excess_threshold: float = 0.0,
        collect_metrics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Penalize all logits or only extreme excess, without tanh attenuation."""

        threshold = float(excess_threshold)
        absolute_logits = mean_logits.abs()
        excess = torch.clamp(absolute_logits - threshold, min=0.0)
        penalty = excess.square().mean() if threshold > 0.0 else mean_logits.square().mean()
        if not collect_metrics:
            return penalty, {}

        detached_logits = mean_logits.detach()
        detached_excess = torch.clamp(detached_logits.abs() - threshold, min=0.0)
        normalized_mean_actions = torch.tanh(detached_logits)
        saturation_threshold = 0.995
        metrics = {
            "actor_mean_logit_l2_penalty_raw": float(penalty.detach().cpu()),
            "actor_mean_logit_excess_threshold": threshold,
            "actor_mean_logit_excess_fraction": float(
                (detached_excess > 0.0)
                .to(dtype=detached_logits.dtype)
                .mean()
                .cpu()
            ),
            "actor_mean_logit_excess_abs_mean": float(detached_excess.mean().cpu()),
            "actor_mean_logit_abs_mean": float(detached_logits.abs().mean().cpu()),
            "actor_mean_logit_abs_max": float(detached_logits.abs().max().cpu()),
            "actor_deterministic_action_saturation_fraction": float(
                (normalized_mean_actions.abs() >= saturation_threshold)
                .to(dtype=normalized_mean_actions.dtype)
                .mean()
                .cpu()
            ),
            "actor_mean_tanh_derivative_mean": float(
                (1.0 - normalized_mean_actions.square()).mean().cpu()
            ),
        }
        return penalty, metrics

    def _self_imitation_actor_loss(
        self,
        observations: torch.Tensor,
        replay_actions: torch.Tensor,
        actor_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        action_low = (self.actor.action_bias - self.actor.action_scale).view(1, -1)
        action_high = (self.actor.action_bias + self.actor.action_scale).view(1, -1)
        clipped_replay_actions = replay_actions.reshape(actor_actions.shape).clamp(action_low, action_high)

        with torch.no_grad():
            q_actor = torch.min(
                self.q1(observations, actor_actions.detach()),
                self.q2(observations, actor_actions.detach()),
            ).view(-1, 1)
            q_replay = torch.min(
                self.q1(observations, clipped_replay_actions),
                self.q2(observations, clipped_replay_actions),
            ).view(-1, 1)
            advantage = q_replay - q_actor
            margin = float(self.cfg.self_imitation_margin)
            positive = advantage > margin
            log_weight_cap = math.log(float(self.cfg.self_imitation_max_weight))
            logits = ((advantage - margin) / float(self.cfg.self_imitation_temperature)).clamp(
                min=-80.0,
                max=log_weight_cap,
            )
            weights = torch.where(positive, torch.exp(logits), torch.zeros_like(logits))

        loss_type = self.cfg.self_imitation_loss_type
        if loss_type == "mse":
            action_scale = torch.clamp(self.actor.action_scale.abs(), min=1e-6).view(1, -1)
            per_sample_loss = ((actor_actions - clipped_replay_actions) / action_scale).pow(2).mean(
                dim=1,
                keepdim=True,
            )
        elif loss_type == "log_prob":
            log_prob = self._actor_log_prob_from_normalized_obs(observations, clipped_replay_actions).view(-1, 1)
            per_sample_loss = -log_prob
        else:
            raise ValueError(f"Unknown self_imitation_loss_type: {loss_type}")

        weight_sum = weights.sum()
        if float(weight_sum.detach().cpu()) > 0.0:
            loss = (weights * per_sample_loss).sum() / weight_sum.clamp_min(1.0)
        else:
            loss = per_sample_loss.sum() * 0.0

        selected = positive.to(per_sample_loss.dtype)
        selected_count = selected.sum()
        detached_action_error = (actor_actions.detach() - clipped_replay_actions).abs()
        if float(selected_count.detach().cpu()) > 0.0:
            masked_action_abs_error = (detached_action_error * selected).sum() / (
                selected_count.clamp_min(1.0) * actor_actions.shape[1]
            )
        else:
            masked_action_abs_error = torch.zeros((), dtype=per_sample_loss.dtype, device=per_sample_loss.device)

        metrics = {
            "self_imitation_actor_loss": float(loss.detach().cpu()),
            "self_imitation_loss_type_is_mse": 1.0 if loss_type == "mse" else 0.0,
            "self_imitation_loss_type_is_log_prob": 1.0 if loss_type == "log_prob" else 0.0,
            "self_imitation_mask_fraction": float(selected.detach().mean().cpu()),
            "self_imitation_q_advantage_mean": float(advantage.detach().mean().cpu()),
            "self_imitation_q_advantage_positive_fraction": float((advantage.detach() > 0.0).float().mean().cpu()),
            "self_imitation_q_actor_mean": float(q_actor.detach().mean().cpu()),
            "self_imitation_q_replay_mean": float(q_replay.detach().mean().cpu()),
            "self_imitation_weight_mean": float(weights.detach().mean().cpu()),
            "self_imitation_weight_max": float(weights.detach().max().cpu()),
            "self_imitation_action_abs_error_mean": float(detached_action_error.mean().cpu()),
            "self_imitation_action_abs_error_masked_mean": float(masked_action_abs_error.detach().cpu()),
        }
        return loss, metrics

    def _filter_sac_actor_loss(
        self,
        *,
        observations: torch.Tensor,
        policy_mean: torch.Tensor,
        reference_actions: torch.Tensor | None,
        per_sample_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, bool, dict[str, float]]:
        mode = self.cfg.sac_actor_filter_mode
        if mode == "none":
            return per_sample_loss.mean(), True, {
                "sac_actor_filter_selected_fraction": 1.0,
                "sac_actor_filter_advantage_mean": 0.0,
                "sac_actor_filter_online_target": 0.0,
            }
        if reference_actions is None:
            raise ValueError(f"sac_actor_filter_mode={mode!r} requires reference actions")

        critic_networks = list(self.q_networks)
        if mode == "reference_online_target_unanimous":
            critic_networks.extend(self.q_target_networks)
        elif mode != "reference_online_unanimous":
            raise ValueError(f"Unknown SAC actor filter mode: {mode}")

        with torch.no_grad():
            advantages = torch.stack(
                [
                    critic(observations, policy_mean.detach()).view(-1, 1)
                    - critic(observations, reference_actions).view(-1, 1)
                    for critic in critic_networks
                ],
                dim=0,
            )
            robust_advantage = advantages.min(dim=0).values
            selected = robust_advantage > float(self.cfg.sac_actor_filter_margin)
            selected_float = selected.to(dtype=per_sample_loss.dtype)
        selected_count = selected_float.sum()
        has_selected = bool(selected.any().item())
        if has_selected:
            loss = (per_sample_loss * selected_float).sum() / selected_count
        else:
            loss = per_sample_loss.sum() * 0.0
        return loss, has_selected, {
            "sac_actor_filter_selected_fraction": float(selected_float.mean().cpu()),
            "sac_actor_filter_advantage_mean": float(robust_advantage.mean().cpu()),
            "sac_actor_filter_advantage_selected_mean": float(
                robust_advantage[selected].mean().cpu() if has_selected else 0.0
            ),
            "sac_actor_filter_online_target": float(
                mode == "reference_online_target_unanimous"
            ),
        }

    def _critic_search_actor_loss(
        self,
        observations: torch.Tensor,
        actor_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        num_actions = int(self.cfg.critic_search_num_actions)
        best_actions, best_q, actor_q = self._critic_search_best_actions(
            observations=observations,
            actor_actions=actor_actions,
            num_actions=num_actions,
        )
        if self.cfg.critic_search_filter_mode in {
            "unanimous_advantage",
            "online_target_unanimous_advantage",
        }:
            critic_networks = list(self.q_networks)
            if (
                self.cfg.critic_search_filter_mode
                == "online_target_unanimous_advantage"
            ):
                critic_networks.extend(self.q_target_networks)
            with torch.no_grad():
                advantage = torch.stack(
                    [
                        q_network(observations, best_actions).view(-1, 1)
                        - q_network(observations, actor_actions.detach()).view(-1, 1)
                        for q_network in critic_networks
                    ],
                    dim=0,
                ).min(dim=0).values
        elif self.cfg.critic_search_filter_mode == "clipped_value":
            advantage = best_q - actor_q
        else:
            raise ValueError(
                f"Unknown critic search actor filter mode: {self.cfg.critic_search_filter_mode}"
            )
        selected = advantage > float(self.cfg.critic_search_margin)

        loss_type = self.cfg.critic_search_actor_loss_type
        if loss_type == "mse":
            action_scale = torch.clamp(self.actor.action_scale.abs(), min=1e-6).view(1, -1)
            per_sample_loss = (
                ((actor_actions - best_actions) / action_scale)
                .pow(2)
                .mean(dim=1, keepdim=True)
            )
        elif loss_type == "log_prob":
            per_sample_loss = -self._actor_log_prob_from_normalized_obs(
                observations,
                best_actions,
            ).view(-1, 1)
        else:
            raise ValueError(f"Unknown critic search actor loss type: {loss_type}")
        selected_float = selected.to(dtype=per_sample_loss.dtype)
        selected_count = selected_float.sum()
        has_selected = bool(selected.any().item())
        if has_selected:
            loss = (selected_float * per_sample_loss).sum() / selected_count
        else:
            loss = per_sample_loss.sum() * 0.0

        metrics = {
            "critic_search_actor_loss": float(loss.detach().cpu()),
            "critic_search_advantage_mean": float(advantage.mean().cpu()),
            "critic_search_advantage_selected_mean": float(
                advantage[selected].mean().cpu() if has_selected else 0.0
            ),
            "critic_search_selected_fraction": float(selected_float.mean().cpu()),
            "critic_search_action_abs_error_mean": float(
                (actor_actions.detach() - best_actions).abs().mean().cpu()
            ),
            "critic_search_best_q_mean": float(best_q.mean().cpu()),
            "critic_search_actor_q_mean": float(actor_q.mean().cpu()),
            "critic_search_num_actions": float(num_actions),
            "critic_search_margin": float(self.cfg.critic_search_margin),
            "critic_search_filter_is_unanimous": float(
                self.cfg.critic_search_filter_mode
                in {"unanimous_advantage", "online_target_unanimous_advantage"}
            ),
            "critic_search_filter_is_online_target": float(
                self.cfg.critic_search_filter_mode
                == "online_target_unanimous_advantage"
            ),
            "critic_search_loss_type_is_mse": 1.0 if loss_type == "mse" else 0.0,
            "critic_search_loss_type_is_log_prob": 1.0 if loss_type == "log_prob" else 0.0,
        }
        return loss, metrics

    def _reference_auxiliary_weight(self, update_step: int) -> float:
        stop_update = int(self.cfg.reference_auxiliary_stop_update)
        if stop_update > 0 and int(update_step) >= stop_update:
            return 0.0
        initial = float(self.cfg.reference_auxiliary_weight)
        final_cfg = self.cfg.reference_auxiliary_weight_final
        final = initial if final_cfg is None else float(final_cfg)
        decay_updates = int(self.cfg.reference_auxiliary_decay_updates)
        if decay_updates <= 0:
            return initial
        fraction = min(max(float(update_step) / float(decay_updates), 0.0), 1.0)
        return initial + fraction * (final - initial)

    def _actor_loss_gradient_alignment(
        self,
        *,
        sac_actor_loss: torch.Tensor,
        reference_actor_loss: torch.Tensor,
        sac_weight: float,
        reference_weight: float,
    ) -> dict[str, float]:
        named_parameters = [
            (name, parameter)
            for name, parameter in self.actor.named_parameters()
            if parameter.requires_grad
        ]
        parameters = [parameter for _name, parameter in named_parameters]
        sac_gradients = torch.autograd.grad(
            sac_actor_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        reference_gradients = torch.autograd.grad(
            reference_actor_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        dot = torch.zeros((), device=self.device)
        sac_squared = torch.zeros((), device=self.device)
        reference_squared = torch.zeros((), device=self.device)
        group_sac_squared = {
            group: torch.zeros((), device=self.device)
            for group in ("shared", "mean_head", "std_head")
        }
        group_reference_squared = {
            group: torch.zeros((), device=self.device)
            for group in ("shared", "mean_head", "std_head")
        }
        for (name, _parameter), sac_gradient, reference_gradient in zip(
            named_parameters,
            sac_gradients,
            reference_gradients,
        ):
            if name.startswith("mean_"):
                group = "mean_head"
            elif name.startswith("std_"):
                group = "std_head"
            else:
                group = "shared"
            if sac_gradient is not None:
                squared = torch.sum(sac_gradient.square())
                sac_squared = sac_squared + squared
                group_sac_squared[group] = group_sac_squared[group] + squared
            if reference_gradient is not None:
                squared = torch.sum(reference_gradient.square())
                reference_squared = reference_squared + squared
                group_reference_squared[group] = group_reference_squared[group] + squared
            if sac_gradient is not None and reference_gradient is not None:
                dot = dot + torch.sum(sac_gradient * reference_gradient)
        sac_norm = torch.sqrt(sac_squared)
        reference_norm = torch.sqrt(reference_squared)
        denominator = (sac_norm * reference_norm).clamp_min(1e-12)
        metrics = {
            "actor_sac_bc_gradient_cosine": float((dot / denominator).detach().cpu()),
            "actor_sac_gradient_norm": float(sac_norm.detach().cpu()),
            "actor_bc_gradient_norm": float(reference_norm.detach().cpu()),
            "actor_base_weighted_sac_gradient_norm": float(
                (sac_norm * float(sac_weight)).detach().cpu()
            ),
            "actor_weighted_sac_gradient_norm": float((sac_norm * float(sac_weight)).detach().cpu()),
            "actor_weighted_bc_gradient_norm": float(
                (reference_norm * float(reference_weight)).detach().cpu()
            ),
        }
        for group in ("shared", "mean_head", "std_head"):
            metrics[f"actor_sac_gradient_norm_{group}"] = float(
                torch.sqrt(group_sac_squared[group]).detach().cpu()
            )
            metrics[f"actor_bc_gradient_norm_{group}"] = float(
                torch.sqrt(group_reference_squared[group]).detach().cpu()
            )
        return metrics

    @staticmethod
    def _safe_actor_gradient_balance_ratio(
        alignment_metrics: dict[str, float],
    ) -> float | None:
        """Return the BC-to-base-SAC gradient ratio, or None when it is not usable."""

        sac_norm = float(alignment_metrics["actor_base_weighted_sac_gradient_norm"])
        bc_norm = float(alignment_metrics["actor_weighted_bc_gradient_norm"])
        if (
            not math.isfinite(sac_norm)
            or not math.isfinite(bc_norm)
            or sac_norm <= 1e-12
            or bc_norm <= 1e-12
        ):
            return None
        ratio = bc_norm / sac_norm
        return ratio if math.isfinite(ratio) else None

    @staticmethod
    def _project_sac_gradient_off_bc(
        sac_gradients: Sequence[torch.Tensor | None],
        bc_gradients: Sequence[torch.Tensor | None],
        *,
        epsilon: float = 1e-12,
    ) -> tuple[list[torch.Tensor | None], dict[str, float]]:
        """Asymmetric PCGrad: remove only the SAC component opposing BC.

        Inputs are already weighted objective gradients.  If ``g_sac . g_bc < 0``,
        this returns ``g_sac - (g_sac . g_bc / ||g_bc||^2) g_bc``.  The caller adds
        the original BC gradient, so BC is authoritative and is never modified.
        ``None`` is treated as a zero gradient for the corresponding parameter.
        """

        if len(sac_gradients) != len(bc_gradients):
            raise ValueError("SAC and BC gradient sequences must have the same length")
        first_gradient = next(
            (
                gradient
                for gradient in (*sac_gradients, *bc_gradients)
                if gradient is not None
            ),
            None,
        )
        if first_gradient is None:
            return [None] * len(sac_gradients), {
                "actor_sac_bc_weighted_gradient_dot_before": 0.0,
                "actor_sac_bc_weighted_gradient_cosine_before": 0.0,
                "actor_sac_bc_weighted_gradient_dot_after": 0.0,
                "actor_sac_bc_weighted_gradient_cosine_after": 0.0,
                "actor_sac_bc_projection_coefficient": 0.0,
                "actor_sac_bc_projection_correction_norm": 0.0,
                "actor_sac_bc_projected_sac_gradient_norm": 0.0,
                "actor_sac_bc_projection_applied": 0.0,
            }

        dot = torch.zeros((), dtype=first_gradient.dtype, device=first_gradient.device)
        sac_squared = torch.zeros_like(dot)
        bc_squared = torch.zeros_like(dot)
        for sac_gradient, bc_gradient in zip(sac_gradients, bc_gradients):
            if sac_gradient is not None:
                sac_squared = sac_squared + torch.sum(sac_gradient.square())
            if bc_gradient is not None:
                bc_squared = bc_squared + torch.sum(bc_gradient.square())
            if sac_gradient is not None and bc_gradient is not None:
                if sac_gradient.shape != bc_gradient.shape:
                    raise ValueError("Paired SAC and BC gradients must have matching shapes")
                dot = dot + torch.sum(sac_gradient * bc_gradient)

        should_project = bool((dot < 0.0).item() and (bc_squared > epsilon).item())
        coefficient = dot / bc_squared if should_project else torch.zeros_like(dot)
        projected_gradients: list[torch.Tensor | None] = []
        projected_squared = torch.zeros_like(dot)
        correction_squared = torch.zeros_like(dot)
        dot_after = torch.zeros_like(dot)
        for sac_gradient, bc_gradient in zip(sac_gradients, bc_gradients):
            if sac_gradient is None and bc_gradient is None:
                projected_gradient = None
            elif bc_gradient is None:
                assert sac_gradient is not None
                projected_gradient = sac_gradient.clone()
            elif sac_gradient is None:
                projected_gradient = -coefficient * bc_gradient if should_project else None
            elif should_project:
                projected_gradient = sac_gradient - coefficient * bc_gradient
            else:
                projected_gradient = sac_gradient.clone()
            projected_gradients.append(projected_gradient)
            if projected_gradient is not None:
                projected_squared = projected_squared + torch.sum(projected_gradient.square())
                if bc_gradient is not None:
                    dot_after = dot_after + torch.sum(projected_gradient * bc_gradient)
            if should_project and bc_gradient is not None:
                correction_squared = correction_squared + torch.sum(
                    (coefficient * bc_gradient).square()
                )

        sac_norm = torch.sqrt(sac_squared)
        bc_norm = torch.sqrt(bc_squared)
        projected_norm = torch.sqrt(projected_squared)
        cosine_before_denominator = (sac_norm * bc_norm).clamp_min(epsilon)
        cosine_after_denominator = (projected_norm * bc_norm).clamp_min(epsilon)
        return projected_gradients, {
            "actor_sac_bc_weighted_gradient_dot_before": float(dot.detach().cpu()),
            "actor_sac_bc_weighted_gradient_cosine_before": float(
                (dot / cosine_before_denominator).detach().cpu()
            ),
            "actor_sac_bc_weighted_gradient_dot_after": float(dot_after.detach().cpu()),
            "actor_sac_bc_weighted_gradient_cosine_after": float(
                (dot_after / cosine_after_denominator).detach().cpu()
            ),
            "actor_sac_bc_projection_coefficient": float(coefficient.detach().cpu()),
            "actor_sac_bc_projection_correction_norm": float(
                torch.sqrt(correction_squared).detach().cpu()
            ),
            "actor_sac_bc_projected_sac_gradient_norm": float(projected_norm.detach().cpu()),
            "actor_sac_bc_projection_applied": float(should_project),
        }

    def _critic_search_symmetric_best_actions(
        self,
        observations: torch.Tensor,
        mirrored_observations: torch.Tensor,
        actor_actions: torch.Tensor,
        num_actions: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Search with each critic averaged over an exact Pendulum reflection."""
        batch_size, action_dim = actor_actions.shape
        if action_dim != 1:
            raise ValueError("Pendulum symmetric critic search requires one action dimension")
        action_low = (self.actor.action_bias - self.actor.action_scale).view(1, 1, action_dim)
        action_high = (self.actor.action_bias + self.actor.action_scale).view(1, 1, action_dim)
        fractions = torch.linspace(
            0.0,
            1.0,
            num_actions,
            dtype=observations.dtype,
            device=observations.device,
        ).view(1, num_actions, 1)
        candidate_actions = action_low + fractions * (action_high - action_low)
        candidate_actions = candidate_actions.expand(batch_size, -1, -1)

        with torch.no_grad():
            expanded_observations = observations.unsqueeze(1).expand(-1, num_actions, -1)
            expanded_mirrored = mirrored_observations.unsqueeze(1).expand(
                -1, num_actions, -1
            )
            flat_observations = expanded_observations.reshape(batch_size * num_actions, -1)
            flat_mirrored = expanded_mirrored.reshape(batch_size * num_actions, -1)
            flat_actions = candidate_actions.reshape(batch_size * num_actions, action_dim)
            symmetric_candidate_values = []
            symmetric_actor_values = []
            for q_network in self.q_networks:
                original_q = q_network(flat_observations, flat_actions).view(
                    batch_size, num_actions
                )
                mirrored_q = q_network(flat_mirrored, -flat_actions).view(
                    batch_size, num_actions
                )
                symmetric_candidate_values.append(0.5 * (original_q + mirrored_q))

                original_actor_q = q_network(
                    observations, actor_actions.detach()
                ).view(batch_size, 1)
                mirrored_actor_q = q_network(
                    mirrored_observations, -actor_actions.detach()
                ).view(batch_size, 1)
                symmetric_actor_values.append(
                    0.5 * (original_actor_q + mirrored_actor_q)
                )

            stacked_candidates = torch.stack(symmetric_candidate_values, dim=0)
            proposal_values = stacked_candidates.min(dim=0).values
            best_indices = proposal_values.argmax(dim=1)
            row_indices = torch.arange(batch_size, device=observations.device)
            best_actions = candidate_actions[row_indices, best_indices]

            critic_indices = torch.arange(
                len(symmetric_candidate_values), device=observations.device
            ).view(-1, 1)
            selected_values = stacked_candidates[
                critic_indices, row_indices.view(1, -1), best_indices.view(1, -1)
            ].unsqueeze(-1)
            actor_values = torch.stack(symmetric_actor_values, dim=0)
            robust_advantage = (selected_values - actor_values).min(dim=0).values
        return best_actions, robust_advantage

    def _critic_search_best_actions(
        self,
        observations: torch.Tensor,
        actor_actions: torch.Tensor,
        num_actions: int,
        critic_networks: Any | None = None,
        candidate_aggregation: str = "min",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, action_dim = actor_actions.shape
        action_low = (self.actor.action_bias - self.actor.action_scale).view(1, 1, action_dim)
        action_high = (self.actor.action_bias + self.actor.action_scale).view(1, 1, action_dim)
        if action_dim == 1:
            fractions = torch.linspace(
                0.0,
                1.0,
                num_actions,
                dtype=observations.dtype,
                device=observations.device,
            ).view(1, num_actions, 1)
            candidate_actions = action_low + fractions * (action_high - action_low)
            candidate_actions = candidate_actions.expand(batch_size, -1, -1)
        else:
            candidate_actions = action_low + torch.rand(
                batch_size,
                num_actions,
                action_dim,
                dtype=observations.dtype,
                device=observations.device,
            ) * (action_high - action_low)

        with torch.no_grad():
            expanded_observations = observations.unsqueeze(1).expand(-1, num_actions, -1)
            flat_observations = expanded_observations.reshape(batch_size * num_actions, -1)
            flat_actions = candidate_actions.reshape(batch_size * num_actions, action_dim)
            scoring_critics = list(self.q_networks) if critic_networks is None else list(critic_networks)
            candidate_q_values = [
                q_network(flat_observations, flat_actions).view(batch_size, num_actions)
                for q_network in scoring_critics
            ]
            stacked_candidate_q = torch.stack(candidate_q_values, dim=0)
            candidate_q = self._aggregate_critic_search_values(
                stacked_candidate_q, candidate_aggregation
            )
            best_indices = candidate_q.argmax(dim=1)
            row_indices = torch.arange(batch_size, device=observations.device)
            best_actions = candidate_actions[row_indices, best_indices]
            best_q = candidate_q[row_indices, best_indices].view(-1, 1)
            stacked_actor_q = torch.stack(
                [q_network(observations, actor_actions.detach()).view(-1, 1) for q_network in scoring_critics],
                dim=0,
            )
            actor_q = self._aggregate_critic_search_values(
                stacked_actor_q, candidate_aggregation
            )
        return best_actions, best_q, actor_q

    @staticmethod
    def _aggregate_critic_search_values(
        stacked_q: torch.Tensor, candidate_aggregation: str
    ) -> torch.Tensor:
        if candidate_aggregation == "min":
            return stacked_q.min(dim=0).values
        if candidate_aggregation == "mean":
            return stacked_q.mean(dim=0)
        if candidate_aggregation.startswith("mid_"):
            coefficient = float(candidate_aggregation.removeprefix("mid_"))
            disagreement = stacked_q.max(dim=0).values - stacked_q.min(dim=0).values
            return stacked_q.mean(dim=0) - coefficient * disagreement
        if candidate_aggregation.startswith("lcb_"):
            beta = float(candidate_aggregation.removeprefix("lcb_"))
            disagreement = stacked_q.max(dim=0).values - stacked_q.min(dim=0).values
            return stacked_q.mean(dim=0) - (0.5 + beta) * disagreement
        raise ValueError(f"Unknown candidate aggregation: {candidate_aggregation}")

    def _reference_auxiliary_actor_loss(
        self,
        observations: torch.Tensor,
        actor_actions: torch.Tensor,
        reference_actions: torch.Tensor,
        replay_sample_count: int | None = None,
        q_filter_active: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        action_scale = torch.clamp(self.actor.action_scale.abs(), min=1e-6).view(1, -1)
        action_error = (actor_actions - reference_actions) / action_scale
        per_sample_loss = action_error.pow(2).mean(dim=1, keepdim=True)

        with torch.no_grad():
            q_filter_mode = self.cfg.reference_auxiliary_q_filter_mode
            if q_filter_mode == "twin_min_difference":
                q_actor = torch.min(
                    self.q1(observations, actor_actions.detach()),
                    self.q2(observations, actor_actions.detach()),
                )
                q_reference = torch.min(
                    self.q1(observations, reference_actions),
                    self.q2(observations, reference_actions),
                )
                q_advantage = q_reference - q_actor
                q_filter_critic_count = 2
            elif q_filter_mode == "online_target_unanimous":
                q_filter_critics = [*self.q_networks, *self.q_target_networks]
                q_actor_values = torch.stack(
                    [
                        critic(observations, actor_actions.detach()).view(-1, 1)
                        for critic in q_filter_critics
                    ],
                    dim=0,
                )
                q_reference_values = torch.stack(
                    [
                        critic(observations, reference_actions).view(-1, 1)
                        for critic in q_filter_critics
                    ],
                    dim=0,
                )
                # A row is admitted only if the reference wins for every
                # paired critic.  Taking the minimum paired advantage is not
                # equivalent to subtracting two independently clipped values.
                q_advantage = (q_reference_values - q_actor_values).min(dim=0).values
                q_actor = q_actor_values.min(dim=0).values
                q_reference = q_reference_values.min(dim=0).values
                q_filter_critic_count = len(q_filter_critics)
            else:
                raise ValueError(
                    "Unknown reference auxiliary Q-filter mode: "
                    f"{q_filter_mode}"
                )
            if self.cfg.reference_auxiliary_mode in {
                "q_filtered_bc",
                "q_filtered_replay_bc",
            }:
                mask = (
                    (q_advantage > float(self.cfg.reference_auxiliary_margin)).to(per_sample_loss.dtype)
                    if q_filter_active
                    else torch.ones_like(per_sample_loss)
                )
                if self.cfg.reference_auxiliary_mode == "q_filtered_replay_bc":
                    if replay_sample_count is None:
                        raise ValueError(
                            "q_filtered_replay_bc requires replay_sample_count so anchor samples "
                            "can remain unconditional"
                        )
                    replay_sample_count = int(replay_sample_count)
                    if not 0 <= replay_sample_count <= int(mask.shape[0]):
                        raise ValueError(
                            "replay_sample_count must lie within the reference auxiliary batch"
                        )
                    mask[replay_sample_count:] = 1.0
            elif self.cfg.reference_auxiliary_mode == "bc":
                mask = torch.ones_like(per_sample_loss)
            else:
                raise ValueError(f"Unknown reference auxiliary mode: {self.cfg.reference_auxiliary_mode}")

        replay_count = int(mask.shape[0]) if replay_sample_count is None else int(replay_sample_count)
        anchor_count = int(mask.shape[0]) - replay_count
        selected = mask.sum()
        full_batch_normalization = (
            self.cfg.reference_auxiliary_mode == "q_filtered_replay_bc"
            and self.cfg.reference_auxiliary_replay_normalization == "full_batch_mean"
        )
        normalizer = (
            torch.as_tensor(
                float(replay_count + anchor_count),
                dtype=per_sample_loss.dtype,
                device=per_sample_loss.device,
            )
            if full_batch_normalization
            else selected
        )
        if float(normalizer.detach().cpu()) > 0.0:
            loss = (per_sample_loss * mask).sum() / normalizer.clamp_min(1.0)
        else:
            loss = per_sample_loss.sum() * 0.0
        if float(selected.detach().cpu()) > 0.0:
            masked_error = ((actor_actions - reference_actions).abs() * mask).sum() / (
                selected.clamp_min(1.0) * actor_actions.shape[1]
            )
        else:
            masked_error = torch.zeros((), dtype=per_sample_loss.dtype, device=per_sample_loss.device)

        replay_mask_fraction = (
            float(mask[:replay_count].detach().mean().cpu()) if replay_count > 0 else 0.0
        )
        anchor_mask_fraction = (
            float(mask[replay_count:].detach().mean().cpu()) if anchor_count > 0 else 0.0
        )
        metrics = {
            "reference_actor_bc_loss": float(loss.detach().cpu()),
            "reference_actor_bc_mask_fraction": float(mask.detach().mean().cpu()),
            "reference_actor_bc_replay_mask_fraction": replay_mask_fraction,
            "reference_actor_bc_anchor_mask_fraction": anchor_mask_fraction,
            "reference_actor_bc_selected_count": float(selected.detach().cpu()),
            "reference_actor_bc_replay_selected_count": float(
                mask[:replay_count].detach().sum().cpu()
            ),
            "reference_actor_bc_replay_count": float(replay_count),
            "reference_actor_bc_anchor_count": float(anchor_count),
            "reference_actor_bc_normalizer_count": float(normalizer.detach().cpu()),
            "reference_actor_bc_replay_normalization_is_full_batch": float(
                full_batch_normalization
            ),
            "reference_actor_bc_anchor_unconditional": float(
                self.cfg.reference_auxiliary_mode == "q_filtered_replay_bc"
            ),
            "reference_actor_bc_q_filter_active": float(
                q_filter_active
                and self.cfg.reference_auxiliary_mode
                in {"q_filtered_bc", "q_filtered_replay_bc"}
            ),
            "reference_actor_bc_q_filter_is_online_target_unanimous": float(
                q_filter_mode == "online_target_unanimous"
            ),
            "reference_actor_bc_q_filter_critic_count": float(q_filter_critic_count),
            "reference_actor_q_advantage_mean": float(q_advantage.detach().mean().cpu()),
            "reference_actor_q_advantage_selected_mean": float(
                q_advantage[mask.bool()].detach().mean().cpu()
                if bool(mask.bool().any().item())
                else 0.0
            ),
            "reference_actor_q_advantage_positive_fraction": float((q_advantage.detach() > 0.0).float().mean().cpu()),
            "reference_actor_q_actor_mean": float(q_actor.detach().mean().cpu()),
            "reference_actor_q_reference_mean": float(q_reference.detach().mean().cpu()),
            "reference_actor_action_abs_error_mean": float((actor_actions - reference_actions).detach().abs().mean().cpu()),
            "reference_actor_action_abs_error_masked_mean": float(masked_error.detach().cpu()),
        }
        return loss, metrics

    def diagnostics(self, data: Any, dormant_relative_threshold: float) -> dict[str, float]:
        observations = self._normalize_obs_tensor(data.observations)
        with torch.no_grad():
            actor_acts = self._actor_hidden_activations(observations)
            q1_acts = self._q_hidden_activations(self.q1, observations, data.actions)
            q2_acts = self._q_hidden_activations(self.q2, observations, data.actions)
        out: dict[str, float] = {
            "actor_param_norm": parameter_norm(self.actor),
            "q1_param_norm": parameter_norm(self.q1),
            "q2_param_norm": parameter_norm(self.q2),
            "alpha": float(self._alpha_tensor.cpu()),
        }
        out.update(activation_statistics("actor", actor_acts, dormant_relative_threshold))
        out.update(activation_statistics("q1", q1_acts, dormant_relative_threshold))
        out.update(activation_statistics("q2", q2_acts, dormant_relative_threshold))
        return out

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": (
                "CleanRL SAC update; "
                f"simba_backbone={self.cfg.simba_backbone}; "
                f"simba_observation_norm={self.cfg.simba_observation_norm}; "
                f"simba_weight_projection={self.cfg.simba_weight_projection}; "
                f"simba_distributional_critic={self.cfg.simba_distributional_critic}; "
                f"simba_reward_scaling={self.cfg.simba_reward_scaling}; "
                f"redq_num_critics={len(self.q_networks)}; "
                f"redq_target_subset_size={self._redq_target_subset_size()}; "
                f"target_q_aggregation={self.cfg.target_q_aggregation}; "
                f"redo_interval_updates={self.cfg.redo_interval_updates}; "
                f"redo_dormant_threshold={self.cfg.redo_dormant_threshold}"
            ),
            "actor": self.actor.state_dict(),
            "qf1": self.q1.state_dict(),
            "qf2": self.q2.state_dict(),
            "qf1_target": self.q1_target.state_dict(),
            "qf2_target": self.q2_target.state_dict(),
            "q_ensemble": [q_network.state_dict() for q_network in self.q_networks],
            "q_target_ensemble": [q_network.state_dict() for q_network in self.q_target_networks],
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha": float(self._alpha_tensor.cpu()),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "obs_rms": self.obs_rms.state_dict() if self.obs_rms is not None else None,
            "reward_scaler": self.reward_scaler.state_dict() if self.reward_scaler is not None else None,
            "extra": extra or {},
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path, load_optimizers: bool = False) -> dict[str, Any]:
        try:
            payload = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older PyTorch.
            payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
        if "q_ensemble" in payload and len(payload["q_ensemble"]) == len(self.q_networks):
            for q_network, state in zip(self.q_networks, payload["q_ensemble"]):
                q_network.load_state_dict(state)
            for q_network, state in zip(self.q_target_networks, payload["q_target_ensemble"]):
                q_network.load_state_dict(state)
        else:
            self.q1.load_state_dict(payload["qf1"])
            self.q2.load_state_dict(payload["qf2"])
            self.q1_target.load_state_dict(payload["qf1_target"])
            self.q2_target.load_state_dict(payload["qf2_target"])
        self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
        self._alpha_tensor = self.log_alpha.exp().detach()
        self.alpha = float(self._alpha_tensor.cpu())
        if self.obs_rms is not None and payload.get("obs_rms") is not None:
            self.obs_rms.load_state_dict(payload["obs_rms"])
        if self.reward_scaler is not None and payload.get("reward_scaler") is not None:
            self.reward_scaler.load_state_dict(payload["reward_scaler"])

        if load_optimizers:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            self.q_optimizer.load_state_dict(payload["q_optimizer"])
            self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])

        self.actor.eval()
        for q_network in self.q_networks:
            q_network.eval()
        for q_network in self.q_target_networks:
            q_network.eval()
        return payload

    def load_actor_checkpoint(self, path: str | Path, load_obs_rms: bool = True) -> dict[str, Any]:
        try:
            payload = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older PyTorch.
            payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
        if load_obs_rms and self.obs_rms is not None and payload.get("obs_rms") is not None:
            self.obs_rms.load_state_dict(payload["obs_rms"])
        if self.cfg.simba_weight_projection:
            project_simba_weights_to_unit_norm(self.actor)
        self.actor.train()
        return payload

    def _distributional_critic_losses(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_actions: torch.Tensor,
        next_log_pi: torch.Tensor,
        update_step: int,
        replay_importance_weights: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if not all(isinstance(q_network, SimbaCategoricalQNetwork) for q_network in self.q_networks):
            raise TypeError("Distributional critic loss requires SimbaCategoricalQNetwork critics.")
        if not all(isinstance(q_network, SimbaCategoricalQNetwork) for q_network in self.q_target_networks):
            raise TypeError("Distributional critic loss requires categorical target critics.")

        with torch.no_grad():
            target_qs = []
            target_log_probs = []
            for index in self._redq_target_indices(update_step):
                q_network = self.q_target_networks[index]
                next_q, next_log_prob = q_network.distribution(next_observations, next_actions)
                target_qs.append(next_q.view(-1))
                target_log_probs.append(next_log_prob)
            next_q_stack = torch.stack(target_qs, dim=0)
            next_log_probs_stack = torch.stack(target_log_probs, dim=0)
            batch_indices = torch.arange(next_q_stack.shape[1], device=next_q_stack.device)
            if self.cfg.target_q_aggregation == "min":
                selected_indices = next_q_stack.argmin(dim=0)
                next_log_probs = next_log_probs_stack[selected_indices, batch_indices]
            elif self.cfg.target_q_aggregation == "mean":
                next_probs = next_log_probs_stack.exp().mean(dim=0).clamp_min(1e-12)
                next_log_probs = next_probs.log()
            elif self.cfg.target_q_aggregation == "max":
                selected_indices = next_q_stack.argmax(dim=0)
                next_log_probs = next_log_probs_stack[selected_indices, batch_indices]
            else:
                raise ValueError(f"Unknown target_q_aggregation: {self.cfg.target_q_aggregation}")
            target_probs = self._categorical_target_projection(
                target_log_probs=next_log_probs,
                rewards=rewards.flatten(),
                dones=dones.flatten(),
                next_log_pi=next_log_pi.flatten(),
                support=self.q1.bin_values,
            )

        losses = []
        for q_network in self.q_networks:
            _q, pred_log_probs = q_network.distribution(observations, actions)
            per_sample_loss = -(target_probs * pred_log_probs).sum(dim=1)
            if replay_importance_weights is None:
                losses.append(per_sample_loss.mean())
            else:
                losses.append((per_sample_loss * replay_importance_weights).mean())
        return losses

    def _categorical_target_projection(
        self,
        target_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_log_pi: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        discounts = self.cfg.gamma * (1.0 - dones)
        offsets = rewards - discounts * self._alpha_tensor * next_log_pi
        return self._categorical_project_distribution(
            target_log_probs=target_log_probs,
            offsets=offsets,
            discounts=discounts,
            support=support,
        )

    @staticmethod
    def _categorical_project_distribution(
        target_log_probs: torch.Tensor,
        offsets: torch.Tensor,
        discounts: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        num_bins = support.shape[0]
        v_min = float(support[0].item())
        v_max = float(support[-1].item())
        delta_z = (v_max - v_min) / float(num_bins - 1)
        target_z = offsets[:, None] + discounts[:, None] * support.view(1, -1)
        target_z = target_z.clamp(v_min, v_max)
        b = (target_z - v_min) / delta_z
        lower = b.floor().long().clamp(0, num_bins - 1)
        upper = b.ceil().long().clamp(0, num_bins - 1)
        probs = target_log_probs.exp()
        lower_weight = upper.float() - b
        upper_weight = b - lower.float()
        equal = lower == upper
        lower_weight = torch.where(equal, torch.ones_like(lower_weight), lower_weight)
        upper_weight = torch.where(equal, torch.zeros_like(upper_weight), upper_weight)

        projected = torch.zeros_like(probs)
        offset = torch.arange(probs.shape[0], device=probs.device).view(-1, 1) * num_bins
        projected.view(-1).scatter_add_(0, (lower + offset).reshape(-1), (probs * lower_weight).reshape(-1))
        projected.view(-1).scatter_add_(0, (upper + offset).reshape(-1), (probs * upper_weight).reshape(-1))
        return projected

    def _actor_hidden_activations(self, obs: torch.Tensor) -> list[torch.Tensor]:
        if hasattr(self.actor, "hidden_activations"):
            return self.actor.hidden_activations(obs)
        first = F.relu(self.actor.fc1(obs))
        second = F.relu(self.actor.fc2(first))
        return [first, second]

    @staticmethod
    def _q_hidden_activations(network: nn.Module, obs: torch.Tensor, action: torch.Tensor) -> list[torch.Tensor]:
        if hasattr(network, "hidden_activations"):
            return network.hidden_activations(obs, action)
        x = torch.cat([obs, action], 1)
        first = F.relu(network.fc1(x))
        second = F.relu(network.fc2(first))
        return [first, second]
