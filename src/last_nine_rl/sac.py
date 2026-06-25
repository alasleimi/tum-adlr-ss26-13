from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
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
        if cfg.simba_weight_projection and not cfg.simba_backbone:
            raise ValueError(
                "simba_weight_projection requires simba_backbone because SimbaV2 projection is defined on "
                "bias-free HyperDense weights, not CleanRL Linear layers."
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

        action_dim = int(np.prod(self.env_spec.single_action_space.shape))
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
            self.actor = Actor(self.env_spec).to(self.device)
            self.q1 = SoftQNetwork(self.env_spec).to(self.device)
            self.q2 = SoftQNetwork(self.env_spec).to(self.device)
            self.q1_target = SoftQNetwork(self.env_spec).to(self.device)
            self.q2_target = SoftQNetwork(self.env_spec).to(self.device)
        if cfg.simba_weight_projection:
            self._project_weights()
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=cfg.policy_lr)
        self._q_lr_initial = float(cfg.q_lr)
        self._policy_lr_initial = float(cfg.policy_lr)

        action_dim_tensor = torch.prod(torch.Tensor(self.env_spec.single_action_space.shape).to(self.device)).item()
        self.target_entropy = float(cfg.target_entropy_scale) * float(action_dim_tensor)
        self.log_alpha = torch.full(
            (1,),
            math.log(float(cfg.alpha_initial_value)),
            requires_grad=True,
            device=self.device,
        )
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.q_lr)
        self.last_actor_grad_norm = 0.0
        self.last_q_grad_norm = 0.0
        self.feature_norms: dict[str, float] = {}
        self._feature_hook_handles: list[Any] = []
        if cfg.update_diagnostics:
            self._register_feature_hooks()

    def observe(self, observation: np.ndarray) -> None:
        if self.obs_rms is not None:
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

        for prefix, module in (("actor", self.actor), ("q1", self.q1), ("q2", self.q2)):
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
        for module in (self.actor, self.q1, self.q2):
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
        self._set_optimizer_lr(self.q_optimizer, q_lr)
        self._set_optimizer_lr(self.alpha_optimizer, q_lr)
        self._set_optimizer_lr(self.actor_optimizer, policy_lr)
        return q_lr, policy_lr

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

    def update(
        self,
        data: Any,
        update_step: int,
        reference_actions: Any | None = None,
        reference_critic_actions: Any | None = None,
    ) -> dict[str, float]:
        if update_step <= 0:
            raise ValueError("update_step is one-indexed and must be positive.")
        if self.cfg.update_diagnostics:
            self.feature_norms.clear()
        q_lr, policy_lr = self._apply_lr_schedule(update_step)

        observations = self._normalize_obs_tensor(data.observations)
        reference_actions_tensor = self._prepare_reference_actions(reference_actions, data.actions, observations)
        reference_critic_actions_tensor = self._prepare_reference_critic_actions(
            reference_critic_actions,
            data.actions,
            observations,
        )
        qf1_a_values = self.q1(observations, data.actions).view(-1)
        qf2_a_values = self.q2(observations, data.actions).view(-1)

        sacn_metrics: dict[str, float] = {}
        if self._uses_sacn_batch(data):
            if self.cfg.simba_distributional_critic:
                qf1_loss, qf2_loss, next_q_value, sacn_metrics = self._distributional_sacn_critic_losses(
                    data=data,
                    observations=observations,
                    actions=data.actions,
                )
            else:
                qf1_loss, qf2_loss, next_q_value, sacn_metrics = self._scalar_sacn_critic_losses(
                    data=data,
                    observations=observations,
                    actions=data.actions,
                )
        else:
            next_observations = self._normalize_obs_tensor(data.next_observations)
            rewards = data.rewards
            if self.reward_scaler is not None:
                rewards = self.reward_scaler.scale_tensor(rewards)

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = self.actor.get_action(next_observations)
                qf1_next_target = self.q1_target(next_observations, next_state_actions)
                qf2_next_target = self.q2_target(next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * self.cfg.gamma * min_qf_next_target.view(-1)

            if self.cfg.simba_distributional_critic:
                qf1_loss, qf2_loss = self._distributional_critic_losses(
                    observations=observations,
                    actions=data.actions,
                    next_observations=next_observations,
                    rewards=rewards,
                    dones=data.dones,
                    next_actions=next_state_actions,
                    next_log_pi=next_state_log_pi,
                )
            else:
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)

        reference_critic_metrics: dict[str, float] = {}
        if reference_critic_actions_tensor is not None:
            ref_qf1_loss, ref_qf2_loss, reference_critic_metrics = self._reference_critic_margin_losses(
                observations=observations,
                reference_actions=reference_critic_actions_tensor,
            )
            qf1_loss = qf1_loss + float(self.cfg.reference_critic_weight) * ref_qf1_loss
            qf2_loss = qf2_loss + float(self.cfg.reference_critic_weight) * ref_qf2_loss
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        q_params = list(self.q1.parameters()) + list(self.q2.parameters())
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
            project_simba_weights_to_unit_norm(self.q1)
            project_simba_weights_to_unit_norm(self.q2)
        redo_metrics = self._maybe_redo_critics(observations, data.actions, update_step)
        q_update_norm = parameter_delta_norm(q_params_before, q_params) if self.cfg.update_diagnostics else 0.0

        metrics: dict[str, float] = {
            "q1_loss": float(qf1_loss.detach().cpu()),
            "q2_loss": float(qf2_loss.detach().cpu()),
            "q_loss": float(qf_loss.detach().cpu()),
            "q_loss_cleanrl_logged": float((qf_loss.detach() / 2.0).cpu()),
            "q1_mean": float(qf1_a_values.detach().mean().cpu()),
            "q2_mean": float(qf2_a_values.detach().mean().cpu()),
            "target_q_mean": float(next_q_value.detach().mean().cpu()),
            "alpha": float(self.alpha),
            "q_lr": float(q_lr),
            "policy_lr": float(policy_lr),
        }
        metrics.update(sacn_metrics)
        metrics.update(reference_critic_metrics)
        metrics.update(redo_metrics)
        if self.cfg.update_diagnostics:
            metrics["q_grad_norm"] = self.last_q_grad_norm
            self._record_gradient_norms(metrics, "q1", self.q1)
            self._record_gradient_norms(metrics, "q2", self.q2)

        if update_step % self.cfg.policy_frequency == 0:
            for _ in range(self.cfg.policy_frequency):
                pi, log_pi, policy_mean = self.actor.get_action(observations)
                qf1_pi = self.q1(observations, pi)
                qf2_pi = self.q2(observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                sac_actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
                actor_loss = sac_actor_loss
                reference_actor_metrics: dict[str, float] = {}
                if reference_actions_tensor is not None:
                    reference_actor_loss, reference_actor_metrics = self._reference_auxiliary_actor_loss(
                        observations=observations,
                        actor_actions=policy_mean,
                        reference_actions=reference_actions_tensor,
                    )
                    actor_loss = actor_loss + float(self.cfg.reference_auxiliary_weight) * reference_actor_loss

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_params = list(self.actor.parameters())
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

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()

            metrics.update(
                {
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "sac_actor_loss": float(sac_actor_loss.detach().cpu()),
                    "alpha_loss": float(alpha_loss.detach().cpu()),
                    "policy_log_prob_mean": float(log_pi.detach().mean().cpu()),
                    "policy_entropy_estimate": float((-log_pi.detach()).mean().cpu()),
                    "alpha": float(self.alpha),
                }
            )
            metrics.update(reference_actor_metrics)
            if self.cfg.update_diagnostics:
                metrics.update(
                    {
                        "actor_grad_norm": self.last_actor_grad_norm,
                        "actor_update_norm": actor_update_norm,
                        "actor_update_norm_ratio": actor_update_norm / max(actor_param_norm_before, 1e-12),
                    }
                )

        if update_step % self.cfg.target_network_frequency == 0:
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)
            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)

        if self.cfg.update_diagnostics:
            actor_norm = parameter_norm(self.actor)
            q_norm = parameter_norm(self.q1) + parameter_norm(self.q2)
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

    def _scalar_sacn_critic_losses(
        self,
        data: Any,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        with torch.no_grad():
            common = self._sacn_common_target_inputs(data)
            target_q1 = self.q1_target(common["successor_observations"], common["successor_actions"])
            target_q2 = self.q2_target(common["successor_observations"], common["successor_actions"])
            bootstrap_values = torch.min(target_q1, target_q2).view(common["batch_size"], common["n_step"])
            targets = common["offsets"] + common["discounts"] * bootstrap_values

        q1_pred = self.q1(observations, actions).view(-1, 1)
        q2_pred = self.q2(observations, actions).view(-1, 1)
        weights = common["weights"]
        qf1_loss = self._sacn_weighted_horizon_loss((q1_pred - targets).pow(2), weights, common["horizon_mask"])
        qf2_loss = self._sacn_weighted_horizon_loss((q2_pred - targets).pow(2), weights, common["horizon_mask"])
        metrics = self._sacn_metrics(
            targets=targets,
            weights=weights,
            log_omega=common["log_omega"],
            log_importance_clip=common["log_importance_clip"],
            entropy_sample_counts=common["entropy_sample_counts"],
            horizon_ess_fraction=common["horizon_ess_fraction"],
            horizon_mask=common["horizon_mask"],
        )
        return qf1_loss, qf2_loss, targets, metrics

    def _distributional_sacn_critic_losses(
        self,
        data: Any,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        if not isinstance(self.q1, SimbaCategoricalQNetwork) or not isinstance(self.q2, SimbaCategoricalQNetwork):
            raise TypeError("Distributional SACn critic loss requires SimbaCategoricalQNetwork critics.")
        if not isinstance(self.q1_target, SimbaCategoricalQNetwork) or not isinstance(
            self.q2_target, SimbaCategoricalQNetwork
        ):
            raise TypeError("Distributional SACn critic loss requires categorical target critics.")

        with torch.no_grad():
            common = self._sacn_common_target_inputs(data)
            next_q1, next_log_probs1 = self.q1_target.distribution(
                common["successor_observations"], common["successor_actions"]
            )
            next_q2, next_log_probs2 = self.q2_target.distribution(
                common["successor_observations"], common["successor_actions"]
            )
            use_q1 = next_q1 <= next_q2
            next_log_probs = torch.where(use_q1, next_log_probs1, next_log_probs2).view(
                common["batch_size"],
                common["n_step"],
                -1,
            )
            bootstrap_values = torch.min(next_q1, next_q2).view(common["batch_size"], common["n_step"])
            target_values = common["offsets"] + common["discounts"] * bootstrap_values

        _q1, pred_log_probs1 = self.q1.distribution(observations, actions)
        _q2, pred_log_probs2 = self.q2.distribution(observations, actions)
        target_probs = self._categorical_project_distribution(
            target_log_probs=next_log_probs.reshape(common["batch_size"] * common["n_step"], -1),
            offsets=common["offsets"].reshape(-1),
            discounts=common["discounts"].reshape(-1),
            support=self.q1.bin_values,
        ).reshape(common["batch_size"], common["n_step"], -1)
        ce1 = -(target_probs * pred_log_probs1[:, None, :]).sum(dim=2)
        ce2 = -(target_probs * pred_log_probs2[:, None, :]).sum(dim=2)
        qf1_loss = self._sacn_weighted_horizon_loss(ce1, common["weights"], common["horizon_mask"])
        qf2_loss = self._sacn_weighted_horizon_loss(ce2, common["weights"], common["horizon_mask"])

        metrics = self._sacn_metrics(
            targets=target_values,
            weights=common["weights"],
            log_omega=common["log_omega"],
            log_importance_clip=common["log_importance_clip"],
            entropy_sample_counts=common["entropy_sample_counts"],
            horizon_ess_fraction=common["horizon_ess_fraction"],
            horizon_mask=common["horizon_mask"],
        )
        return qf1_loss, qf2_loss, target_values, metrics

    def _sacn_common_target_inputs(self, data: Any) -> dict[str, Any]:
        rewards = data.trajectory_rewards
        if self.reward_scaler is not None:
            rewards = self.reward_scaler.scale_tensor(rewards)
        rewards = rewards.squeeze(-1)
        dones = data.trajectory_dones.squeeze(-1).to(dtype=rewards.dtype)
        batch_size, n_step = rewards.shape

        weights, log_omega, log_importance_clip = self._sacn_importance_weights(data)
        weights, horizon_mask, horizon_ess_fraction = self._sacn_apply_horizon_support(weights)
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
            "horizon_ess_fraction": horizon_ess_fraction,
            "log_omega": log_omega,
            "log_importance_clip": log_importance_clip,
            "offsets": offsets,
            "discounts": discounts,
            "successor_observations": successor_observations.reshape(batch_size * n_step, -1),
            "successor_actions": successor_actions.reshape(batch_size * n_step, -1),
            "entropy_sample_counts": entropy_sample_counts,
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
        horizon_mask: torch.Tensor,
    ) -> torch.Tensor:
        active_horizons = horizon_mask.to(dtype=per_horizon_loss.dtype).sum().clamp_min(1.0)
        normalizer = float(per_horizon_loss.shape[0]) * active_horizons
        return (per_horizon_loss * weights).sum() / normalizer

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
                    float(self.alpha)
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
    ) -> dict[str, float]:
        detached_targets = targets.detach()
        detached_weights = weights.detach()
        detached_log_omega = log_omega.detach()
        detached_ess_fraction = horizon_ess_fraction.detach()
        detached_horizon_mask = horizon_mask.detach()
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
            "sacn_horizon_last_active": float(detached_horizon_mask[-1].cpu()),
            "sacn_log_omega_mean": float(detached_log_omega.mean().cpu()),
            "sacn_log_omega_std": float(detached_log_omega.std(unbiased=False).cpu()),
            "sacn_log_importance_clip": float(log_importance_clip.detach().cpu()),
            "sacn_entropy_samples_max": float(max(entropy_sample_counts)),
            "sacn_importance_is_density": 1.0 if self.cfg.sacn_importance_mode == "density" else 0.0,
            "sacn_non_soft_targets": 1.0 if self.cfg.sacn_non_soft_targets else 0.0,
        }

    def _prepare_reference_actions(
        self,
        reference_actions: Any | None,
        sampled_actions: torch.Tensor,
        observations: torch.Tensor,
    ) -> torch.Tensor | None:
        if (
            reference_actions is None
            or self.cfg.reference_auxiliary_mode == "none"
            or self.cfg.reference_auxiliary_weight <= 0.0
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

    def _reference_auxiliary_actor_loss(
        self,
        observations: torch.Tensor,
        actor_actions: torch.Tensor,
        reference_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        action_scale = torch.clamp(self.actor.action_scale.abs(), min=1e-6).view(1, -1)
        action_error = (actor_actions - reference_actions) / action_scale
        per_sample_loss = action_error.pow(2).mean(dim=1, keepdim=True)

        with torch.no_grad():
            q_actor = torch.min(self.q1(observations, actor_actions.detach()), self.q2(observations, actor_actions.detach()))
            q_reference = torch.min(self.q1(observations, reference_actions), self.q2(observations, reference_actions))
            q_advantage = q_reference - q_actor
            if self.cfg.reference_auxiliary_mode == "q_filtered_bc":
                mask = (q_advantage > float(self.cfg.reference_auxiliary_margin)).to(per_sample_loss.dtype)
            elif self.cfg.reference_auxiliary_mode == "bc":
                mask = torch.ones_like(per_sample_loss)
            else:
                raise ValueError(f"Unknown reference auxiliary mode: {self.cfg.reference_auxiliary_mode}")

        selected = mask.sum()
        if float(selected.detach().cpu()) > 0.0:
            loss = (per_sample_loss * mask).sum() / selected.clamp_min(1.0)
            masked_error = ((actor_actions - reference_actions).abs() * mask).sum() / (
                selected.clamp_min(1.0) * actor_actions.shape[1]
            )
        else:
            loss = per_sample_loss.sum() * 0.0
            masked_error = torch.zeros((), dtype=per_sample_loss.dtype, device=per_sample_loss.device)

        metrics = {
            "reference_actor_bc_loss": float(loss.detach().cpu()),
            "reference_actor_bc_mask_fraction": float(mask.detach().mean().cpu()),
            "reference_actor_q_advantage_mean": float(q_advantage.detach().mean().cpu()),
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
            "alpha": float(self.alpha),
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
                f"redo_interval_updates={self.cfg.redo_interval_updates}; "
                f"redo_dormant_threshold={self.cfg.redo_dormant_threshold}"
            ),
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
        self.q1.load_state_dict(payload["qf1"])
        self.q2.load_state_dict(payload["qf2"])
        self.q1_target.load_state_dict(payload["qf1_target"])
        self.q2_target.load_state_dict(payload["qf2_target"])
        self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
        self.alpha = float(self.log_alpha.exp().item())
        if self.obs_rms is not None and payload.get("obs_rms") is not None:
            self.obs_rms.load_state_dict(payload["obs_rms"])
        if self.reward_scaler is not None and payload.get("reward_scaler") is not None:
            self.reward_scaler.load_state_dict(payload["reward_scaler"])

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

    def _distributional_critic_losses(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_actions: torch.Tensor,
        next_log_pi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(self.q1, SimbaCategoricalQNetwork) or not isinstance(self.q2, SimbaCategoricalQNetwork):
            raise TypeError("Distributional critic loss requires SimbaCategoricalQNetwork critics.")
        if not isinstance(self.q1_target, SimbaCategoricalQNetwork) or not isinstance(
            self.q2_target, SimbaCategoricalQNetwork
        ):
            raise TypeError("Distributional critic loss requires categorical target critics.")

        with torch.no_grad():
            next_q1, next_log_probs1 = self.q1_target.distribution(next_observations, next_actions)
            next_q2, next_log_probs2 = self.q2_target.distribution(next_observations, next_actions)
            use_q1 = next_q1 <= next_q2
            next_log_probs = torch.where(use_q1, next_log_probs1, next_log_probs2)
            target_probs = self._categorical_target_projection(
                target_log_probs=next_log_probs,
                rewards=rewards.flatten(),
                dones=dones.flatten(),
                next_log_pi=next_log_pi.flatten(),
                support=self.q1.bin_values,
            )

        _q1, pred_log_probs1 = self.q1.distribution(observations, actions)
        _q2, pred_log_probs2 = self.q2.distribution(observations, actions)
        loss1 = -(target_probs * pred_log_probs1).sum(dim=1).mean()
        loss2 = -(target_probs * pred_log_probs2).sum(dim=1).mean()
        return loss1, loss2

    def _categorical_target_projection(
        self,
        target_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_log_pi: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        discounts = self.cfg.gamma * (1.0 - dones)
        offsets = rewards - discounts * self.alpha * next_log_pi
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
