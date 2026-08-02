from __future__ import annotations

from typing import Any

import numpy as np
import torch


def reflection_averaged_actor_actions(
    actor_agent: Any, observations: np.ndarray
) -> np.ndarray:
    """Project a Pendulum actor onto the exact reflection-equivariant policy class."""
    observations = np.asarray(observations, dtype=np.float32)
    mirrored = observations.copy()
    mirrored[:, 1:] *= -1.0
    direct = np.asarray(
        actor_agent.act_batch(observations, deterministic=True), dtype=np.float32
    ).reshape(-1, 1)
    reflected = np.asarray(
        actor_agent.act_batch(mirrored, deterministic=True), dtype=np.float32
    ).reshape(-1, 1)
    return 0.5 * (direct - reflected)


class ReflectionAveragedActorPolicy:
    """Expose the fixed Pendulum reflection projection through ``act_batch``."""

    def __init__(self, actor_agent: Any) -> None:
        self.actor_agent = actor_agent

    def act_batch(self, observations: np.ndarray, deterministic: bool = True) -> np.ndarray:
        del deterministic
        return reflection_averaged_actor_actions(self.actor_agent, observations)

    def selection_metrics(self) -> dict[str, float]:
        return {
            "selected_count": 0.0,
            "total_decisions": 0.0,
            "switch_fraction": 0.0,
            "selected_abs_action_delta_mean": 0.0,
            "selected_abs_action_delta_max": 0.0,
            "selected_unanimous_advantage_mean": 0.0,
        }


class FixedLocalCriticQSearchPolicy:
    """Apply one fixed local action search scored by a pure-RL critic ensemble."""

    def __init__(
        self,
        actor_agent: Any,
        critic_agent: Any,
        num_actions: int,
        margin: float,
        search_radius: float,
        symmetric_actor_fallback: bool = False,
    ) -> None:
        self.actor_agent = actor_agent
        self.critic_agent = critic_agent
        self.num_actions = int(num_actions)
        self.margin = float(margin)
        self.search_radius = float(search_radius)
        self.symmetric_actor_fallback = bool(symmetric_actor_fallback)
        self.selected_count = 0
        self.total_count = 0
        self.selected_abs_delta_sum = 0.0
        self.selected_abs_delta_max = 0.0
        self.selected_advantage_sum = 0.0

    def act_batch(self, observations: np.ndarray, deterministic: bool = True) -> np.ndarray:
        del deterministic
        actor_np = (
            reflection_averaged_actor_actions(self.actor_agent, observations)
            if self.symmetric_actor_fallback
            else np.asarray(
                self.actor_agent.act_batch(observations, deterministic=True),
                dtype=np.float32,
            ).reshape(-1, 1)
        )
        device = self.critic_agent.device
        with torch.no_grad():
            raw_obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
            normalized = self.critic_agent._normalize_obs_tensor(raw_obs)
            actor_actions = torch.as_tensor(actor_np, dtype=torch.float32, device=device)
            action_low = (
                self.critic_agent.actor.action_bias - self.critic_agent.actor.action_scale
            ).view(1, 1, 1)
            action_high = (
                self.critic_agent.actor.action_bias + self.critic_agent.actor.action_scale
            ).view(1, 1, 1)
            offsets = torch.linspace(
                -self.search_radius,
                self.search_radius,
                self.num_actions,
                dtype=actor_actions.dtype,
                device=device,
            ).view(1, self.num_actions, 1)
            candidates = torch.clamp(
                actor_actions.unsqueeze(1) + offsets,
                min=action_low,
                max=action_high,
            )
            batch = int(len(actor_actions))
            expanded_obs = normalized.unsqueeze(1).expand(-1, self.num_actions, -1)
            flat_obs = expanded_obs.reshape(batch * self.num_actions, -1)
            flat_actions = candidates.reshape(batch * self.num_actions, 1)
            candidate_per_critic = torch.stack(
                [
                    network(flat_obs, flat_actions).reshape(batch, self.num_actions)
                    for network in self.critic_agent.q_networks
                ],
                dim=0,
            )
            clipped_candidate_q = candidate_per_critic.min(dim=0).values
            best_indices = clipped_candidate_q.argmax(dim=1)
            row = torch.arange(batch, device=device)
            best_actions = candidates[row, best_indices]
            per_critic_best = torch.stack(
                [
                    network(normalized, best_actions).view(-1)
                    for network in self.critic_agent.q_networks
                ],
                dim=0,
            )
            per_critic_actor = torch.stack(
                [
                    network(normalized, actor_actions).view(-1)
                    for network in self.critic_agent.q_networks
                ],
                dim=0,
            )
            unanimous_advantage = (per_critic_best - per_critic_actor).min(dim=0).values
            use_best = unanimous_advantage > self.margin
            selected = int(use_best.sum().item())
            self.selected_count += selected
            self.total_count += int(use_best.numel())
            if selected:
                delta = (best_actions[use_best] - actor_actions[use_best]).abs()
                self.selected_abs_delta_sum += float(delta.sum().item())
                self.selected_abs_delta_max = max(
                    self.selected_abs_delta_max,
                    float(delta.max().item()),
                )
                self.selected_advantage_sum += float(
                    unanimous_advantage[use_best].sum().item()
                )
            return torch.where(
                use_best.view(-1, 1), best_actions, actor_actions
            ).cpu().numpy()

    def selection_metrics(self) -> dict[str, float]:
        return {
            "selected_count": float(self.selected_count),
            "total_decisions": float(self.total_count),
            "switch_fraction": float(self.selected_count / max(self.total_count, 1)),
            "selected_abs_action_delta_mean": float(
                self.selected_abs_delta_sum / max(self.selected_count, 1)
            ),
            "selected_abs_action_delta_max": float(self.selected_abs_delta_max),
            "selected_unanimous_advantage_mean": float(
                self.selected_advantage_sum / max(self.selected_count, 1)
            ),
        }


class FixedGlobalCriticQSearchPolicy:
    """A coordinate-agnostic global grid search with conservative acceptance.

    The candidate grid is fixed over the checkpoint's full action support.  A
    proposal is used only when every online critic improves over the actor
    fallback by more than ``margin`` and, when supplied, it lies inside the
    fixed action-delta trust region.  No state coordinates or reference values
    enter the policy.
    """

    def __init__(
        self,
        actor_agent: Any,
        critic_agent: Any,
        num_actions: int,
        margin: float,
        max_action_delta: float,
        symmetric_actor_fallback: bool = False,
    ) -> None:
        if int(num_actions) < 2:
            raise ValueError("num_actions must be at least 2")
        if float(margin) < 0.0:
            raise ValueError("margin must be nonnegative")
        if float(max_action_delta) <= 0.0:
            raise ValueError("max_action_delta must be positive")
        self.actor_agent = actor_agent
        self.critic_agent = critic_agent
        self.num_actions = int(num_actions)
        self.margin = float(margin)
        self.max_action_delta = float(max_action_delta)
        self.symmetric_actor_fallback = bool(symmetric_actor_fallback)
        self.selected_count = 0
        self.total_count = 0
        self.trust_region_rejected_count = 0
        self.selected_abs_delta_sum = 0.0
        self.selected_abs_delta_max = 0.0
        self.selected_advantage_sum = 0.0

    def act_batch(self, observations: np.ndarray, deterministic: bool = True) -> np.ndarray:
        del deterministic
        actor_np = (
            reflection_averaged_actor_actions(self.actor_agent, observations)
            if self.symmetric_actor_fallback
            else np.asarray(
                self.actor_agent.act_batch(observations, deterministic=True),
                dtype=np.float32,
            ).reshape(-1, 1)
        )
        device = self.critic_agent.device
        with torch.no_grad():
            raw_obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
            normalized = self.critic_agent._normalize_obs_tensor(raw_obs)
            actor_actions = torch.as_tensor(actor_np, dtype=torch.float32, device=device)
            action_low = (
                self.critic_agent.actor.action_bias
                - self.critic_agent.actor.action_scale
            ).reshape(1)
            action_high = (
                self.critic_agent.actor.action_bias
                + self.critic_agent.actor.action_scale
            ).reshape(1)
            action_grid = torch.linspace(
                float(action_low.item()),
                float(action_high.item()),
                self.num_actions,
                dtype=actor_actions.dtype,
                device=device,
            ).view(1, self.num_actions, 1)
            batch = int(len(actor_actions))
            candidates = action_grid.expand(batch, -1, -1)
            expanded_obs = normalized.unsqueeze(1).expand(-1, self.num_actions, -1)
            flat_obs = expanded_obs.reshape(batch * self.num_actions, -1)
            flat_actions = candidates.reshape(batch * self.num_actions, 1)
            candidate_per_critic = torch.stack(
                [
                    network(flat_obs, flat_actions).reshape(batch, self.num_actions)
                    for network in self.critic_agent.q_networks
                ],
                dim=0,
            )
            clipped_candidate_q = candidate_per_critic.min(dim=0).values
            best_indices = clipped_candidate_q.argmax(dim=1)
            row = torch.arange(batch, device=device)
            best_actions = candidates[row, best_indices]
            per_critic_best = torch.stack(
                [
                    network(normalized, best_actions).view(-1)
                    for network in self.critic_agent.q_networks
                ],
                dim=0,
            )
            per_critic_actor = torch.stack(
                [
                    network(normalized, actor_actions).view(-1)
                    for network in self.critic_agent.q_networks
                ],
                dim=0,
            )
            unanimous_advantage = (per_critic_best - per_critic_actor).min(dim=0).values
            passes_advantage = unanimous_advantage > self.margin
            action_delta = (best_actions - actor_actions).abs().view(-1)
            passes_trust_region = action_delta <= self.max_action_delta
            self.trust_region_rejected_count += int(
                (passes_advantage & ~passes_trust_region).sum().item()
            )
            use_best = passes_advantage & passes_trust_region
            selected = int(use_best.sum().item())
            self.selected_count += selected
            self.total_count += int(use_best.numel())
            if selected:
                selected_delta = action_delta[use_best]
                self.selected_abs_delta_sum += float(selected_delta.sum().item())
                self.selected_abs_delta_max = max(
                    self.selected_abs_delta_max,
                    float(selected_delta.max().item()),
                )
                self.selected_advantage_sum += float(
                    unanimous_advantage[use_best].sum().item()
                )
            return torch.where(
                use_best.view(-1, 1), best_actions, actor_actions
            ).cpu().numpy()

    def selection_metrics(self) -> dict[str, float]:
        return {
            "selected_count": float(self.selected_count),
            "total_decisions": float(self.total_count),
            "switch_fraction": float(self.selected_count / max(self.total_count, 1)),
            "trust_region_rejected_count": float(self.trust_region_rejected_count),
            "trust_region_rejected_fraction": float(
                self.trust_region_rejected_count / max(self.total_count, 1)
            ),
            "selected_abs_action_delta_mean": float(
                self.selected_abs_delta_sum / max(self.selected_count, 1)
            ),
            "selected_abs_action_delta_max": float(self.selected_abs_delta_max),
            "selected_unanimous_advantage_mean": float(
                self.selected_advantage_sum / max(self.selected_count, 1)
            ),
        }
