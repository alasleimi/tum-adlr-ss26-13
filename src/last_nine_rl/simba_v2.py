from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from last_nine_rl.config import SACConfig


EPS = 1e-8
SIMBA_LOG_STD_MAX = 2.0
SIMBA_LOG_STD_MIN = -10.0


def l2normalize(x: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim, eps=eps)


class SimbaScaler(nn.Module):
    """Trainable scale vector with decoupled initial forward scale."""

    def __init__(self, dim: int, init: float = 1.0, scale: float = 1.0):
        super().__init__()
        self.scaler = nn.Parameter(torch.full((dim,), float(scale)))
        self.forward_scaler = float(init) / float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaler * self.forward_scaler * x


class SimbaHyperDense(nn.Module):
    """Bias-free linear layer used by SimbaV2's projected hypernetwork blocks."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        nn.init.orthogonal_(self.linear.weight, gain=1.0)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SimbaHyperEmbedder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        c_shift: float,
        use_shift: bool,
        use_feature_norm: bool,
    ):
        super().__init__()
        projected_input_dim = input_dim + 1 if use_shift else input_dim
        self.use_shift = use_shift
        self.use_feature_norm = use_feature_norm
        self.c_shift = float(c_shift)
        self.w = SimbaHyperDense(projected_input_dim, hidden_dim)
        self.scaler = SimbaScaler(hidden_dim, scaler_init, scaler_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_shift:
            new_axis = torch.full((*x.shape[:-1], 1), self.c_shift, device=x.device, dtype=x.dtype)
            x = torch.cat([x, new_axis], dim=-1)
        if self.use_feature_norm:
            x = l2normalize(x, dim=-1)
        x = self.w(x)
        x = self.scaler(x)
        if self.use_feature_norm:
            x = l2normalize(x, dim=-1)
        return x


class SimbaHyperMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        scaler_init: float,
        scaler_scale: float,
        use_feature_norm: bool,
        eps: float = EPS,
    ):
        super().__init__()
        self.w1 = SimbaHyperDense(out_dim, hidden_dim)
        self.scaler = SimbaScaler(hidden_dim, scaler_init, scaler_scale)
        self.w2 = SimbaHyperDense(hidden_dim, out_dim)
        self.use_feature_norm = use_feature_norm
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.scaler(x)
        x = F.relu(x) + self.eps
        x = self.w2(x)
        if self.use_feature_norm:
            x = l2normalize(x, dim=-1)
        return x


class SimbaHyperLERPBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        expansion: int,
        use_feature_norm: bool,
    ):
        super().__init__()
        self.mlp = SimbaHyperMLP(
            hidden_dim=hidden_dim * expansion,
            out_dim=hidden_dim,
            scaler_init=scaler_init / math.sqrt(expansion),
            scaler_scale=scaler_scale / math.sqrt(expansion),
            use_feature_norm=use_feature_norm,
        )
        self.alpha_scaler = SimbaScaler(hidden_dim, init=alpha_init, scale=alpha_scale)
        self.use_feature_norm = use_feature_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.mlp(x)
        x = residual + self.alpha_scaler(x - residual)
        if self.use_feature_norm:
            x = l2normalize(x, dim=-1)
        return x


class SimbaBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        c_shift: float,
        block_expansion: int,
        use_input_shift: bool,
        use_feature_norm: bool,
    ):
        super().__init__()
        scaler_init = math.sqrt(2.0 / hidden_dim)
        scaler_scale = math.sqrt(2.0 / hidden_dim)
        alpha_init = 1.0 / (num_blocks + 1.0)
        alpha_scale = 1.0 / math.sqrt(hidden_dim)
        self.embedder = SimbaHyperEmbedder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
            use_shift=use_input_shift,
            use_feature_norm=use_feature_norm,
        )
        self.blocks = nn.ModuleList(
            [
                SimbaHyperLERPBlock(
                    hidden_dim=hidden_dim,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                    expansion=block_expansion,
                    use_feature_norm=use_feature_norm,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x)
        for block in self.blocks:
            x = block(x)
        return x

    def forward_with_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        activations = []
        x = self.embedder(x)
        activations.append(x)
        for block in self.blocks:
            x = block(x)
            activations.append(x)
        return activations


class SimbaNormalTanhActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        cfg: SACConfig,
    ):
        super().__init__()
        action_dim = int(np.prod(action_low.shape))
        hidden_dim = cfg.simba_actor_hidden_dim
        self.backbone = SimbaBackbone(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_blocks=cfg.simba_actor_blocks,
            c_shift=cfg.simba_c_shift,
            block_expansion=cfg.simba_block_expansion,
            use_input_shift=cfg.simba_input_shift,
            use_feature_norm=cfg.simba_feature_norm,
        )
        self.mean_w1 = SimbaHyperDense(hidden_dim, hidden_dim)
        self.mean_scaler = SimbaScaler(hidden_dim, init=1.0, scale=1.0)
        self.mean_w2 = SimbaHyperDense(hidden_dim, action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim))
        self.std_w1 = SimbaHyperDense(hidden_dim, hidden_dim)
        self.std_scaler = SimbaScaler(hidden_dim, init=1.0, scale=1.0)
        self.std_w2 = SimbaHyperDense(hidden_dim, action_dim)
        self.std_bias = nn.Parameter(torch.zeros(action_dim))
        self.register_buffer("action_scale", torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(x)
        mean = self.mean_w2(self.mean_scaler(self.mean_w1(z))) + self.mean_bias
        log_std = self.std_w2(self.std_scaler(self.std_w1(z))) + self.std_bias
        log_std = torch.tanh(log_std)
        log_std = SIMBA_LOG_STD_MIN + 0.5 * (SIMBA_LOG_STD_MAX - SIMBA_LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def hidden_activations(self, obs: torch.Tensor) -> list[torch.Tensor]:
        return self.backbone.forward_with_activations(obs)


class SimbaScalarQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg: SACConfig):
        super().__init__()
        hidden_dim = cfg.simba_critic_hidden_dim
        self.backbone = SimbaBackbone(
            input_dim=obs_dim + action_dim,
            hidden_dim=hidden_dim,
            num_blocks=cfg.simba_critic_blocks,
            c_shift=cfg.simba_c_shift,
            block_expansion=cfg.simba_block_expansion,
            use_input_shift=cfg.simba_input_shift,
            use_feature_norm=cfg.simba_feature_norm,
        )
        self.value_w1 = SimbaHyperDense(hidden_dim, hidden_dim)
        self.value_scaler = SimbaScaler(hidden_dim, init=1.0, scale=1.0)
        self.value_w2 = SimbaHyperDense(hidden_dim, 1)
        self.value_bias = nn.Parameter(torch.zeros(1))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=1)
        z = self.backbone(x)
        return self.value_w2(self.value_scaler(self.value_w1(z))) + self.value_bias

    def hidden_activations(self, obs: torch.Tensor, action: torch.Tensor) -> list[torch.Tensor]:
        x = torch.cat([obs, action], dim=1)
        return self.backbone.forward_with_activations(x)


class SimbaCategoricalQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg: SACConfig):
        super().__init__()
        hidden_dim = cfg.simba_critic_hidden_dim
        self.backbone = SimbaBackbone(
            input_dim=obs_dim + action_dim,
            hidden_dim=hidden_dim,
            num_blocks=cfg.simba_critic_blocks,
            c_shift=cfg.simba_c_shift,
            block_expansion=cfg.simba_block_expansion,
            use_input_shift=cfg.simba_input_shift,
            use_feature_norm=cfg.simba_feature_norm,
        )
        self.value_w1 = SimbaHyperDense(hidden_dim, hidden_dim)
        self.value_scaler = SimbaScaler(hidden_dim, init=1.0, scale=1.0)
        self.value_w2 = SimbaHyperDense(hidden_dim, cfg.simba_critic_num_bins)
        self.value_bias = nn.Parameter(torch.zeros(cfg.simba_critic_num_bins))
        self.register_buffer(
            "bin_values",
            torch.linspace(cfg.simba_critic_min_v, cfg.simba_critic_max_v, cfg.simba_critic_num_bins),
        )

    def logits(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=1)
        z = self.backbone(x)
        return self.value_w2(self.value_scaler(self.value_w1(z))) + self.value_bias

    def distribution(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_probs = F.log_softmax(self.logits(obs, action), dim=1)
        values = torch.sum(log_probs.exp() * self.bin_values.view(1, -1), dim=1, keepdim=True)
        return values, log_probs

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        values, _log_probs = self.distribution(obs, action)
        return values

    def hidden_activations(self, obs: torch.Tensor, action: torch.Tensor) -> list[torch.Tensor]:
        x = torch.cat([obs, action], dim=1)
        return self.backbone.forward_with_activations(x)


def project_simba_weights_to_unit_norm(module: nn.Module, eps: float = 1e-12) -> None:
    """Project only SimbaV2 HyperDense output vectors onto the unit hypersphere."""
    with torch.no_grad():
        for layer in module.modules():
            if isinstance(layer, SimbaHyperDense):
                layer.weight.copy_(F.normalize(layer.weight, p=2, dim=1, eps=eps))
