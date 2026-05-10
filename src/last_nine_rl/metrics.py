from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn


def parameter_norm(module: nn.Module) -> float:
    return parameter_list_norm(module.parameters())


def parameter_list_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    params = list(parameters)
    if not params:
        return 0.0
    total = torch.zeros((), device=params[0].device)
    for param in params:
        total = total + param.detach().pow(2).sum()
    return float(torch.sqrt(total).cpu())


def gradient_norm(module: nn.Module) -> float:
    return gradient_list_norm(module.parameters())


def gradient_list_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0
    total = torch.zeros((), device=params[0].device)
    for param in params:
        total = total + param.grad.detach().pow(2).sum()
    return float(torch.sqrt(total).cpu())


def snapshot_parameters(parameters: Iterable[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [param.detach().clone() for param in parameters]


def parameter_delta_norm(
    before: list[torch.Tensor],
    parameters: Iterable[torch.nn.Parameter],
) -> float:
    params = list(parameters)
    if not before or not params:
        return 0.0
    total = torch.zeros((), device=params[0].device)
    for old_param, new_param in zip(before, params, strict=True):
        total = total + (new_param.detach() - old_param.to(new_param.device)).pow(2).sum()
    return float(torch.sqrt(total).cpu())


def activation_statistics(
    name: str,
    activations: list[torch.Tensor],
    dormant_relative_threshold: float,
) -> dict[str, float]:
    stats: dict[str, float] = {}
    for idx, activation in enumerate(activations):
        x = activation.detach().float()
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
        mean_abs_by_unit = x.abs().mean(dim=0)
        layer_mean = mean_abs_by_unit.mean()
        if float(layer_mean.cpu()) <= 0.0:
            dormant_fraction = 1.0
        else:
            threshold = dormant_relative_threshold * layer_mean
            dormant_fraction = float((mean_abs_by_unit <= threshold).float().mean().cpu())

        stats[f"{name}_layer{idx}_activation_abs_mean"] = float(x.abs().mean().cpu())
        stats[f"{name}_layer{idx}_activation_abs_max"] = float(x.abs().max().cpu())
        stats[f"{name}_layer{idx}_dormant_fraction"] = dormant_fraction
        stats[f"{name}_layer{idx}_effective_rank_fraction"] = _effective_rank_fraction(x)
    return stats


def _effective_rank_fraction(x: torch.Tensor) -> float:
    if x.shape[0] < 2 or x.shape[1] < 2:
        return 1.0
    centered = (x - x.mean(dim=0, keepdim=True)).cpu()
    try:
        singular_values = torch.linalg.svdvals(centered)
    except RuntimeError:
        return math.nan
    total = singular_values.sum()
    if float(total.cpu()) <= 0.0:
        return 0.0
    probabilities = singular_values / total
    entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum()
    effective_rank = torch.exp(entropy)
    normalizer = min(centered.shape[0], centered.shape[1])
    return float((effective_rank / normalizer).cpu())
