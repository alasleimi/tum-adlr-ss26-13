from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


INHERITED_SAC_VERIFICATION_KIND = "exact_inherited_sac_state_v1"


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor values, names, dtypes, and shapes deterministically."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(np.asarray(tensor))
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def inherited_sac_source(
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return manifest-pinned source config/checkpoint fingerprints."""
    completion = candidate.get("completion")
    if not isinstance(completion, Mapping) or completion.get("kind") != "specialized_run_manifest":
        raise ValueError("inherited-critic SAC candidate lacks specialized completion provenance")
    source = completion.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("inherited-critic SAC candidate lacks a pinned source")
    config = source.get("config")
    checkpoint = source.get("checkpoint")
    for label, fingerprint in (("config", config), ("checkpoint", checkpoint)):
        if not isinstance(fingerprint, Mapping):
            raise ValueError(f"inherited-critic SAC source {label} is not fingerprinted")
        if not Path(str(fingerprint.get("path", ""))).is_file():
            raise FileNotFoundError(f"inherited-critic SAC source {label} is missing")
    return dict(config), dict(checkpoint)


def _module_hashes(modules: Any, label: str) -> list[str]:
    try:
        values = list(modules)
    except TypeError as exc:
        raise ValueError(f"SAC candidate has no iterable {label}") from exc
    if len(values) < 2:
        raise ValueError(f"SAC candidate must expose at least two {label}")
    return [state_dict_sha256(module.state_dict()) for module in values]


def _obs_rms_hash(agent: Any) -> str | None:
    obs_rms = getattr(agent, "obs_rms", None)
    return None if obs_rms is None else state_dict_sha256(obs_rms.state_dict())


def verify_inherited_sac_state(
    derived_agent: Any,
    source_agent: Any,
    *,
    derived_checkpoint: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove a specialized 3-observation SAC artifact retained its H0 Q state."""
    derived_online = _module_hashes(derived_agent.q_networks, "online critics")
    source_online = _module_hashes(source_agent.q_networks, "online critics")
    derived_target = _module_hashes(derived_agent.q_target_networks, "target critics")
    source_target = _module_hashes(source_agent.q_target_networks, "target critics")
    derived_obs_rms = _obs_rms_hash(derived_agent)
    source_obs_rms = _obs_rms_hash(source_agent)
    if derived_online != source_online:
        raise ValueError("specialized SAC checkpoint changed its source online critics")
    if derived_target != source_target:
        raise ValueError("specialized SAC checkpoint changed its source target critics")
    if derived_obs_rms != source_obs_rms:
        raise ValueError("specialized SAC checkpoint changed its source observation RMS")
    return {
        "kind": INHERITED_SAC_VERIFICATION_KIND,
        "verified_for_own_q_search": True,
        "derived_checkpoint": dict(derived_checkpoint),
        "source_checkpoint": dict(source_checkpoint),
        "online_critic_sha256": derived_online,
        "target_critic_sha256": derived_target,
        "observation_rms_sha256": derived_obs_rms,
        "online_critics_identical": True,
        "target_critics_identical": True,
        "observation_rms_identical": True,
    }


def validate_inherited_sac_verification(evidence: Mapping[str, Any]) -> None:
    if evidence.get("kind") != INHERITED_SAC_VERIFICATION_KIND:
        raise ValueError("inherited-critic SAC verification kind drift")
    if evidence.get("verified_for_own_q_search") is not True:
        raise ValueError("inherited-critic SAC state is not verified for own-Q search")
    for field in (
        "online_critics_identical",
        "target_critics_identical",
        "observation_rms_identical",
    ):
        if evidence.get(field) is not True:
            raise ValueError(f"inherited-critic SAC verification failed: {field}")
    online = evidence.get("online_critic_sha256")
    target = evidence.get("target_critic_sha256")
    if not isinstance(online, list) or len(online) < 2 or not all(isinstance(x, str) for x in online):
        raise ValueError("inherited-critic SAC online hashes are incomplete")
    if not isinstance(target, list) or len(target) < 2 or not all(isinstance(x, str) for x in target):
        raise ValueError("inherited-critic SAC target hashes are incomplete")
    for field in ("derived_checkpoint", "source_checkpoint"):
        fingerprint = evidence.get(field)
        if not isinstance(fingerprint, Mapping) or not fingerprint.get("sha256"):
            raise ValueError(f"inherited-critic SAC {field} fingerprint is incomplete")
