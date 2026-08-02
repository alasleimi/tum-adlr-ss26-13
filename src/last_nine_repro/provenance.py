from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

import pandas as pd

from .manifest import Finding, load_json


def normalize_run_path(value: object) -> str:
    text = str(value).replace("/", "\\")
    parts = PureWindowsPath(text).parts
    lowered = [part.lower() for part in parts]
    if "runs" in lowered:
        index = lowered.index("runs")
        parts = parts[index:]
    return "/".join(part.lower() for part in parts)


def load_provenance(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "provenance" / "mixed_lineages.json"
    if not path.is_file():
        raise FileNotFoundError(f"Mixed provenance ledger is missing: {path}")
    return load_json(path)


def _run_dirs_by_seed(frame: pd.DataFrame) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for seed, seed_frame in frame.groupby("actual_seed", sort=True):
        result[str(int(seed))] = sorted({normalize_run_path(value) for value in seed_frame["run_dir"]})
    return result


def audit_mixed_provenance(
    data_dir: Path,
    frames: Mapping[str, pd.DataFrame],
) -> list[Finding]:
    try:
        ledger = load_provenance(data_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [Finding("error", "PROVENANCE_LEDGER_INVALID", str(exc), "mixed_lineages.json")]

    findings: list[Finding] = []
    methods = ledger.get("methods", {})
    if not isinstance(methods, dict):
        return [Finding("error", "PROVENANCE_LEDGER_INVALID", "'methods' must be an object")]

    all_initializers: set[str] = set()
    for method, spec in methods.items():
        if not isinstance(spec, dict):
            findings.append(Finding("error", "PROVENANCE_METHOD_INVALID", "Invalid method row", method))
            continue
        initializer = spec.get("initializer_by_seed", {})
        if not isinstance(initializer, dict) or set(initializer) != {str(i) for i in range(5)}:
            findings.append(
                Finding(
                    "error",
                    "PROVENANCE_SEEDS_INCOMPLETE",
                    "Initializer provenance must cover seeds 0 through 4",
                    method,
                )
            )
        else:
            all_initializers.update(str(value) for value in initializer.values())

        expected_dirs = spec.get("rollout_run_dir_by_seed")
        if method in frames and isinstance(expected_dirs, dict):
            actual_dirs = _run_dirs_by_seed(frames[method])
            for seed, expected in expected_dirs.items():
                actual = actual_dirs.get(str(seed), [])
                expected_normalized = normalize_run_path(expected)
                if actual != [expected_normalized]:
                    findings.append(
                        Finding(
                            "error",
                            "PROVENANCE_RUN_DIR_MISMATCH",
                            f"seed {seed}: expected {expected_normalized!r}, found {actual!r}",
                            method,
                        )
                    )

    if len(all_initializers) <= 1:
        findings.append(
            Finding(
                "error",
                "PROVENANCE_SPLIT_NOT_DETECTED",
                "The documented mixed seed-0 versus seed-1--4 lineage split is absent",
                "mixed_lineages.json",
            )
        )
    else:
        findings.append(
            Finding(
                "warning",
                "PROV_MIXED_SEED_LINEAGE_SPLIT",
                "Mixed evidence uses a seed-0 initializer family distinct from seeds 1--4.",
                "mixed_lineages.json",
            )
        )

    sampler = ledger.get("shared_initializer_sampler", {})
    if not isinstance(sampler, dict) or float(sampler.get("near_upright_fraction", 0.0)) <= 0.0:
        findings.append(
            Finding(
                "error",
                "PROVENANCE_SAMPLER_CAVEAT_MISSING",
                "Angle-targeted initializer sampling is not encoded",
                "mixed_lineages.json",
            )
        )
    else:
        findings.append(
            Finding(
                "warning",
                "PROV_MIXED_ANGLE_TARGETED_INITIALIZER",
                "Retained mixed ablations descend from the documented angle-targeted initializer.",
                "mixed_lineages.json",
            )
        )

    contrasts = ledger.get("contrasts", [])
    if not isinstance(contrasts, list):
        findings.append(Finding("error", "PROVENANCE_CONTRASTS_INVALID", "'contrasts' must be an array"))
    else:
        for contrast in contrasts:
            if not isinstance(contrast, dict) or "id" not in contrast or "status" not in contrast:
                findings.append(Finding("error", "PROVENANCE_CONTRAST_INVALID", "Malformed contrast row"))
                continue
            status = str(contrast["status"])
            if status != "matched":
                findings.append(
                    Finding(
                        "warning",
                        "PROV_MIXED_CONTRAST_CAVEAT",
                        f"{contrast['id']}: {contrast.get('reason', status)}",
                        str(contrast["id"]),
                    )
                )
    return findings
