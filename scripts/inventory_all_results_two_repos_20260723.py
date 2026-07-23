from __future__ import annotations

"""Forensic inventory of every stored experiment and result in two worktrees.

This crawler deliberately keeps four units of account separate:

1. filesystem artifacts, so a result cannot disappear because it was not listed;
2. physical run instances, including partial and checkpoint-only runs;
3. intended multi-seed run families;
4. standardized grid evaluations, with aliases deduplicated by row-table hash.

The older five-seed leaderboard crawler remains the source of the recursive
learning-transition ledger. This script broadens discovery to every seed count,
every training budget, incomplete runs, and the neighboring SAC-N worktree.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import inventory_systematic_results as five_seed


EXPECTED_CELLS = 2501
EXPECTED_EPSILON = 5.0
TARGET_STEPS = 100_000
SEED_RE = re.compile(r"(?i)(?:^|[/_\-])seed[_=\-]?(\d+)(?=$|[/_\-])")
STEP_HINT_RE = re.compile(r"(?i)(?<!\d)(\d{1,4})k(?!\w)")
ARTIFACT_ROOTS = ("reports", "runs", "configs", "scripts", "docs")
ROOT_ARTIFACT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".pdf",
    ".png",
    ".svg",
    ".pptx",
}


def read_json_value(path: Path) -> Any:
    raw = path.read_bytes()
    if raw.startswith(b"version https://git-lfs.github.com/spec"):
        raise ValueError("Git LFS pointer is present instead of materialized JSON")
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            value = json.loads(raw.decode(encoding))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            errors.append(f"{encoding}: {error}")
            continue
        return value
    raise ValueError("; ".join(errors))


def read_json(path: Path) -> dict[str, Any]:
    value = read_json_value(path)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is {type(value).__name__}, expected object")
    return value


def decode_text(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    if raw.startswith(b"version https://git-lfs.github.com/spec"):
        raise ValueError("Git LFS pointer is present instead of materialized text")
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError as error:
            errors.append(f"{encoding}: {error}")
    raise ValueError("; ".join(errors))


def safe_json(path: Path) -> dict[str, Any]:
    try:
        return read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {}


def as_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def pipe(values: Iterable[Any]) -> str:
    return " | ".join(str(value) for value in values if value not in (None, ""))


def repo_relative(repo: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def canonical_run_rel(path: Path | str | None) -> str:
    if path is None:
        return ""
    normalized = str(path).replace("\\", "/")
    match = re.search(r"(?i)(?:^|/)runs/(.+)$", normalized)
    return "runs/" + match.group(1).strip("/") if match else ""


def run_dir_from_checkpoint(path: str | None) -> str:
    rel = canonical_run_rel(path)
    if not rel:
        return ""
    parts = Path(rel).parts
    if len(parts) >= 3 and parts[-2].lower() == "checkpoints":
        return Path(*parts[:-2]).as_posix()
    return Path(*parts[:-1]).as_posix()


class HashCache:
    def __init__(self) -> None:
        self.cache: dict[Path, str] = {}

    def sha256(self, path: Path) -> str:
        path = path.resolve()
        if path in self.cache:
            return self.cache[path]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        value = digest.hexdigest()
        self.cache[path] = value
        return value


def is_within(path: Path, directory: Path | None) -> bool:
    if directory is None:
        return False
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def candidate_artifact_files(repo: Path) -> Iterable[Path]:
    yielded: set[Path] = set()
    for root in ARTIFACT_ROOTS:
        base = repo / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path not in yielded
            ):
                yielded.add(path)
                yield path
    for path in repo.iterdir():
        if (
            path.is_file()
            and path.suffix.lower() in ROOT_ARTIFACT_SUFFIXES
            and path not in yielded
        ):
            yield path


def snapshot_file_state(
    repos: Sequence[tuple[str, Path]], excluded: Path | None = None
) -> dict[tuple[str, str], tuple[int, int]]:
    state: dict[tuple[str, str], tuple[int, int]] = {}
    for repo_id, repo in repos:
        for path in candidate_artifact_files(repo):
            if is_within(path, excluded):
                continue
            stat = path.stat()
            state[(repo_id, repo_relative(repo, path))] = (
                stat.st_size,
                stat.st_mtime_ns,
            )
    return state


def compare_snapshots(
    before: Mapping[tuple[str, str], tuple[int, int]],
    after: Mapping[tuple[str, str], tuple[int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(set(before) | set(after)):
        left = before.get(key)
        right = after.get(key)
        if left == right:
            continue
        if left is None:
            change = "created_during_scan"
        elif right is None:
            change = "removed_during_scan"
        else:
            change = "modified_during_scan"
        rows.append(
            {
                "repo_id": key[0],
                "relative_path": key[1],
                "change": change,
                "bytes_before": left[0] if left else "",
                "bytes_after": right[0] if right else "",
                "mtime_ns_before": left[1] if left else "",
                "mtime_ns_after": right[1] if right else "",
            }
        )
    return rows


def artifact_role(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    parent = path.parent.name.lower()
    if name == "config.json":
        return "run_config"
    if name == "distillation_summary.json":
        return "distillation_summary"
    if name == "training_summary.json":
        return "custom_training_summary"
    if name == "run_manifest.json":
        return "custom_run_manifest"
    if name == "summary.json":
        return "generic_summary"
    if name == "relative_summary.json":
        return "relative_evaluation_summary"
    if name == "authoritative_summary.json":
        return "authoritative_evaluation_summary"
    if name == "pendulum_grid_summary.json":
        return "grid_summary"
    if name == "pendulum_grid_summary.csv":
        return "grid_summary_table"
    if name == "posthoc_eval_summary.json":
        return "posthoc_summary"
    if name.endswith("aggregate.json") or name == "aggregate.json":
        return "aggregate_summary"
    if name == "relative_rollouts.csv":
        return "relative_rollout_rows"
    if name == "pendulum_grid_rollouts.csv":
        return "grid_rollout_rows"
    if name == "eval_episodes.csv":
        return "run_eval_rows"
    if name == "metrics.csv":
        return "run_metric_rows"
    if name == "events.jsonl":
        return "run_event_log"
    if name == "final.pt":
        return "final_checkpoint"
    if suffix == ".pt" and parent == "checkpoints":
        return "intermediate_checkpoint"
    if suffix == ".npz" and "replay" in name:
        return "replay_buffer"
    if suffix in {".out", ".err", ".log", ".jsonl"}:
        return "process_log"
    if suffix in {".md", ".html", ".pdf"}:
        return "human_report"
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".svg"}:
        return "plot_or_animation"
    if suffix == ".csv":
        return "derived_table"
    if suffix == ".json":
        return "derived_json"
    if suffix in {".ps1", ".py", ".sh"}:
        return "analysis_or_runner_code"
    return "other"


HASHED_ROLES = {
    "run_config",
    "distillation_summary",
    "custom_training_summary",
    "custom_run_manifest",
    "generic_summary",
    "relative_evaluation_summary",
    "authoritative_evaluation_summary",
    "grid_summary",
    "posthoc_summary",
    "aggregate_summary",
    "relative_rollout_rows",
    "grid_rollout_rows",
    "human_report",
    "derived_table",
    "derived_json",
}


def build_artifact_manifest(
    repos: Sequence[tuple[str, Path]],
    hash_cache: HashCache,
    excluded: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_repo: dict[str, dict[str, Path]] = {}
    for repo_id, repo in repos:
        files: dict[str, Path] = {}
        for path in candidate_artifact_files(repo):
            if is_within(path, excluded):
                continue
            files[repo_relative(repo, path)] = path
        per_repo[repo_id] = files

    primary_id, _primary = repos[0]
    neighbor_id, _neighbor = repos[1]
    primary_files = per_repo[primary_id]
    neighbor_files = per_repo[neighbor_id]
    all_relpaths = sorted(set(primary_files) | set(neighbor_files))
    reconciliation: list[dict[str, Any]] = []
    relation_by_pair: dict[tuple[str, str], str] = {}
    for relpath in all_relpaths:
        left = primary_files.get(relpath)
        right = neighbor_files.get(relpath)
        if left is None:
            relation = "neighbor_only"
        elif right is None:
            relation = "primary_only"
        elif left.stat().st_size != right.stat().st_size:
            relation = "same_path_changed"
        else:
            relation = (
                "exact_mirror"
                if hash_cache.sha256(left) == hash_cache.sha256(right)
                else "same_path_changed"
            )
        reconciliation.append(
            {
                "relative_path": relpath,
                "artifact_role": artifact_role(left or right),  # type: ignore[arg-type]
                "project15_exists": left is not None,
                "sacn_worktree_exists": right is not None,
                "relation": relation,
                "project15_bytes": left.stat().st_size if left else "",
                "sacn_worktree_bytes": right.stat().st_size if right else "",
            }
        )
        if left is not None:
            relation_by_pair[(primary_id, relpath)] = relation
        if right is not None:
            relation_by_pair[(neighbor_id, relpath)] = relation

    manifest: list[dict[str, Any]] = []
    for repo_id, repo in repos:
        for relpath, path in sorted(per_repo[repo_id].items()):
            role = artifact_role(path)
            should_hash = role in HASHED_ROLES and path.stat().st_size <= 12_000_000
            manifest.append(
                {
                    "repo_id": repo_id,
                    "relative_path": relpath,
                    "artifact_role": role,
                    "bytes": path.stat().st_size,
                    "modified_utc": pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat(),
                    "sha256": hash_cache.sha256(path) if should_hash else "",
                    "cross_repo_relation": relation_by_pair[(repo_id, relpath)],
                }
            )
    return manifest, reconciliation


def metric_value(value: Any) -> tuple[float | None, int | None, int | None]:
    if isinstance(value, (int, float)):
        number = as_float(value)
        return number, None, None
    if not isinstance(value, Mapping):
        return None, None, None
    successes = as_int(value.get("successes"))
    total = as_int(value.get("total"))
    if total is None:
        total = as_int(value.get("trials"))
    rate = as_float(value.get("rate"))
    if rate is None and successes is not None and total:
        rate = successes / total
    return rate, successes, total


def metric_nodes(
    value: Any, pointer: str = ""
) -> Iterable[tuple[str, Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        keys = set(value)
        strict_key = next(
            (
                key
                for key in (
                    "beats_best_known_return",
                    "strict_beats_best_known_return",
                )
                if key in keys
            ),
            None,
        )
        if (
            "near_best_known_return_eps" in keys
            and "task_success" in keys
            and strict_key is not None
        ):
            yield pointer or "/", value
        for key, child in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield from metric_nodes(child, f"{pointer}/{escaped}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from metric_nodes(child, f"{pointer}/{index}")


def discover_metric_jsons(
    repos: Sequence[tuple[str, Path]], excluded: Path | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for repo_id, repo in repos:
        for path in sorted((repo / "reports").rglob("*.json")):
            if is_within(path, excluded):
                continue
            try:
                payload = read_json_value(path)
                parse_status = "parsed"
                error = ""
            except Exception as exc:
                payload = {}
                parse_status = "failed"
                error = str(exc)
            nodes = list(metric_nodes(payload))
            audit_rows.append(
                {
                    "repo_id": repo_id,
                    "json_path": repo_relative(repo, path),
                    "bytes": path.stat().st_size,
                    "parse_status": parse_status,
                    "parse_error": error,
                    "metric_signature_nodes": len(nodes),
                }
            )
            for pointer, node in nodes:
                strict_key = (
                    "beats_best_known_return"
                    if "beats_best_known_return" in node
                    else "strict_beats_best_known_return"
                )
                near_rate, near_count, near_total = metric_value(
                    node["near_best_known_return_eps"]
                )
                task_rate, task_count, task_total = metric_value(node["task_success"])
                strict_rate, strict_count, strict_total = metric_value(node[strict_key])
                metric_rows.append(
                    {
                        "repo_id": repo_id,
                        "json_path": repo_relative(repo, path),
                        "json_pointer": pointer,
                        "summary_basename": path.name,
                        "near_rate": near_rate,
                        "near_successes": near_count,
                        "near_total": near_total,
                        "task_rate": task_rate,
                        "task_successes": task_count,
                        "task_total": task_total,
                        "strict_rate": strict_rate,
                        "strict_successes": strict_count,
                        "strict_total": strict_total,
                        "strict_metric_key": strict_key,
                        "handled_by_standard_summary_crawler": path.name
                        in {"relative_summary.json", "authoritative_summary.json"}
                        and pointer in {"/", "/criteria", "/result"},
                    }
                )
    return metric_rows, audit_rows


RAW_OR_REGIONAL_TABLE_NAMES = {
    "relative_rollouts.csv",
    "relative_cell_summary.csv",
    "relative_region_summary.csv",
    "relative_criterion_summary.csv",
    "pendulum_grid_rollouts.csv",
    "eval_episodes.csv",
    "metrics.csv",
    "training_metrics.csv",
    "validation_metrics.csv",
    "rl_target_metrics.csv",
    "dagger_collection_metrics.csv",
    "posthoc_eval_episodes.csv",
    "controller_grid.csv",
}


def _metric_column(
    columns: Sequence[str], tokens: Sequence[str]
) -> str | None:
    normalized = {column: column.lower().strip() for column in columns}
    for token in tokens:
        for column, lowered in normalized.items():
            if token in lowered:
                return column
    return None


def _metric_rate_column(
    columns: Sequence[str], tokens: Sequence[str]
) -> str | None:
    candidates = [
        column
        for column in columns
        if any(token in column.lower().strip() for token in tokens)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda column: (
            "rate" in column.lower()
            or "percent" in column.lower()
            or "pct" in column.lower(),
            "successes" not in column.lower()
            and "count" not in column.lower(),
            -columns.index(column),
        ),
    )


def _rate_from_tabular_value(value: Any) -> float | None:
    parsed = as_float(value)
    if parsed is None:
        return None
    return parsed / 100.0 if parsed > 1.0 + 1e-9 else parsed


def _tabular_metric_rate(
    row: Mapping[str, Any], column: str | None, total_column: str | None
) -> float | None:
    if not column:
        return None
    parsed = as_float(row.get(column))
    if parsed is None:
        return None
    lowered = column.lower()
    if "successes" in lowered or lowered.endswith("_count"):
        total = as_float(row.get(total_column)) if total_column else None
        return parsed / total if total and total > 0 else None
    return _rate_from_tabular_value(parsed)


def discover_tabular_metric_rows(
    repos: Sequence[tuple[str, Path]],
    excluded: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Index aggregate result rows that are not canonical JSON summaries.

    Raw rollout, per-cell, and trainer telemetry tables are intentionally
    excluded here because they are already accounted for in artifact_manifest
    and, for relative rollouts, in the canonical-evaluation table.
    """

    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for repo_id, repo in repos:
        for path in sorted((repo / "reports").rglob("*.csv")):
            if is_within(path, excluded):
                continue
            lowered_name = path.name.lower()
            if (
                lowered_name in RAW_OR_REGIONAL_TABLE_NAMES
                or lowered_name.endswith("_rows.csv")
                or lowered_name.endswith("_grid.csv")
            ):
                continue
            try:
                text, encoding = decode_text(path)
                reader = csv.DictReader(text.splitlines())
                columns = list(reader.fieldnames or [])
            except Exception as error:
                audit.append(
                    {
                        "repo_id": repo_id,
                        "csv_path": repo_relative(repo, path),
                        "parse_status": "failed",
                        "parse_error": str(error),
                        "encoding": "",
                        "result_rows": 0,
                    }
                )
                continue
            near_column = _metric_rate_column(
                columns,
                (
                    "near_best_known_return",
                    "near_reference",
                    "reference_success",
                ),
            )
            task_column = _metric_rate_column(columns, ("task_success",))
            beats_column = _metric_rate_column(
                columns,
                (
                    "beats_best_known_return",
                    "strict_beats_best_known_return",
                    "beats_reference",
                ),
            )
            historical_strict_column = _metric_rate_column(
                columns, ("strict_success",)
            )
            total_column = _metric_column(
                columns, ("trials", "total_evaluations", "total")
            )
            metric_columns = [
                column
                for column in (
                    near_column,
                    task_column,
                    beats_column,
                    historical_strict_column,
                )
                if column
            ]
            if len(set(metric_columns)) < 2:
                continue
            method_column = _metric_column(
                columns,
                (
                    "method",
                    "condition",
                    "candidate",
                    "recipe",
                    "experiment",
                    "label",
                ),
            )
            seed_column = _metric_column(
                columns, ("seed_count", "num_seeds", "n_seeds", "seeds")
            )
            step_column = _metric_column(
                columns,
                (
                    "environment_steps",
                    "learning_steps",
                    "total_steps",
                    "steps",
                ),
            )
            result_count = 0
            for index, raw_row in enumerate(reader, start=2):
                populated = {
                    str(key): value
                    for key, value in raw_row.items()
                    if key is not None and value not in (None, "")
                }
                if not populated:
                    continue
                metric_values = [
                    _tabular_metric_rate(populated, column, total_column)
                    for column in set(metric_columns)
                ]
                if not any(value is not None for value in metric_values):
                    continue
                result_count += 1
                raw_identity = hashlib.sha256(
                    json.dumps(
                        populated, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest()
                rows.append(
                    {
                        "repo_id": repo_id,
                        "csv_path": repo_relative(repo, path),
                        "row_number": index,
                        "row_sha256": raw_identity,
                        "method_or_condition": populated.get(method_column, "")
                        if method_column
                        else "",
                        "seed_count_hint": as_int(populated.get(seed_column))
                        if seed_column
                        else "",
                        "less_than_5_seed_hint": (
                            as_int(populated.get(seed_column)) < 5
                            if seed_column
                            and as_int(populated.get(seed_column)) is not None
                            else ""
                        ),
                        "step_hint": as_int(populated.get(step_column))
                        if step_column
                        else step_hint(repo_relative(repo, path)),
                        "near_rate": _tabular_metric_rate(
                            populated, near_column, total_column
                        )
                        if near_column
                        else "",
                        "task_rate": _tabular_metric_rate(
                            populated, task_column, total_column
                        )
                        if task_column
                        else "",
                        "beats_reference_rate": _tabular_metric_rate(
                            populated, beats_column, total_column
                        )
                        if beats_column
                        else "",
                        "historical_strict_rate_not_beats_reference": (
                            _tabular_metric_rate(
                                populated,
                                historical_strict_column,
                                total_column,
                            )
                            if historical_strict_column
                            and historical_strict_column != beats_column
                            else ""
                        ),
                        "metric_columns": pipe(sorted(set(metric_columns))),
                        "raw_row_json": json.dumps(
                            populated, sort_keys=True, ensure_ascii=False
                        ),
                    }
                )
            audit.append(
                {
                    "repo_id": repo_id,
                    "csv_path": repo_relative(repo, path),
                    "parse_status": "parsed",
                    "parse_error": "",
                    "encoding": encoding,
                    "result_rows": result_count,
                }
            )
    return rows, audit


def _walk_items(value: Any, pointer: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_pointer = f"{pointer}/{key}"
            yield child_pointer, child
            yield from _walk_items(child, child_pointer)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_items(child, f"{pointer}/{index}")


def _run_rels_in_value(value: Any) -> list[str]:
    output: list[str] = []
    for _pointer, child in _walk_items(value):
        if not isinstance(child, str):
            continue
        rel = canonical_run_rel(child)
        if rel:
            output.append(rel)
    return list(dict.fromkeys(output))


def _deployment_flags(value: Any) -> dict[str, bool]:
    hard_router = False
    reference_at_inference = False
    mixture = False
    router_active = False
    router_has_fixed_region = False
    for pointer, child in _walk_items(value):
        lowered = pointer.lower()
        if "router" in lowered:
            if isinstance(child, str):
                mode = child.lower().strip()
                if mode not in {
                    "",
                    "none",
                    "disabled",
                    "global",
                    "reflection",
                    "qsearch",
                    "learned",
                    "automatic",
                }:
                    router_active = True
            elif isinstance(child, bool):
                router_active = router_active or child
        if "router" in lowered and any(
            token in lowered
            for token in ("angle", "theta", "velocity", "omega", "range", "bound")
        ):
            if child not in (None, "", False, 0, 0.0):
                router_has_fixed_region = True
        if any(
            token in lowered
            for token in (
                "reference_at_inference",
                "runtime_reference",
                "reference_policy_calls",
                "reference_calls",
            )
        ):
            numeric = as_float(child)
            if child is True or (numeric is not None and numeric > 0):
                reference_at_inference = True
        if any(
            token in lowered
            for token in (
                "actor_mixture",
                "model_mixture",
                "ensemble_members",
                "specialist_members",
            )
        ):
            numeric = as_float(child)
            if child is True or (numeric is not None and numeric > 1):
                mixture = True
    hard_router = router_active and router_has_fixed_region
    return {
        "hardcoded_inference_router": hard_router,
        "reference_at_inference": reference_at_inference,
        "inference_model_mixture": mixture,
    }


def _classification_from_runs(
    run_rels: Sequence[str],
    primary_run_map: Mapping[str, Mapping[str, Any]],
    fallback_text: str,
) -> dict[str, Any]:
    direct = [
        primary_run_map[run_rel]
        for run_rel in run_rels
        if run_rel in primary_run_map
    ]
    categories = {
        str(row.get("category_direct", "unknown")) for row in direct
    }
    if "RL + supervised" in categories or (
        "pure RL" in categories and "supervised only" in categories
    ):
        category = "RL + supervised"
    elif categories == {"supervised only"}:
        category = "supervised only"
    elif categories and categories <= {"pure RL"}:
        category = "pure RL"
    else:
        lowered = fallback_text.lower()
        supervised_tokens = (
            "dagger",
            "distill",
            "reference_guidance",
            "reference_label",
            "behavior_clone",
            "bc_only",
        )
        rl_tokens = ("sac", "simba", "fastsacn", "sacn", "redo", "redq")
        if any(token in lowered for token in supervised_tokens):
            category = "RL + supervised"
        elif any(token in lowered for token in rl_tokens):
            category = "pure RL"
        else:
            category = "unknown"
    manual_evidence = sorted(
        {
            str(row["manual_state_region_evidence"])
            for row in direct
            if row.get("manual_state_region_evidence")
        }
    )
    if not manual_evidence:
        lowered = fallback_text.lower()
        token = next(
            (
                candidate
                for candidate in (
                    "hard_reset",
                    "hard_replay",
                    "hard120",
                    "near_down",
                    "neardown",
                    "targeted_failuremix",
                    "fixed_angle_window",
                    "modelrollout",
                )
                if candidate in lowered
            ),
            None,
        )
        if token:
            manual_evidence.append(
                f"name/row declares fixed training-state region token '{token}'"
            )
    return {
        "category": category,
        "manual_state_region_training": bool(manual_evidence),
        "manual_state_region_evidence": pipe(manual_evidence),
        "resolved_run_paths": pipe(run_rels),
        "classification_basis": (
            "resolved run metadata" if direct else "schema/name heuristic"
        ),
    }


def enrich_noncanonical_result_rows(
    metric_json_rows: list[dict[str, Any]],
    tabular_rows: list[dict[str, Any]],
    aliases: Sequence[Mapping[str, Any]],
    repos: Sequence[tuple[str, Path]],
    primary_run_map: Mapping[str, Mapping[str, Any]],
) -> None:
    tabular_alias_groups: dict[tuple[str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in tabular_rows:
        tabular_alias_groups[
            (str(row["csv_path"]), str(row["row_sha256"]))
        ].append(row)
    for group_key, group in tabular_alias_groups.items():
        identity = hashlib.sha256(
            f"{group_key[0]}|{group_key[1]}".encode()
        ).hexdigest()[:16]
        for row in group:
            row["canonical_tabular_row_id"] = identity
            row["cross_repo_alias_count"] = len(group)
            row["present_in_both_repos"] = (
                len({str(item["repo_id"]) for item in group}) > 1
            )
            row["is_primary_tabular_representative"] = (
                row["repo_id"] == repos[0][0]
            )
    alias_map = {
        (str(row["repo_id"]), str(row["summary_path"])): row for row in aliases
    }
    repo_map = dict(repos)
    payload_cache: dict[tuple[str, str], Any] = {}
    for row in metric_json_rows:
        key = (str(row["repo_id"]), str(row["json_path"]))
        alias = alias_map.get(key)
        if alias is not None:
            row.update(
                {
                    "category": alias["category"],
                    "pure_rl": alias["category"] == "pure RL",
                    "manual_state_region_training": alias[
                        "manual_state_region_training"
                    ],
                    "hardcoded_inference_router": alias[
                        "hardcoded_inference_router"
                    ],
                    "reference_at_inference": alias["reference_at_inference"],
                    "inference_model_mixture": alias[
                        "inference_model_mixture"
                    ],
                    "cheating_status": alias["cheating_status"],
                    "seed_count": alias["seed_count"],
                    "less_than_5_seeds": alias["less_than_5_seeds"],
                    "classification_basis": "canonical evaluation lineage",
                    "resolved_run_paths": alias["run_paths"],
                    "result_disposition": "canonical evaluation metric or metric slice",
                }
            )
            continue
        if key not in payload_cache:
            try:
                payload_cache[key] = read_json_value(
                    repo_map[key[0]] / key[1]
                )
            except Exception:
                payload_cache[key] = {}
        payload = payload_cache[key]
        run_rels = _run_rels_in_value(payload)
        fallback = f"{row['json_path']} {json.dumps(payload, default=str)}"
        inherited = _classification_from_runs(
            run_rels, primary_run_map, fallback
        )
        deployment = _deployment_flags(payload)
        manual = bool(inherited["manual_state_region_training"])
        cheating = (
            manual
            or deployment["hardcoded_inference_router"]
            or deployment["reference_at_inference"]
        )
        total = (
            as_int(row.get("near_total"))
            or as_int(row.get("task_total"))
            or as_int(row.get("strict_total"))
        )
        seed_count = (
            total // EXPECTED_CELLS
            if total and total % EXPECTED_CELLS == 0
            else None
        )
        row.update(
            {
                **inherited,
                **deployment,
                "pure_rl": inherited["category"] == "pure RL",
                "cheating_status": (
                    "cheating"
                    if cheating
                    else "clean"
                    if inherited["classification_basis"]
                    == "resolved run metadata"
                    else "unknown"
                ),
                "seed_count": seed_count if seed_count is not None else "",
                "less_than_5_seeds": (
                    seed_count < 5 if seed_count is not None else ""
                ),
                "result_disposition": "noncanonical report or diagnostic metric",
            }
        )

    for row in tabular_rows:
        try:
            payload = json.loads(str(row["raw_row_json"]))
        except json.JSONDecodeError:
            payload = {}
        run_rels = _run_rels_in_value(payload)
        fallback = (
            f"{row['csv_path']} {row.get('method_or_condition', '')} "
            f"{row.get('raw_row_json', '')}"
        )
        inherited = _classification_from_runs(
            run_rels, primary_run_map, fallback
        )
        deployment = _deployment_flags(payload)
        manual = bool(inherited["manual_state_region_training"])
        cheating = (
            manual
            or deployment["hardcoded_inference_router"]
            or deployment["reference_at_inference"]
        )
        basis_known = inherited["classification_basis"] == "resolved run metadata"
        seed_count = as_int(row.get("seed_count_hint"))
        row.update(
            {
                **inherited,
                **deployment,
                "pure_rl": inherited["category"] == "pure RL",
                "cheating_status": (
                    "cheating"
                    if cheating
                    else "clean"
                    if basis_known
                    else "unknown"
                ),
                "seed_count": seed_count if seed_count is not None else "",
                "less_than_5_seeds": (
                    seed_count < 5 if seed_count is not None else ""
                ),
                "result_disposition": "aggregate/report-only/diagnostic table row",
            }
        )


def positive(config: Mapping[str, Any], key: str) -> bool:
    return (as_float(config.get(key), 0.0) or 0.0) > 0


def manual_region_evidence(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    distillation: Mapping[str, Any],
) -> list[str]:
    env = config.get("env", {}) or {}
    sac = config.get("sac", {}) or {}
    evidence: list[str] = []
    if positive(env, "pendulum_hard_reset_prob") or positive(
        env, "pendulum_hard_reset_final_prob"
    ):
        evidence.append("nonzero hardcoded-angle reset probability")
    if positive(sac, "pendulum_hard_replay_fraction") or positive(
        sac, "pendulum_hard_replay_final_fraction"
    ):
        evidence.append("nonzero hardcoded-angle replay fraction")
    if positive(sac, "pendulum_model_replay_ratio"):
        evidence.append("model replay selected by fixed angle/velocity bounds")
    if positive(sac, "pendulum_model_rollout_ratio"):
        evidence.append("model rollouts selected by fixed angle/velocity bounds")
    if positive(sac, "pendulum_potential_shaping_weight"):
        low = as_float(sac.get("pendulum_potential_shaping_abs_theta_low"), 0.0) or 0.0
        high = as_float(
            sac.get("pendulum_potential_shaping_abs_theta_high"), math.pi
        ) or math.pi
        velocity = as_float(sac.get("pendulum_potential_shaping_velocity_limit"), 0.0) or 0.0
        if low > 0 or high < math.pi - 1e-9 or velocity > 0:
            evidence.append("potential shaping restricted by fixed state bounds")
    anchor_ratio = as_float(sac.get("reference_anchor_ratio"), 0.0) or 0.0
    if anchor_ratio > 0:
        velocity = as_float(sac.get("reference_anchor_velocity_limit"), 0.0) or 0.0
        reset_velocity = (
            as_float(sac.get("reference_anchor_reset_velocity_limit"), 0.0) or 0.0
        )
        if velocity > 0 or reset_velocity > 0:
            evidence.append("reference anchors selected by fixed velocity bounds")

    combined = [summary, distillation]
    for payload in combined:
        source_counts = payload.get("static_source_counts", {}) or {}
        for key, value in source_counts.items():
            if (as_int(value, 0) or 0) <= 0:
                continue
            text = str(key).lower()
            if any(token in text for token in ("near_down", "hard", "window", "band")):
                evidence.append(f"static supervised source uses manual region '{key}'")
        if payload.get("tight_failure_bands"):
            evidence.append("summary declares tight fixed failure bands")
        for key in (
            "hard_teacher_fraction",
            "targeted_failure_fraction",
            "near_down_fraction",
            "failure_window_fraction",
        ):
            if positive(payload, key):
                evidence.append(f"summary enables {key}")
        initial_source = str(payload.get("initial_dataset_source", "")).lower()
        if any(
            token in initial_source
            for token in ("near_down", "hard_window", "hard_band", "targeted_angle")
        ):
            evidence.append(f"initial dataset source '{initial_source}'")
    return list(dict.fromkeys(evidence))


def automatic_selection_evidence(
    config: Mapping[str, Any], summary: Mapping[str, Any]
) -> list[str]:
    env = config.get("env", {}) or {}
    sac = config.get("sac", {}) or {}
    evidence: list[str] = []
    if (as_int(env.get("pendulum_failure_curriculum_candidate_count"), 0) or 0) > 0:
        evidence.append("automatic reward-based failure curriculum")
    if str(sac.get("replay_priority_mode", "none")).lower() not in {
        "",
        "none",
        "uniform",
    }:
        evidence.append(f"automatic replay priority mode {sac.get('replay_priority_mode')}")
    for row in summary.get("dagger_collection", []) or []:
        if (as_int(row.get("priority_candidate_count"), 0) or 0) > 0:
            evidence.append("automatic performance-priority DAgger collection")
    return list(dict.fromkeys(evidence))


def reference_training_evidence(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    distillation: Mapping[str, Any],
) -> list[str]:
    sac = config.get("sac", {}) or {}
    evidence: list[str] = []
    if distillation:
        evidence.append("distillation/DAgger summary uses reference labels")
    if summary.get("dagger_collection") is not None:
        evidence.append("corrected DAgger collection")
    if summary.get("collections") is not None:
        evidence.append("custom DAgger/reference-labelled collections")
    if summary.get("uses_reference_labels"):
        evidence.append("summary declares reference labels")
    for key in (
        "reference_guidance_mode",
        "reference_prior_mode",
        "reference_auxiliary_mode",
        "reference_critic_mode",
    ):
        value = str(sac.get(key, "none")).lower()
        if value not in {"", "none"}:
            evidence.append(f"{key}={value}")
    for key in (
        "reference_guidance_probability",
        "reference_prior_dataset_steps",
        "reference_auxiliary_weight",
        "reference_critic_weight",
    ):
        if positive(sac, key):
            evidence.append(f"{key}>0")
    if positive(sac, "pendulum_potential_shaping_weight"):
        source = str(
            sac.get("pendulum_potential_shaping_source", "")
        ).lower()
        if any(
            token in source
            for token in ("best", "reference", "controller", "dp")
        ):
            evidence.append(
                f"reference-derived potential shaping source={source}"
            )
    return list(dict.fromkeys(evidence))


def dependency_run_rels(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    distillation: Mapping[str, Any],
    training_summary: Mapping[str, Any] | None = None,
    run_manifest: Mapping[str, Any] | None = None,
) -> list[str]:
    dependencies: list[str] = []
    sac = config.get("sac", {}) or {}
    for raw in (
        summary.get("source_run"),
        summary.get("dagger_source_run"),
        summary.get("rl_critic_source_run"),
    ):
        rel = canonical_run_rel(raw)
        if rel:
            dependencies.append(rel)
    for raw in (
        distillation.get("init_checkpoint"),
        sac.get("actor_init_checkpoint_path"),
    ):
        rel = run_dir_from_checkpoint(raw)
        if rel:
            dependencies.append(rel)
    for payload in (summary, distillation, training_summary or {}, run_manifest or {}):
        stack: list[tuple[str, Any]] = [("", payload)]
        while stack:
            key, value = stack.pop()
            if isinstance(value, Mapping):
                stack.extend((str(child_key), child) for child_key, child in value.items())
            elif isinstance(value, list):
                stack.extend((key, child) for child in value)
            elif isinstance(value, str) and any(
                token in key.lower()
                for token in (
                    "source_run",
                    "frozen_simbav2_source_run",
                    "source_checkpoint",
                    "init_checkpoint",
                    "critic_run",
                )
            ):
                rel = (
                    run_dir_from_checkpoint(value)
                    if "checkpoint" in key.lower()
                    else canonical_run_rel(value)
                )
                if rel:
                    dependencies.append(rel)
    return list(dict.fromkeys(dependencies))


def event_progress(path: Path) -> dict[str, Any]:
    result = {
        "max_event_step": None,
        "run_complete_step": None,
        "run_complete_updates": None,
        "checkpoint_saved_steps": [],
    }
    if not path.exists():
        return result
    max_step: int | None = None
    checkpoint_steps: list[int] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                step = as_int(event.get("step"))
                if step is not None:
                    max_step = step if max_step is None else max(max_step, step)
                if event.get("type") == "checkpoint_saved" and step is not None:
                    checkpoint_steps.append(step)
                if event.get("type") == "run_complete":
                    result["run_complete_step"] = step
                    result["run_complete_updates"] = as_int(
                        (event.get("payload") or {}).get("updates")
                    )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        result["event_parse_error"] = True
    result["max_event_step"] = max_step
    result["checkpoint_saved_steps"] = checkpoint_steps
    return result


def eval_episode_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {"eval_parse_error": True}
    if frame.empty or "step" not in frame:
        return {}
    step = int(frame["step"].max())
    final = frame[frame["step"] == step]
    task_column = next(
        (
            key
            for key in ("task_success", "success", "strict_success")
            if key in final.columns
        ),
        None,
    )
    return {
        "local_eval_step": step,
        "local_eval_episodes": len(final),
        "local_eval_mean_return": float(final["return"].mean())
        if "return" in final
        else None,
        "local_eval_task_rate": float(final[task_column].mean())
        if task_column
        else None,
        "local_eval_strict_rate": float(final["strict_success"].mean())
        if "strict_success" in final
        else None,
        "local_eval_collapse_rate": float(final["collapse"].mean())
        if "collapse" in final
        else None,
    }


def metrics_file_integrity(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path.stat().st_size == 0:
        return "empty"
    try:
        raw = path.read_bytes()
    except OSError:
        return "read_error"
    if b"\x00" in raw:
        return "nul_corrupted"
    try:
        text = raw.decode("utf-8-sig")
        header = next(csv.reader(text.splitlines()), [])
    except (UnicodeDecodeError, csv.Error, StopIteration):
        return "csv_parse_error"
    return "ok" if len(header) >= 2 else "invalid_header"


def normalize_signature_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(config))
    payload.pop("seed", None)
    sac = payload.get("sac", {}) or {}
    sac.pop("device", None)
    for key in ("actor_init_checkpoint_path", "reference_guidance_dp_solution_path"):
        if isinstance(sac.get(key), str):
            sac[key] = re.sub(r"(?i)seed[_=\-]?\d+", "seed{n}", sac[key].replace("\\", "/"))
            sac[key] = re.sub(r"^[A-Za-z]:/.*?/runs/", "runs/", sac[key])
    telemetry = payload.get("telemetry", {}) or {}
    telemetry.pop("run_root", None)
    telemetry.pop("overwrite", None)
    return payload


def signature(config: Mapping[str, Any]) -> str:
    payload = normalize_signature_payload(config)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def family_key(run_rel: str) -> str:
    value = run_rel.replace("\\", "/").rstrip("/")
    value = re.sub(r"(?i)/seed\d+$", "", value)
    value = re.sub(r"(?i)seed[_=\-]?\d+", "seed{n}", value)
    return value


def feature_tags(
    run_rel: str,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    distillation: Mapping[str, Any],
) -> list[str]:
    text = run_rel.lower()
    sac = config.get("sac", {}) or {}
    tags: list[str] = []
    if bool(sac.get("simba_backbone")) or "simba" in text:
        tags.append("SimbaV2")
    n_step = as_int(sac.get("sacn_n_step"), 1) or 1
    if n_step > 1 or "sacn" in text:
        mode = str(sac.get("sacn_importance_mode", "")).lower()
        target = str(sac.get("sacn_target_mode", "")).lower()
        tags.append("FastSACN" if mode == "none" or "fastsacn" in text else "SACn")
        tags.append(f"n={n_step}")
        if target:
            tags.append(f"target={target}")
    updates = as_int(sac.get("updates_per_step"), 1) or 1
    tags.append(f"UTD={updates}")
    if (as_int(sac.get("redq_num_critics"), 2) or 2) > 2:
        tags.append("REDQ")
    if positive(sac, "cql_alpha"):
        tags.append("CQL")
    if positive(sac, "pendulum_actor_symmetry_weight") or bool(
        sac.get("pendulum_symmetry_augmentation")
    ):
        tags.append("actor_symmetry")
    if positive(sac, "pendulum_critic_symmetry_weight"):
        tags.append("critic_symmetry")
    if positive(sac, "uniform_exploration_initial_probability"):
        tags.append("uniform_exploration")
    if (as_float(sac.get("alpha_min_value"), 0.0) or 0.0) > 0:
        tags.append("alpha_floor")
    if summary.get("dagger_collection") is not None or distillation:
        tags.append("DAgger_or_distillation")
    if positive(sac, "sac_actor_loss_weight"):
        tags.append("joint_SAC_BC_loss")
    if positive(sac, "critic_search_actor_weight"):
        tags.append("critic_search_actor_loss")
    if (as_int((config.get("env") or {}).get("pendulum_failure_curriculum_candidate_count"), 0) or 0) > 0:
        tags.append("automatic_failure_curriculum")
    return list(dict.fromkeys(tags))


def discover_runs(
    repos: Sequence[tuple[str, Path]],
    checkpoint_alias_lineage: Mapping[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    checkpoint_alias_lineage = checkpoint_alias_lineage or {}
    physical: list[dict[str, Any]] = []
    primary_by_rel: dict[str, dict[str, Any]] = {}
    for repo_id, repo in repos:
        run_root = repo / "runs"
        run_dirs: set[Path] = set()
        for name in (
            "config.json",
            "summary.json",
            "distillation_summary.json",
            "training_summary.json",
            "run_manifest.json",
            "events.jsonl",
            "metrics.csv",
            "eval_episodes.csv",
        ):
            run_dirs.update(path.parent for path in run_root.rglob(name))
        run_dirs.update(
            path.parent.parent
            for path in run_root.rglob("*.pt")
            if path.parent.name.lower() == "checkpoints"
        )
        for run_dir in sorted(run_dirs):
            rel = repo_relative(repo, run_dir)
            is_checkpoint_alias = "/checkpoint_eval_aliases/" in (
                "/" + rel.replace("\\", "/")
            )
            config = safe_json(run_dir / "config.json")
            summary = safe_json(run_dir / "summary.json")
            distillation = safe_json(run_dir / "distillation_summary.json")
            training_summary = safe_json(run_dir / "training_summary.json")
            run_manifest = safe_json(run_dir / "run_manifest.json")
            seed = as_int(config.get("seed"))
            if seed is None:
                match = SEED_RE.search(rel.replace("\\", "/") + "/")
                seed = int(match.group(1)) if match else None
            sac = config.get("sac", {}) or {}
            configured_steps = as_int(sac.get("total_steps"))
            if is_checkpoint_alias:
                executed_steps = 0
                metadata_kind = "checkpoint_or_swa_alias"
            elif distillation:
                executed_steps = as_int(distillation.get("training_environment_steps"))
                if executed_steps is None:
                    initial = as_int(
                        (distillation.get("initial_collection_metrics") or {}).get("samples"),
                        0,
                    ) or 0
                    executed_steps = initial + sum(
                        as_int(row.get("samples"), 0) or 0
                        for row in distillation.get("dagger_collection_metrics", []) or []
                    )
                metadata_kind = "distillation_or_dagger"
            elif summary.get("dagger_collection") is not None:
                executed_steps = sum(
                    as_int(row.get("samples"), 0) or 0
                    for row in summary.get("dagger_collection", []) or []
                )
                metadata_kind = "corrected_dagger_or_hybrid"
            elif summary.get("collections") is not None:
                executed_steps = sum(
                    as_int(row.get("samples"), 0) or 0
                    for row in summary.get("collections", []) or []
                )
                metadata_kind = "custom_dagger_or_adapter"
            elif training_summary or run_manifest:
                manifest_budget = (
                    ((run_manifest.get("spec") or {}).get("budget") or {})
                    if run_manifest
                    else {}
                )
                executed_steps = (
                    as_int(training_summary.get("cumulative_environment_steps"))
                    or as_int(manifest_budget.get("cumulative_environment_steps"))
                    or as_int(manifest_budget.get("strict_all_simulator_use"))
                    or as_int(manifest_budget.get("lineage_cap"))
                )
                metadata_kind = "custom_training_manifest"
            elif not config and not summary and not distillation:
                executed_steps = None
                metadata_kind = "artifact_only_unknown"
            else:
                executed_steps = configured_steps
                metadata_kind = "online_rl"

            events = event_progress(run_dir / "events.jsonl")
            metrics_integrity = metrics_file_integrity(run_dir / "metrics.csv")
            local_eval = eval_episode_summary(run_dir / "eval_episodes.csv")
            checkpoint_candidates = list((run_dir / "checkpoints").glob("*.pt"))
            checkpoint_candidates += list(run_dir.glob("*.pt"))
            declared_checkpoint = (
                ((run_manifest.get("artifacts") or {}).get("checkpoint"))
                if run_manifest
                else None
            )
            if declared_checkpoint:
                declared_checkpoint_path = run_dir / str(declared_checkpoint)
                if (
                    declared_checkpoint_path.exists()
                    and declared_checkpoint_path not in checkpoint_candidates
                ):
                    checkpoint_candidates.append(declared_checkpoint_path)
            final_checkpoint = run_dir / "checkpoints" / "final.pt"
            final_model_exists = final_checkpoint.exists() or bool(
                [
                    path
                    for path in checkpoint_candidates
                    if path.name.lower()
                    in {
                        "final.pt",
                        "actor.pt",
                        "final_actor.pt",
                        "time_conditioned_actor.pt",
                    }
                ]
            )
            intermediate_checkpoints = checkpoint_candidates
            progress_candidates = [
                as_int(events.get("max_event_step")),
                as_int(events.get("run_complete_step")),
                as_int(local_eval.get("local_eval_step")),
            ]
            progress = max((value for value in progress_candidates if value is not None), default=None)
            inherited_checkpoint_step = (
                checkpoint_alias_lineage.get(canonical_run_rel(rel))
                or step_hint(rel)
                if is_checkpoint_alias
                else None
            )
            if is_checkpoint_alias:
                actual_steps = inherited_checkpoint_step
            elif metadata_kind != "online_rl" and executed_steps is not None:
                actual_steps = executed_steps
            else:
                actual_steps = progress
                if actual_steps is None and final_model_exists:
                    actual_steps = configured_steps
            if executed_steps is not None and actual_steps is None:
                actual_steps = executed_steps
            if is_checkpoint_alias:
                observed_learning_steps = 0
            elif metadata_kind == "online_rl":
                observed_learning_steps = actual_steps
            else:
                observed_learning_steps = executed_steps
            if is_checkpoint_alias:
                target_for_completion = 0
                completed = final_model_exists
            elif metadata_kind == "artifact_only_unknown":
                target_for_completion = None
                completed = bool(
                    final_model_exists
                    and events.get("run_complete_step") is not None
                )
            else:
                target_for_completion = (
                    executed_steps if metadata_kind != "online_rl" else configured_steps
                )
                completed = bool(
                    final_model_exists
                    and target_for_completion is not None
                    and actual_steps is not None
                    and actual_steps >= target_for_completion
                )

            manual_evidence = manual_region_evidence(config, summary, distillation)
            manual_evidence.extend(
                manual_region_evidence({}, training_summary, run_manifest)
            )
            if metadata_kind in {
                "distillation_or_dagger",
                "corrected_dagger_or_hybrid",
                "custom_dagger_or_adapter",
                "custom_training_manifest",
            } and any(
                token in rel.lower()
                for token in (
                    "near_down",
                    "neardown",
                    "targeted_failuremix",
                    "failuremix",
                    "hard120",
                    "hard200k",
                    "hardteacher",
                    "hard_window",
                    "tight_dpdemonstrations",
                )
            ):
                manual_evidence.append(
                    "supervised/DAgger run name declares a manually fixed training-state region"
                )
            manual_evidence = list(dict.fromkeys(manual_evidence))
            automatic_evidence = automatic_selection_evidence(config, summary)
            reference_evidence = reference_training_evidence(config, summary, distillation)
            custom_text = json.dumps(
                {"training_summary": training_summary, "run_manifest": run_manifest}
            ).lower()
            if (
                '"training_labels_only": true' in custom_text
                or "reference_action" in custom_text
                or "reference_targets" in custom_text
                or "dagger" in custom_text
            ):
                reference_evidence.append(
                    "custom manifest/summary declares reference-labelled training"
                )
            reference_evidence = list(dict.fromkeys(reference_evidence))
            dependencies = dependency_run_rels(
                config,
                summary,
                distillation,
                training_summary,
                run_manifest,
            )
            uses_rl_component = metadata_kind in {
                "online_rl",
                "checkpoint_or_swa_alias",
            } or bool(dependencies) or any(
                (
                    summary.get("uses_fixed_pure_rl_critics_during_training"),
                    summary.get("uses_fixed_pure_rl_critic_at_inference"),
                    summary.get("rl_critic_source_run"),
                    positive(sac, "sac_actor_loss_weight"),
                    positive(sac, "critic_search_actor_weight"),
                )
            )
            if metadata_kind == "artifact_only_unknown":
                category = "unknown"
            elif metadata_kind == "custom_training_manifest" and reference_evidence:
                category = "RL + supervised"
            elif reference_evidence and uses_rl_component:
                category = "RL + supervised"
            elif reference_evidence:
                category = "supervised only"
            else:
                category = "pure RL"
            cheat_status = "cheating" if manual_evidence else "clean"
            optimizer_updates = (
                as_int(summary.get("actor_optimizer_steps"))
                or as_int(distillation.get("actor_optimizer_steps"))
                or as_int(events.get("run_complete_updates"))
            )
            final_eval = summary.get("final_eval") or distillation.get("final_eval") or {}
            if final_eval:
                local_eval = {
                    **local_eval,
                    "local_eval_episodes": as_int(final_eval.get("num_eval_episodes")),
                    "local_eval_mean_return": as_float(final_eval.get("mean_return")),
                    "local_eval_task_rate": as_float(final_eval.get("task_success_rate")),
                    "local_eval_strict_rate": as_float(final_eval.get("strict_success_rate")),
                    "local_eval_collapse_rate": as_float(final_eval.get("collapse_rate")),
                }
            row = {
                "repo_id": repo_id,
                "run_path": rel,
                "canonical_run_path": canonical_run_rel(rel),
                "family_key": family_key(canonical_run_rel(rel)),
                "seed": seed if seed is not None else "",
                "metadata_kind": metadata_kind,
                "config_signature": signature(config)
                if config
                else signature(
                    (run_manifest.get("spec") or {})
                    if run_manifest
                    else training_summary
                ),
                "metadata_files": pipe(
                    name
                    for name, present in (
                        ("config.json", bool(config)),
                        ("summary.json", bool(summary)),
                        ("distillation_summary.json", bool(distillation)),
                        ("training_summary.json", bool(training_summary)),
                        ("run_manifest.json", bool(run_manifest)),
                    )
                    if present
                ),
                "configured_total_steps": configured_steps if configured_steps is not None else "",
                "direct_executed_learning_steps": observed_learning_steps
                if observed_learning_steps is not None
                else "",
                "inherited_checkpoint_step": inherited_checkpoint_step
                if inherited_checkpoint_step is not None
                else "",
                "actual_progress_steps": actual_steps if actual_steps is not None else "",
                "optimizer_update_iterations": optimizer_updates
                if optimizer_updates is not None
                else "",
                "run_complete_event": events.get("run_complete_step") is not None,
                "events_parse_error": bool(events.get("event_parse_error")),
                "metrics_file_integrity": metrics_integrity,
                "final_checkpoint": final_model_exists,
                "checkpoint_count": len(intermediate_checkpoints),
                "completed": completed,
                "execution_status": (
                    "derived_checkpoint_alias"
                    if is_checkpoint_alias
                    else "complete"
                    if completed
                    else "artifact_only_unresolved"
                    if metadata_kind == "artifact_only_unknown"
                    else "partial_with_checkpoint"
                    if final_model_exists
                    else "incomplete_or_running"
                ),
                "category_direct": category,
                "reference_training": bool(reference_evidence),
                "reference_training_evidence": pipe(reference_evidence),
                "manual_state_region_training": bool(manual_evidence),
                "manual_state_region_evidence": pipe(manual_evidence),
                "automatic_performance_selection": bool(automatic_evidence),
                "automatic_selection_evidence": pipe(automatic_evidence),
                "cheating_status_direct": cheat_status,
                "dependencies": pipe(dependencies),
                "feature_tags": pipe(feature_tags(rel, config, summary, distillation)),
                **local_eval,
            }
            physical.append(row)
            if repo_id == repos[0][0]:
                primary_by_rel[row["canonical_run_path"]] = row
    # Propagate provenance through actor initializers, continuations, frozen
    # backbones, and other resolved run dependencies. This closes alternative
    # custom schemas such as `frozen_simbav2_source_run`.
    for _iteration in range(12):
        changed = False
        for row in primary_by_rel.values():
            dependency_rows = [
                primary_by_rel[dependency]
                for dependency in str(row.get("dependencies", "")).split(" | ")
                if dependency in primary_by_rel
            ]
            dependency_categories = {
                str(dependency["category_direct"])
                for dependency in dependency_rows
            }
            category = str(row["category_direct"])
            if "RL + supervised" in dependency_categories:
                propagated_category = "RL + supervised"
            elif (
                category == "supervised only"
                and "pure RL" in dependency_categories
            ):
                propagated_category = "RL + supervised"
            else:
                propagated_category = category
            inherited_manual = sorted(
                {
                    str(dependency["manual_state_region_evidence"])
                    for dependency in dependency_rows
                    if dependency.get("manual_state_region_evidence")
                }
            )
            if inherited_manual:
                direct_evidence = [
                    value
                    for value in str(
                        row.get("manual_state_region_evidence", "")
                    ).split(" | ")
                    if value
                ]
                propagated_evidence = pipe(
                    list(dict.fromkeys(direct_evidence + inherited_manual))
                )
            else:
                propagated_evidence = str(
                    row.get("manual_state_region_evidence", "")
                )
            if (
                propagated_category != row["category_direct"]
                or propagated_evidence
                != row.get("manual_state_region_evidence", "")
            ):
                row["category_direct"] = propagated_category
                row["manual_state_region_evidence"] = propagated_evidence
                row["manual_state_region_training"] = bool(
                    propagated_evidence
                )
                row["cheating_status_direct"] = (
                    "cheating" if propagated_evidence else "clean"
                )
                changed = True
        if not changed:
            break
    lineage_cache: dict[str, int | None] = {}

    def estimated_lineage(run_path: str, active: set[str]) -> int | None:
        if run_path in lineage_cache:
            return lineage_cache[run_path]
        if run_path in active or run_path not in primary_by_rel:
            return None
        row = primary_by_rel[run_path]
        own = as_int(row.get("direct_executed_learning_steps"), 0) or 0
        dependency_values = [
            value
            for dependency in str(row.get("dependencies", "")).split(" | ")
            if dependency
            for value in [
                estimated_lineage(dependency, active | {run_path})
            ]
            if value is not None
        ]
        value = own + sum(dependency_values)
        lineage_cache[run_path] = value if value > 0 else None
        return lineage_cache[run_path]

    for run_path, row in primary_by_rel.items():
        estimate = estimated_lineage(run_path, set())
        row["recursive_lineage_step_estimate"] = (
            estimate if estimate is not None else ""
        )
    return physical, primary_by_rel


def checkpoint_step_from_name(path: Path) -> int | None:
    match = re.search(r"(?i)step[_\-]?(\d+)", path.name)
    if match:
        return int(match.group(1))
    return step_hint(path.as_posix())


def build_checkpoint_alias_audit(
    primary: Path, hash_cache: HashCache
) -> list[dict[str, Any]]:
    alias_root = primary / "runs" / "checkpoint_eval_aliases"
    if not alias_root.exists():
        return []
    alias_files = sorted(alias_root.rglob("final.pt"))
    alias_hashes: dict[str, list[Path]] = defaultdict(list)
    sizes: set[int] = set()
    for path in alias_files:
        alias_hashes[hash_cache.sha256(path)].append(path)
        sizes.add(path.stat().st_size)

    source_by_hash: dict[str, list[Path]] = defaultdict(list)
    for path in (primary / "runs").rglob("*.pt"):
        if alias_root in path.parents or path.stat().st_size not in sizes:
            continue
        digest = hash_cache.sha256(path)
        if digest in alias_hashes:
            source_by_hash[digest].append(path)

    rows: list[dict[str, Any]] = []
    for path in alias_files:
        digest = hash_cache.sha256(path)
        run_dir = path.parent.parent
        sources = source_by_hash.get(digest, [])
        rel = repo_relative(primary, run_dir)
        if sources:
            inherited = max(
                (
                    checkpoint_step_from_name(source)
                    or step_hint(source.as_posix())
                    or 0
                )
                for source in sources
            )
            alias_type = "byte_exact_checkpoint_copy"
            provenance = "exact source checkpoint recovered by SHA-256"
            source_text = pipe(repo_relative(primary, source) for source in sources)
        elif "swa_30_40_50" in rel.lower():
            inherited = 50_000
            alias_type = "synthesized_actor_swa"
            seed_match = SEED_RE.search(rel + "/")
            seed = int(seed_match.group(1)) if seed_match else 1
            inferred_sources = [
                primary
                / "runs"
                / "simbav2_fastsacn8_lam05_utd2_50k_20260704"
                / f"seed{seed}"
                / "checkpoints"
                / f"step_{step}.pt"
                for step in (30_000, 40_000, 50_000)
            ]
            source_text = pipe(
                repo_relative(primary, source)
                for source in inferred_sources
                if source.exists()
            )
            provenance = "SWA source set inferred from alias name; creation manifest missing"
        else:
            inherited = step_hint(rel)
            alias_type = "unresolved_checkpoint_transform"
            source_text = ""
            provenance = "no byte-identical source or creation manifest found"
        config = safe_json(run_dir / "config.json")
        rows.append(
            {
                "alias_run_path": rel,
                "alias_type": alias_type,
                "sha256": digest,
                "bytes": path.stat().st_size,
                "copied_config_total_steps": as_int(
                    (config.get("sac", {}) or {}).get("total_steps")
                ),
                "new_environment_steps": 0,
                "inherited_checkpoint_lineage_step": inherited
                if inherited is not None
                else "",
                "source_checkpoints": source_text,
                "provenance_status": provenance,
            }
        )
    return rows


def _evaluation_protocol(payload: Mapping[str, Any]) -> tuple[tuple[int, ...], int, int, float]:
    if "result" in payload:
        protocol = payload.get("protocol", {}) or {}
        seeds = tuple(as_int(seed, 0) or 0 for seed in protocol.get("actor_seeds", []))
        cells = as_int(protocol.get("initial_condition_cells"), 0) or 0
        epsilon = as_float(protocol.get("epsilon_return"), EXPECTED_EPSILON) or EXPECTED_EPSILON
        near = (payload.get("result", {}) or {}).get("near_best_known_return_eps", {}) or {}
        trials = as_int(near.get("trials"), 0) or 0
    else:
        seeds = tuple(as_int(seed, 0) or 0 for seed in payload.get("actual_seeds", []))
        cells = as_int(payload.get("num_initial_condition_cells"), 0) or 0
        epsilon = as_float(payload.get("epsilon_return"), EXPECTED_EPSILON) or EXPECTED_EPSILON
        near = (
            (payload.get("criteria", {}) or {}).get("near_best_known_return_eps", {})
            or {}
        )
        trials = as_int(near.get("total"), 0) or 0
    return seeds, cells, trials, epsilon


def _resolve_artifact(repo: Path, raw: Path | str | None) -> Path | None:
    if raw is None or str(raw).strip() == "":
        return None
    path = Path(str(raw).replace("\\", "/"))
    if not path.is_absolute():
        path = repo / path
    return path.resolve()


def step_hint(text: str) -> int | None:
    values = [int(match.group(1)) * 1000 for match in STEP_HINT_RE.finditer(text)]
    plausible = [value for value in values if 1_000 <= value <= 5_000_000]
    return max(plausible) if plausible else None


def explicit_checkpoint_step_hint(text: str) -> int | None:
    values = [
        int(match.group(1))
        for match in re.finditer(
            r"(?i)(?:^|[/_\-])step[_\-]?(\d+)(?=$|[/_\-])", text
        )
    ]
    return max(values) if values else None


def discover_evaluation_aliases(
    repos: Sequence[tuple[str, Path]],
    hash_cache: HashCache,
    checkpoint_alias_lineage: Mapping[str, int],
    primary_run_map: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repo_id, repo in repos:
        resolver = five_seed.LineageResolver(repo)
        paths = sorted((repo / "reports").rglob("relative_summary.json"))
        paths += sorted((repo / "reports").rglob("authoritative_summary.json"))
        for summary_path in paths:
            payload = safe_json(summary_path)
            metrics = five_seed._metrics_from_summary(payload)
            if metrics is None:
                continue
            near, task, strict, trials_from_metrics = metrics
            seeds, cells, trials, epsilon = _evaluation_protocol(payload)
            trials = trials or trials_from_metrics
            rollout, relative = five_seed._artifact_paths(repo, summary_path, payload)
            stale_rollout_recovered = False
            if rollout is None or not rollout.exists():
                stable_sibling = (
                    summary_path.parent.parent
                    / "grid"
                    / "pendulum_grid_rollouts.csv"
                )
                if stable_sibling.exists():
                    rollout = stable_sibling.resolve()
                    stale_rollout_recovered = True
            run_dirs = five_seed._run_dirs_from_artifact(repo, payload, rollout)
            artifact = five_seed.EvaluationArtifact(
                summary_path=summary_path,
                payload=payload,
                method=five_seed._artifact_method(summary_path, payload),
                near_successes=near,
                task_successes=task,
                strict_successes=strict,
                trials=trials,
                mean_return=five_seed._mean_return(payload, rollout),
                actual_seeds=seeds,
                cells=cells,
                epsilon_return=epsilon,
                rollout_path=rollout,
                relative_rollout_path=relative,
                run_dirs=run_dirs,
                content_hash="",
            )
            try:
                classified = five_seed.classify_artifact(repo, artifact, resolver)
            except Exception as error:
                classified = {
                    "category": "unknown",
                    "executed_learning_transition_steps": None,
                    "selected_checkpoint_learning_transition_steps": None,
                    "automatic_discovery_rollout_steps": None,
                    "reference_oracle_label_calls": None,
                    "optimizer_update_iterations": None,
                    "manual_training_region": False,
                    "hardcoded_inference_router": False,
                    "inference_model_mixture": False,
                    "reference_at_inference": False,
                    "deployment_evidence": "",
                    "selection_protocol": "unknown",
                    "dependency_nodes": "",
                    "lineage_evidence": f"classification error: {error}",
                }
            identity_path = None
            for candidate in (relative, rollout):
                if candidate is not None and candidate.exists():
                    identity_path = candidate
                    break
            if identity_path is not None:
                content_identity = "rows:" + hash_cache.sha256(identity_path)
            else:
                content_identity = "summary:" + hash_cache.sha256(summary_path)
                # Two summaries in the historical worktree omit their local
                # row table even though the byte-identical summary and row
                # table both survive in the primary worktree. Resolve that
                # provenance alias before deduplication.
                if repo_id != repos[0][0]:
                    primary_summary = (
                        repos[0][1] / repo_relative(repo, summary_path)
                    )
                    if (
                        primary_summary.exists()
                        and hash_cache.sha256(primary_summary)
                        == hash_cache.sha256(summary_path)
                    ):
                        primary_payload = safe_json(primary_summary)
                        primary_rollout, primary_relative = (
                            five_seed._artifact_paths(
                                repos[0][1],
                                primary_summary,
                                primary_payload,
                            )
                        )
                        for candidate in (primary_relative, primary_rollout):
                            if candidate is not None and candidate.exists():
                                content_identity = (
                                    "rows:" + hash_cache.sha256(candidate)
                                )
                                break
            manual = bool(classified.get("manual_training_region"))
            hard_router = bool(classified.get("hardcoded_inference_router"))
            reference_inference = bool(classified.get("reference_at_inference"))
            mixture = bool(classified.get("inference_model_mixture"))
            cheat_reasons = []
            if manual:
                cheat_reasons.append("manual fixed state region used during training")
            if hard_router:
                cheat_reasons.append("hardcoded state-range router at inference")
            if reference_inference:
                cheat_reasons.append("reference policy queried at inference")
            lineage_known = classified.get("executed_learning_transition_steps") is not None
            if cheat_reasons:
                cheat_status = "cheating"
            elif not lineage_known and not run_dirs:
                cheat_status = "unknown"
            else:
                cheat_status = "clean"
            method = artifact.method
            inferred_steps = step_hint(method + " " + repo_relative(repo, summary_path))
            selected_step_hint = explicit_checkpoint_step_hint(
                repo_relative(repo, summary_path)
            )
            if selected_step_hint is not None:
                classified[
                    "selected_checkpoint_learning_transition_steps"
                ] = selected_step_hint
            run_rels = [canonical_run_rel(path) for path in run_dirs]
            direct_rows = [
                primary_run_map[run_rel]
                for run_rel in run_rels
                if run_rel in primary_run_map
            ]
            critic_run_rel = canonical_run_rel(
                (payload.get("protocol", {}) or {}).get("critic_run")
            )
            if (
                critic_run_rel in primary_run_map
                and primary_run_map[critic_run_rel] not in direct_rows
            ):
                direct_rows.append(primary_run_map[critic_run_rel])
            direct_manual_evidence = sorted(
                {
                    str(direct["manual_state_region_evidence"])
                    for direct in direct_rows
                    if direct.get("manual_state_region_evidence")
                }
            )
            if direct_manual_evidence:
                manual = True
                cheat_reasons.extend(direct_manual_evidence)
                cheat_status = "cheating"
            supplemental_summary: dict[str, Any] = {}
            reports_root = (repo / "reports").resolve()
            for ancestor in summary_path.parents:
                if ancestor.resolve() == reports_root:
                    break
                candidate = ancestor / "summary.json"
                if candidate.exists():
                    supplemental_summary = safe_json(candidate)
                    if supplemental_summary:
                        break
            hard120_selected = any(
                "selected_hard120_specialist" in pointer.lower()
                and (as_int(value, 0) or 0) > 0
                for pointer, value in _walk_items(supplemental_summary)
            )
            specialist_selection_keys = {
                pointer
                for pointer, value in _walk_items(supplemental_summary)
                if "selected_" in pointer.lower()
                and "specialist" in pointer.lower()
                and (as_int(value, 0) or 0) > 0
            }
            if hard120_selected:
                manual = True
                cheat_reasons.append(
                    "deployed specialist mixture selects a hard120 actor trained on a fixed angle band"
                )
                cheat_status = "cheating"
            if len(specialist_selection_keys) > 1:
                mixture = True
                classified["inference_model_mixture"] = True
                classified["deployment_evidence"] = pipe(
                    [
                        classified.get("deployment_evidence", ""),
                        "artifact-local summary selects multiple specialist actors",
                    ]
                )
            direct_categories = {
                str(direct.get("category_direct", "unknown"))
                for direct in direct_rows
            }
            if "RL + supervised" in direct_categories:
                classified["category"] = "RL + supervised"
            elif (
                "supervised only" in direct_categories
                and "pure RL" not in direct_categories
            ):
                classified["category"] = "supervised only"
            elif (
                "supervised only" in direct_categories
                and "pure RL" in direct_categories
            ):
                classified["category"] = "RL + supervised"
            custom_lineage_values = [
                as_int(direct.get("recursive_lineage_step_estimate"))
                for direct in direct_rows
                if direct.get("metadata_kind")
                in {
                    "custom_training_manifest",
                    "custom_dagger_or_adapter",
                }
            ]
            custom_lineage_values = [
                value
                for value in custom_lineage_values
                if value is not None
            ]
            if custom_lineage_values:
                custom_lineage = max(custom_lineage_values)
                classified["executed_learning_transition_steps"] = (
                    custom_lineage
                )
                classified[
                    "selected_checkpoint_learning_transition_steps"
                ] = custom_lineage
                classified["lineage_evidence"] = pipe(
                    [
                        classified.get("lineage_evidence", ""),
                        "custom run-manifest/collections lineage override",
                    ]
                )
            lineage_known = (
                classified.get("executed_learning_transition_steps")
                is not None
            )
            if (
                not cheat_reasons
                and not lineage_known
                and not direct_rows
            ):
                cheat_status = "unknown"
            alias_steps = [
                checkpoint_alias_lineage[run_rel]
                for run_rel in run_rels
                if run_rel in checkpoint_alias_lineage
            ]
            alias_lineage_override = bool(alias_steps)
            if alias_steps:
                critic_raw = (payload.get("protocol", {}) or {}).get("critic_run")
                critic_path = _resolve_artifact(repo, critic_raw)
                critic_steps = None
                if critic_path is not None and critic_path.exists():
                    try:
                        critic_steps = resolver.pipeline(
                            [critic_path]
                        ).executed_learning_transitions
                    except Exception:
                        critic_steps = None
                inherited = max(alias_steps)
                # These aliases are actor views of an existing SAC checkpoint.
                # A critic from the same reward lineage is a union, not another
                # execution. Use max rather than the copied-config sum.
                if critic_steps is not None:
                    inherited = max(inherited, critic_steps)
                classified["executed_learning_transition_steps"] = inherited
                classified["selected_checkpoint_learning_transition_steps"] = inherited
                previous = str(classified.get("lineage_evidence", ""))
                classified["lineage_evidence"] = pipe(
                    [
                        previous,
                        "checkpoint_eval_alias lineage corrected from copied config to source checkpoint step",
                    ]
                )
            row = {
                "repo_id": repo_id,
                "summary_path": repo_relative(repo, summary_path),
                "summary_kind": summary_path.name,
                "method": method,
                "content_identity": content_identity,
                "seed_ids": pipe(seeds),
                "seed_count": len(seeds),
                "less_than_5_seeds": len(seeds) < 5,
                "cells_per_seed": cells,
                "trials": trials,
                "epsilon_return": epsilon,
                "standard_grid_protocol": bool(
                    cells == EXPECTED_CELLS
                    and trials == EXPECTED_CELLS * len(seeds)
                    and math.isclose(epsilon, EXPECTED_EPSILON)
                ),
                "near_successes": near,
                "near_rate": near / trials if trials else np.nan,
                "task_successes": task,
                "task_rate": task / trials if trials else np.nan,
                "strict_successes": strict,
                "strict_rate": strict / trials if trials else np.nan,
                "mean_return": artifact.mean_return,
                "category": classified.get("category", "unknown"),
                "executed_learning_transition_steps": classified.get(
                    "executed_learning_transition_steps"
                ),
                "selected_checkpoint_learning_transition_steps": classified.get(
                    "selected_checkpoint_learning_transition_steps"
                ),
                "automatic_discovery_rollout_steps": classified.get(
                    "automatic_discovery_rollout_steps"
                ),
                "reference_oracle_label_calls": classified.get(
                    "reference_oracle_label_calls"
                ),
                "optimizer_update_iterations": classified.get(
                    "optimizer_update_iterations"
                ),
                "step_hint_from_name": inferred_steps if inferred_steps is not None else "",
                "explicit_selected_checkpoint_step_hint": selected_step_hint
                if selected_step_hint is not None
                else "",
                "checkpoint_alias_lineage_override": alias_lineage_override,
                "manual_state_region_training": manual,
                "hardcoded_inference_router": hard_router,
                "reference_at_inference": reference_inference,
                "inference_model_mixture": mixture,
                "cheating_status": cheat_status,
                "cheating_evidence": pipe(cheat_reasons),
                "deployment_evidence": classified.get("deployment_evidence", ""),
                "selection_protocol": classified.get("selection_protocol", ""),
                "run_paths": pipe(run_rels),
                "dependency_nodes": classified.get("dependency_nodes", ""),
                "lineage_evidence": classified.get("lineage_evidence", ""),
                "source_rollouts": repo_relative(repo, rollout)
                if rollout is not None
                else "",
                "source_rollouts_exist": bool(rollout is not None and rollout.exists()),
                "stale_rollout_pointer_recovered_from_stable_sibling": stale_rollout_recovered,
                "relative_rollouts": repo_relative(repo, relative)
                if relative is not None
                else "",
                "relative_rollouts_exist": bool(relative is not None and relative.exists()),
            }
            rows.append(row)
    return rows


def choose_evaluation_representative(
    aliases: Sequence[dict[str, Any]],
    primary_id: str,
    canonical_identity: str,
) -> dict[str, Any]:
    def preference(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
        return (
            int(row["repo_id"] == primary_id),
            int(row["summary_kind"] == "authoritative_summary.json"),
            int(row["seed_count"]),
            -len(str(row["summary_path"])),
        )

    chosen = dict(max(aliases, key=preference))
    chosen["unique_evaluation_id"] = hashlib.sha256(
        canonical_identity.encode()
    ).hexdigest()[:16]
    chosen["canonical_result_identity"] = canonical_identity
    chosen["physical_outcome_table_identity"] = chosen["content_identity"]
    chosen["alias_count"] = len(aliases)
    chosen["alias_locations"] = pipe(
        f"{row['repo_id']}:{row['summary_path']}" for row in aliases
    )
    chosen["present_in_both_repos"] = len({row["repo_id"] for row in aliases}) > 1
    chosen["alias_method_values"] = pipe(
        sorted({str(row["method"]) for row in aliases})
    )
    chosen["alias_category_values"] = pipe(
        sorted({str(row["category"]) for row in aliases})
    )
    chosen["alias_cheating_status_values"] = pipe(
        sorted({str(row["cheating_status"]) for row in aliases})
    )
    chosen["alias_classification_consistent"] = (
        len({str(row["category"]) for row in aliases}) == 1
        and len({str(row["cheating_status"]) for row in aliases}) == 1
    )
    return chosen


def deduplicate_evaluations(
    aliases: list[dict[str, Any]], primary_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outcome_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aliases:
        outcome_groups[str(row["content_identity"])].append(row)
    groups = outcome_groups
    unique: list[dict[str, Any]] = []
    alias_rows: list[dict[str, Any]] = []
    for canonical_identity, group in groups.items():
        chosen = choose_evaluation_representative(
            group, primary_id, canonical_identity
        )
        unique.append(chosen)
        for row in group:
            alias_rows.append(
                {
                    **row,
                    "unique_evaluation_id": chosen["unique_evaluation_id"],
                    "canonical_result_identity": canonical_identity,
                    "physical_outcome_table_identity": row["content_identity"],
                    "is_representative": (
                        row["repo_id"] == chosen["repo_id"]
                        and row["summary_path"] == chosen["summary_path"]
                    ),
                }
            )
    unique.sort(
        key=lambda row: (
            int(row["seed_count"]),
            float(row["near_rate"]),
            float(row["task_rate"]),
            float(row["strict_rate"]),
        ),
        reverse=True,
    )
    alias_rows.sort(key=lambda row: (row["unique_evaluation_id"], row["repo_id"], row["summary_path"]))
    return unique, alias_rows


def aggregate_run_families(
    physical_runs: Sequence[dict[str, Any]],
    unique_evaluations: Sequence[dict[str, Any]],
    primary_id: str,
) -> list[dict[str, Any]]:
    canonical_runs = [row for row in physical_runs if row["repo_id"] == primary_id]
    eval_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for evaluation in unique_evaluations:
        for run_rel in str(evaluation.get("run_paths", "")).split(" | "):
            if run_rel:
                eval_by_run[run_rel].append(evaluation)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in canonical_runs:
        groups[str(row["family_key"])].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        configured_seeds = sorted(
            {int(row["seed"]) for row in rows if row["seed"] not in ("", None)}
        )
        completed_seeds = sorted(
            {
                int(row["seed"])
                for row in rows
                if row["completed"] and row["seed"] not in ("", None)
            }
        )
        evaluations = {
            evaluation["unique_evaluation_id"]: evaluation
            for row in rows
            for evaluation in eval_by_run.get(str(row["canonical_run_path"]), [])
        }
        best = max(
            evaluations.values(),
            key=lambda item: (
                item["seed_count"],
                item["near_rate"],
                item["task_rate"],
                item["strict_rate"],
            ),
            default=None,
        )
        categories = sorted({str(row["category_direct"]) for row in rows})
        cheating = any(row["cheating_status_direct"] == "cheating" for row in rows)
        planned_five = bool(
            re.search(r"(?i)(5seed|n5|five[_\-]?seed)", key)
            or set(configured_seeds) >= {0, 1, 2, 3, 4}
        )
        local_candidates = [
            row
            for row in rows
            if row.get("local_eval_task_rate") not in (None, "")
        ]
        best_local = max(
            local_candidates,
            key=lambda row: (
                float(row.get("local_eval_task_rate") or -1),
                float(row.get("local_eval_mean_return") or -math.inf),
            ),
            default={},
        )
        output.append(
            {
                "family_key": key,
                "run_instance_count": len(rows),
                "configured_seed_ids": pipe(configured_seeds),
                "configured_seed_count": len(configured_seeds),
                "completed_seed_ids": pipe(completed_seeds),
                "completed_seed_count": len(completed_seeds),
                "less_than_5_completed_seeds": len(completed_seeds) < 5,
                "planned_five_seed_family": planned_five,
                "planned_five_seed_incomplete": planned_five
                and len(completed_seeds) < 5,
                "categories": pipe(categories),
                "cheating_status": "cheating" if cheating else "clean",
                "manual_state_region_evidence": pipe(
                    sorted(
                        {
                            row["manual_state_region_evidence"]
                            for row in rows
                            if row["manual_state_region_evidence"]
                        }
                    )
                ),
                "target_step_values": pipe(
                    sorted(
                        {
                            int(row["direct_executed_learning_steps"])
                            for row in rows
                            if row["direct_executed_learning_steps"] not in ("", None)
                        }
                    )
                ),
                "actual_progress_values": pipe(
                    sorted(
                        {
                            int(row["actual_progress_steps"])
                            for row in rows
                            if row["actual_progress_steps"] not in ("", None)
                        }
                    )
                ),
                "inherited_checkpoint_lineage_step_values": pipe(
                    sorted(
                        {
                            int(row["inherited_checkpoint_step"])
                            for row in rows
                            if row["inherited_checkpoint_step"]
                            not in ("", None)
                        }
                    )
                ),
                "selected_model_lineage_step_values": pipe(
                    sorted(
                        {
                            max(
                                as_int(
                                    row.get(
                                        "direct_executed_learning_steps"
                                    ),
                                    0,
                                )
                                or 0,
                                as_int(
                                    row.get("inherited_checkpoint_step"), 0
                                )
                                or 0,
                            )
                            for row in rows
                        }
                    )
                ),
                "all_instances_complete": all(bool(row["completed"]) for row in rows),
                "standardized_evaluation_count": len(evaluations),
                "best_evaluation_id": best["unique_evaluation_id"] if best else "",
                "best_evaluation_seed_count": best["seed_count"] if best else "",
                "best_near_rate": best["near_rate"] if best else "",
                "best_task_rate": best["task_rate"] if best else "",
                "best_strict_rate": best["strict_rate"] if best else "",
                "best_local_task_rate": best_local.get("local_eval_task_rate", ""),
                "best_local_mean_return": best_local.get("local_eval_mean_return", ""),
                "feature_tags": pipe(
                    sorted(
                        {
                            tag
                            for row in rows
                            for tag in str(row["feature_tags"]).split(" | ")
                            if tag
                        }
                    )
                ),
                "run_paths": pipe(row["canonical_run_path"] for row in rows),
            }
        )
    return output


def aggregate_configuration_signature_inventory(
    physical_runs: Sequence[dict[str, Any]], primary_id: str
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in physical_runs:
        if row["repo_id"] != primary_id:
            continue
        signature_value = str(row.get("config_signature", ""))
        groups[(signature_value, str(row["metadata_kind"]))].append(row)
    output: list[dict[str, Any]] = []
    for (signature_value, metadata_kind), rows in sorted(groups.items()):
        seeds = sorted(
            {
                int(row["seed"])
                for row in rows
                if row.get("seed") not in ("", None)
            }
        )
        completed_seeds = sorted(
            {
                int(row["seed"])
                for row in rows
                if row.get("completed")
                and row.get("seed") not in ("", None)
            }
        )
        output.append(
            {
                "configuration_signature_id": hashlib.sha256(
                    f"{signature_value}|{metadata_kind}".encode()
                ).hexdigest()[:16],
                "config_or_manifest_signature": signature_value,
                "metadata_kind": metadata_kind,
                "family_count": len(
                    {str(row["family_key"]) for row in rows}
                ),
                "signature_conflates_multiple_run_families": len(
                    {str(row["family_key"]) for row in rows}
                )
                > 1,
                "family_keys": pipe(
                    sorted({str(row["family_key"]) for row in rows})
                ),
                "run_instance_count": len(rows),
                "seed_ids": pipe(seeds),
                "seed_count": len(seeds),
                "less_than_5_seeds": len(seeds) < 5,
                "completed_seed_ids": pipe(completed_seeds),
                "completed_seed_count": len(completed_seeds),
                "less_than_5_completed_seeds": len(completed_seeds) < 5,
                "configured_step_values": pipe(
                    sorted(
                        {
                            int(row["configured_total_steps"])
                            for row in rows
                            if row.get("configured_total_steps") not in ("", None)
                        }
                    )
                ),
                "observed_learning_step_values": pipe(
                    sorted(
                        {
                            int(row["direct_executed_learning_steps"])
                            for row in rows
                            if row.get("direct_executed_learning_steps")
                            not in ("", None)
                        }
                    )
                ),
                "inherited_checkpoint_lineage_step_values": pipe(
                    sorted(
                        {
                            int(row["inherited_checkpoint_step"])
                            for row in rows
                            if row.get("inherited_checkpoint_step")
                            not in ("", None)
                        }
                    )
                ),
                "categories": pipe(
                    sorted({str(row["category_direct"]) for row in rows})
                ),
                "cheating_status": (
                    "cheating"
                    if any(
                        row["cheating_status_direct"] == "cheating"
                        for row in rows
                    )
                    else "clean"
                ),
                "manual_state_region_evidence": pipe(
                    sorted(
                        {
                            str(row["manual_state_region_evidence"])
                            for row in rows
                            if row.get("manual_state_region_evidence")
                        }
                    )
                ),
                "feature_tags": pipe(
                    sorted(
                        {
                            tag
                            for row in rows
                            for tag in str(row["feature_tags"]).split(" | ")
                            if tag
                        }
                    )
                ),
                "execution_statuses": pipe(
                    sorted({str(row["execution_status"]) for row in rows})
                ),
                "run_paths": pipe(
                    sorted(str(row["canonical_run_path"]) for row in rows)
                ),
            }
        )
    return output


def aggregate_config_parent_groups(
    physical_runs: Sequence[dict[str, Any]], primary_id: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in physical_runs:
        if row["repo_id"] != primary_id:
            continue
        if "config.json" not in str(row.get("metadata_files", "")):
            continue
        run_path = str(row["canonical_run_path"]).rstrip("/")
        parts = run_path.split("/")
        parent = (
            "/".join(parts[:-1])
            if parts and re.fullmatch(r"(?i)seed\d+", parts[-1])
            else run_path
        )
        groups[parent].append(row)
    output: list[dict[str, Any]] = []
    for parent, rows in sorted(groups.items()):
        seeds = sorted(
            {
                int(row["seed"])
                for row in rows
                if row.get("seed") not in ("", None)
            }
        )
        completed = sorted(
            {
                int(row["seed"])
                for row in rows
                if row.get("completed")
                and row.get("seed") not in ("", None)
            }
        )
        output.append(
            {
                "config_parent_group": parent,
                "config_count": len(rows),
                "seed_ids": pipe(seeds),
                "seed_count": len(seeds),
                "less_than_5_seeds": len(seeds) < 5,
                "completed_seed_ids": pipe(completed),
                "completed_seed_count": len(completed),
                "less_than_5_completed_seeds": len(completed) < 5,
                "categories": pipe(
                    sorted({str(row["category_direct"]) for row in rows})
                ),
                "cheating_status": (
                    "cheating"
                    if any(
                        row["cheating_status_direct"] == "cheating"
                        for row in rows
                    )
                    else "clean"
                ),
                "run_paths": pipe(
                    sorted(str(row["canonical_run_path"]) for row in rows)
                ),
            }
        )
    return output


def classify_wrong_budget(row: Mapping[str, Any]) -> str:
    steps = as_int(row.get("executed_learning_transition_steps"))
    if steps is None:
        steps = as_int(row.get("step_hint_from_name"))
    reasons: list[str] = []
    if steps is None:
        reasons.append("budget lineage unknown")
    elif steps < TARGET_STEPS:
        reasons.append(f"trained below target budget ({steps:,})")
    elif steps > TARGET_STEPS:
        reasons.append(f"trained above target budget ({steps:,})")
    if int(row.get("seed_count", 0)) < 5:
        reasons.append(f"only {row.get('seed_count', 0)} standardized seed(s)")
    return "; ".join(reasons)


def promising_evaluations(
    unique_evaluations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in unique_evaluations:
        if not row["standard_grid_protocol"]:
            continue
        strong = (
            float(row["near_rate"]) >= 0.93
            or float(row["task_rate"]) >= 0.925
            or float(row["strict_rate"]) >= 0.15
        )
        wrong = classify_wrong_budget(row)
        if not strong or not wrong:
            continue
        reasons: list[str] = []
        if float(row["near_rate"]) >= 0.93:
            reasons.append("near-reference >= 93%")
        if float(row["task_rate"]) >= 0.925:
            reasons.append("task success >= 92.5%")
        if float(row["strict_rate"]) >= 0.15:
            reasons.append("strict wins >= 15%")
        output.append(
            {
                "unique_evaluation_id": row["unique_evaluation_id"],
                "method": row["method"],
                "category": row["category"],
                "cheating_status": row["cheating_status"],
                "manual_state_region_training": row["manual_state_region_training"],
                "inference_model_mixture": row["inference_model_mixture"],
                "seed_count": row["seed_count"],
                "near_rate": row["near_rate"],
                "task_rate": row["task_rate"],
                "strict_rate": row["strict_rate"],
                "mean_return": row["mean_return"],
                "executed_learning_transition_steps": row[
                    "executed_learning_transition_steps"
                ],
                "step_hint_from_name": row["step_hint_from_name"],
                "wrong_budget_or_replication_reason": wrong,
                "promising_signal": pipe(reasons),
                "source": row["summary_path"],
                "run_paths": row["run_paths"],
            }
        )
    output.sort(
        key=lambda row: (
            row["cheating_status"] == "clean",
            int(row["seed_count"]),
            float(row["near_rate"]),
            float(row["task_rate"]),
            float(row["strict_rate"]),
        ),
        reverse=True,
    )
    return output


def promising_unstandardized_families(
    families: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in families:
        local_task = as_float(row.get("best_local_task_rate"))
        local_return = as_float(row.get("best_local_mean_return"))
        steps = [
            int(value.strip())
            for value in str(row.get("target_step_values", "")).split("|")
            if value.strip().isdigit()
        ]
        wrong_steps = any(step != TARGET_STEPS for step in steps) or not steps
        underreplicated = int(row.get("completed_seed_count", 0)) < 5
        promising = (
            (local_task is not None and local_task >= 0.9)
            or (local_return is not None and local_return >= -150)
        )
        if (
            promising
            and int(row.get("standardized_evaluation_count", 0)) == 0
            and (wrong_steps or underreplicated)
        ):
            output.append(dict(row))
    output.sort(
        key=lambda row: (
            row["cheating_status"] == "clean",
            float(row.get("best_local_task_rate") or -1),
            float(row.get("best_local_mean_return") or -math.inf),
        ),
        reverse=True,
    )
    return output


def build_exceptions(
    physical_runs: Sequence[dict[str, Any]],
    families: Sequence[dict[str, Any]],
    unique_evaluations: Sequence[dict[str, Any]],
    primary_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    evaluated_runs = {
        run
        for evaluation in unique_evaluations
        for run in str(evaluation.get("run_paths", "")).split(" | ")
        if run
    }
    known_runs = {
        str(run["canonical_run_path"])
        for run in physical_runs
        if run["repo_id"] == primary_id
    }
    for run in physical_runs:
        if run["repo_id"] != primary_id:
            continue
        path = str(run["canonical_run_path"])
        if not run["completed"]:
            rows.append(
                {
                    "severity": "high"
                    if "5seed" in path.lower() or "n5" in path.lower()
                    else "medium",
                    "exception_type": "incomplete_or_aborted_run",
                    "subject": path,
                    "evidence": (
                        f"target={run['direct_executed_learning_steps']}; "
                        f"progress={run['actual_progress_steps']}; "
                        f"final_checkpoint={run['final_checkpoint']}"
                    ),
                }
            )
        if (
            run["metadata_kind"] == "online_rl"
            and run.get("metrics_file_integrity") != "ok"
        ):
            rows.append(
                {
                    "severity": "medium",
                    "exception_type": "trainer_metrics_file_missing_or_corrupt",
                    "subject": path,
                    "evidence": run.get("metrics_file_integrity", ""),
                }
            )
        if run.get("events_parse_error"):
            rows.append(
                {
                    "severity": "medium",
                    "exception_type": "events_jsonl_parse_error",
                    "subject": path,
                    "evidence": "event parser reported malformed content",
                }
            )
        if run["completed"] and path not in evaluated_runs:
            rows.append(
                {
                    "severity": "medium",
                    "exception_type": "completed_checkpoint_without_standard_grid_evaluation",
                    "subject": path,
                    "evidence": (
                        f"local_task={run.get('local_eval_task_rate', '')}; "
                        f"local_return={run.get('local_eval_mean_return', '')}"
                    ),
                }
            )
        for dependency in str(run.get("dependencies", "")).split(" | "):
            if dependency and dependency not in known_runs:
                rows.append(
                    {
                        "severity": "high",
                        "exception_type": "training_lineage_dependency_missing_from_both_worktrees",
                        "subject": path,
                        "evidence": dependency,
                    }
                )
    for family in families:
        if family["planned_five_seed_incomplete"]:
            rows.append(
                {
                    "severity": "high",
                    "exception_type": "named_or_configured_five_seed_family_incomplete",
                    "subject": family["family_key"],
                    "evidence": (
                        f"configured={family['configured_seed_ids']}; "
                        f"completed={family['completed_seed_ids']}"
                    ),
                }
            )
    for evaluation in unique_evaluations:
        for run_path in str(evaluation.get("run_paths", "")).split(" | "):
            if run_path and run_path not in known_runs:
                rows.append(
                    {
                        "severity": "high",
                        "exception_type": "evaluation_run_dependency_missing_from_inventory",
                        "subject": evaluation["summary_path"],
                        "evidence": run_path,
                    }
                )
        if not evaluation["source_rollouts_exist"]:
            rows.append(
                {
                    "severity": "high",
                    "exception_type": "evaluation_summary_missing_source_rollouts",
                    "subject": evaluation["summary_path"],
                    "evidence": evaluation["source_rollouts"],
                }
            )
        if evaluation["cheating_status"] == "unknown":
            rows.append(
                {
                    "severity": "medium",
                    "exception_type": "evaluation_legality_or_lineage_unknown",
                    "subject": evaluation["summary_path"],
                    "evidence": evaluation["lineage_evidence"],
                }
            )
    deduped = {
        (row["exception_type"], row["subject"]): row
        for row in rows
    }
    rank = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        deduped.values(),
        key=lambda row: (rank[row["severity"]], row["exception_type"], row["subject"]),
    )


def build_coverage_checks(
    repos: Sequence[tuple[str, Path]],
    artifact_manifest: Sequence[Mapping[str, Any]],
    reconciliation: Sequence[Mapping[str, Any]],
    json_parse_audit: Sequence[Mapping[str, Any]],
    physical_runs: Sequence[Mapping[str, Any]],
    aliases: Sequence[Mapping[str, Any]],
    unique_evaluations: Sequence[Mapping[str, Any]],
    checkpoint_aliases: Sequence[Mapping[str, Any]],
    changed_during_scan: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    primary_id = repos[0][0]
    primary_relative = [
        row
        for row in aliases
        if row["repo_id"] == primary_id
        and row["summary_kind"] == "relative_summary.json"
    ]
    primary_authority = [
        row
        for row in aliases
        if row["repo_id"] == primary_id
        and row["summary_kind"] == "authoritative_summary.json"
    ]
    physical_primary_relative = [
        row
        for row in artifact_manifest
        if row["repo_id"] == primary_id
        and row["artifact_role"] == "relative_evaluation_summary"
    ]
    physical_primary_rollouts = [
        row
        for row in artifact_manifest
        if row["repo_id"] == primary_id
        and row["artifact_role"] == "relative_rollout_rows"
    ]
    seed_counts = Counter(int(row["seed_count"]) for row in primary_relative)
    checks: list[tuple[str, str, Any, Any, str]] = [
        (
            "neighbor_has_no_unique_artifact",
            "pass"
            if not any(row["relation"] == "neighbor_only" for row in reconciliation)
            else "fail",
            0,
            sum(row["relation"] == "neighbor_only" for row in reconciliation),
            "The historical worktree must be a contained alias, not a hidden result store.",
        ),
        (
            "every_primary_relative_summary_parsed",
            "pass"
            if len(primary_relative) == len(physical_primary_relative)
            else "fail",
            len(physical_primary_relative),
            len(primary_relative),
            "Filename-independent classification still requires every canonical relative summary to parse.",
        ),
        (
            "every_primary_relative_summary_has_scored_row_table",
            "pass"
            if all(bool(row["relative_rollouts_exist"]) for row in primary_relative)
            else "fail",
            len(primary_relative),
            sum(bool(row["relative_rollouts_exist"]) for row in primary_relative),
            "The scored relative tables, rather than stale raw-grid pointers, define the result identity.",
        ),
        (
            "relative_summary_and_rollout_counts_match",
            "pass"
            if len(physical_primary_relative) == len(physical_primary_rollouts)
            else "warn",
            len(physical_primary_relative),
            len(physical_primary_rollouts),
            "A mismatch can be legitimate only when an authoritative wrapper reuses another result.",
        ),
        (
            "known_seed_distribution_reconciled",
            "pass"
            if seed_counts
            == Counter({1: 315, 2: 10, 3: 15, 4: 7, 5: 62})
            else "fail",
            "1:315|2:10|3:15|4:7|5:62",
            pipe(f"{key}:{value}" for key, value in sorted(seed_counts.items())),
            "Independent actor-seed count from each stored relative evaluation.",
        ),
        (
            "physical_relative_rollout_table_hashes_reconciled",
            "pass"
            if len(
                {
                    row["content_identity"]
                    for row in primary_relative
                    if str(row["content_identity"]).startswith("rows:")
                }
            )
            == 402
            else "warn",
            402,
            len(
                {
                    row["content_identity"]
                    for row in primary_relative
                    if str(row["content_identity"]).startswith("rows:")
                }
            ),
            "Exact row hashes expose report copies and naming aliases without comparing only aggregate metric triples.",
        ),
        (
            "canonical_evaluation_count_reconciled",
            "pass" if len(unique_evaluations) == 402 else "warn",
            402,
            len(unique_evaluations),
            "Collapses cross-worktree mirrors, authority wrappers, copied checkpoint evaluations, and byte-identical specialist aliases.",
        ),
        (
            "checkpoint_alias_census",
            "pass" if len(checkpoint_aliases) == 22 else "warn",
            22,
            len(checkpoint_aliases),
            "Copied intermediate checkpoints and SWA views must not be counted as new executions.",
        ),
        (
            "custom_manifest_runs_included",
            "pass"
            if any(
                row["repo_id"] == primary_id
                and row["metadata_kind"] == "custom_training_manifest"
                for row in physical_runs
            )
            else "fail",
            ">0",
            sum(
                row["repo_id"] == primary_id
                and row["metadata_kind"] == "custom_training_manifest"
                for row in physical_runs
            ),
            "Covers completed custom DAgger/BC families without config.json.",
        ),
        (
            "stale_joint_followup_raw_grid_links_recovered",
            "pass"
            if sum(
                bool(
                    row[
                        "stale_rollout_pointer_recovered_from_stable_sibling"
                    ]
                )
                for row in primary_relative
            )
            == 47
            else "warn",
            47,
            sum(
                bool(
                    row[
                        "stale_rollout_pointer_recovered_from_stable_sibling"
                    ]
                )
                for row in primary_relative
            ),
            "All scored relative tables were present; only temporary raw-grid source links were stale.",
        ),
        (
            "standardized_results_are_exact_protocol",
            "pass"
            if all(bool(row["standard_grid_protocol"]) for row in unique_evaluations)
            else "fail",
            len(unique_evaluations),
            sum(bool(row["standard_grid_protocol"]) for row in unique_evaluations),
            "61x41 cells, epsilon 5, and trials equal cells times actor seeds.",
        ),
        (
            "json_syntax_parse_failures",
            "pass"
            if not any(row["parse_status"] == "failed" for row in json_parse_audit)
            else "warn",
            0,
            sum(row["parse_status"] == "failed" for row in json_parse_audit),
            "UTF-8 BOM and UTF-16 files are explicitly supported; remaining failures need review.",
        ),
        (
            "live_files_changed_during_scan",
            "warn" if changed_during_scan else "pass",
            0,
            len(changed_during_scan),
            "A nonzero value makes this a timestamped provisional snapshot.",
        ),
        (
            "authoritative_summary_wrappers_accounted",
            "pass",
            len(primary_authority),
            len(primary_authority),
            "Authority summaries are indexed but deduplicated by their scored row table when shared.",
        ),
    ]
    return [
        {
            "check": name,
            "status": status,
            "expected": expected,
            "observed": observed,
            "meaning": meaning,
        }
        for name, status, expected, observed, meaning in checks
    ]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ""
                    if value is None or (isinstance(value, float) and not math.isfinite(value))
                    else value
                    for key in fieldnames
                    for value in [row.get(key)]
                }
            )


def make_figures(
    output: Path,
    unique_evaluations: Sequence[dict[str, Any]],
    families: Sequence[dict[str, Any]],
) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    evals = pd.DataFrame(unique_evaluations)
    standard = evals[evals["standard_grid_protocol"] == True].copy()  # noqa: E712
    if not standard.empty:
        counts = (
            standard.groupby(["seed_count", "cheating_status"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        fig, ax = plt.subplots(figsize=(10, 5.5))
        counts.plot(kind="bar", stacked=True, ax=ax, color={"clean": "#2A9D8F", "cheating": "#C8553D", "unknown": "#7A7A7A"})
        ax.set_title("Unique standardized evaluations by seed count and legality")
        ax.set_xlabel("Training seeds in the stored evaluation")
        ax.set_ylabel("Unique rollout-table evaluations")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(figure_dir / "01_evaluations_by_seed_and_legality.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 7))
        colors = {
            "pure RL": "#1F77B4",
            "RL + supervised": "#2A9D8F",
            "supervised only": "#9467BD",
            "unknown": "#7A7A7A",
        }
        for category, group in standard.groupby("category"):
            clean = group[group["cheating_status"] == "clean"]
            cheat = group[group["cheating_status"] != "clean"]
            ax.scatter(
                clean["near_rate"] * 100,
                clean["task_rate"] * 100,
                s=25 + 25 * clean["seed_count"],
                alpha=0.62,
                color=colors.get(category, "#7A7A7A"),
                label=f"{category}, clean",
            )
            if not cheat.empty:
                ax.scatter(
                    cheat["near_rate"] * 100,
                    cheat["task_rate"] * 100,
                    s=25 + 25 * cheat["seed_count"],
                    alpha=0.75,
                    facecolors="none",
                    edgecolors=colors.get(category, "#7A7A7A"),
                    linewidths=1.2,
                    label=f"{category}, flagged/unknown",
                )
        ax.axvline(93, color="#444444", linestyle=":", linewidth=1)
        ax.axhline(92.5, color="#444444", linestyle=":", linewidth=1)
        ax.set_title("All standardized results, including one-seed and wrong-budget probes")
        ax.set_xlabel("Near-reference success (%)")
        ax.set_ylabel("Task success (%)")
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(figure_dir / "02_all_standardized_results_scatter.png", dpi=180)
        plt.close(fig)

    family_frame = pd.DataFrame(families)
    if not family_frame.empty:
        values: list[int] = []
        for text in family_frame["target_step_values"]:
            for raw in str(text).split("|"):
                raw = raw.strip()
                if raw.isdigit():
                    values.append(int(raw))
        if values:
            bins = [0, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, max(values) + 1]
            bins = sorted(set(bins))
            fig, ax = plt.subplots(figsize=(11, 5.5))
            ax.hist(values, bins=bins, color="#3B7EA1", edgecolor="white")
            ax.set_xscale("symlog", linthresh=5_000)
            ax.axvline(TARGET_STEPS, color="#C8553D", linestyle="--", label="100k target")
            ax.set_title("Stored run-family training budgets")
            ax.set_xlabel("Direct learning transitions per run")
            ax.set_ylabel("Family-budget entries")
            ax.legend()
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            fig.savefig(figure_dir / "03_run_family_budget_distribution.png", dpi=180)
            plt.close(fig)


def build_inventory(
    repos: Sequence[tuple[str, Path]], output: Path
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    scan_started = datetime.now(timezone.utc)
    before_snapshot = snapshot_file_state(repos, excluded=output)
    hash_cache = HashCache()
    artifact_manifest, reconciliation = build_artifact_manifest(
        repos, hash_cache, excluded=output
    )
    metric_json_rows, json_parse_audit = discover_metric_jsons(
        repos, excluded=output
    )
    tabular_metric_rows, tabular_metric_audit = discover_tabular_metric_rows(
        repos, excluded=output
    )
    checkpoint_aliases = build_checkpoint_alias_audit(repos[0][1], hash_cache)
    alias_lineage = {
        canonical_run_rel(row["alias_run_path"]): int(
            row["inherited_checkpoint_lineage_step"]
        )
        for row in checkpoint_aliases
        if row["inherited_checkpoint_lineage_step"] not in ("", None)
    }
    physical_runs, primary_run_map = discover_runs(repos, alias_lineage)
    aliases = discover_evaluation_aliases(
        repos, hash_cache, alias_lineage, primary_run_map
    )
    enrich_noncanonical_result_rows(
        metric_json_rows,
        tabular_metric_rows,
        aliases,
        repos,
        primary_run_map,
    )
    unique_evaluations, alias_rows = deduplicate_evaluations(aliases, repos[0][0])
    families = aggregate_run_families(
        physical_runs, unique_evaluations, repos[0][0]
    )
    configuration_signatures = aggregate_configuration_signature_inventory(
        physical_runs, repos[0][0]
    )
    config_parent_groups = aggregate_config_parent_groups(
        physical_runs, repos[0][0]
    )
    promising = promising_evaluations(unique_evaluations)
    promising_local = promising_unstandardized_families(families)
    exceptions = build_exceptions(
        physical_runs, families, unique_evaluations, repos[0][0]
    )

    write_csv(output / "artifact_manifest.csv", artifact_manifest)
    write_csv(output / "repo_reconciliation.csv", reconciliation)
    write_csv(output / "metric_json_inventory.csv", metric_json_rows)
    write_csv(output / "json_parse_audit.csv", json_parse_audit)
    write_csv(output / "tabular_metric_rows.csv", tabular_metric_rows)
    write_csv(output / "tabular_metric_audit.csv", tabular_metric_audit)
    write_csv(output / "checkpoint_alias_audit.csv", checkpoint_aliases)
    write_csv(output / "run_instances.csv", physical_runs)
    write_csv(output / "run_families.csv", families)
    write_csv(
        output / "configuration_signature_inventory.csv",
        configuration_signatures,
    )
    write_csv(output / "config_parent_groups.csv", config_parent_groups)
    write_csv(output / "evaluation_aliases.csv", alias_rows)
    write_csv(output / "unique_standardized_evaluations.csv", unique_evaluations)
    write_csv(output / "promising_wrong_budget_or_underreplicated.csv", promising)
    write_csv(output / "promising_local_eval_only_families.csv", promising_local)
    write_csv(output / "orphan_incomplete_and_missing_eval_audit.csv", exceptions)
    make_figures(output, unique_evaluations, families)
    after_snapshot = snapshot_file_state(repos, excluded=output)
    changed_during_scan = compare_snapshots(before_snapshot, after_snapshot)
    write_csv(output / "changed_during_scan.csv", changed_during_scan)
    coverage_checks = build_coverage_checks(
        repos,
        artifact_manifest,
        reconciliation,
        json_parse_audit,
        physical_runs,
        aliases,
        unique_evaluations,
        checkpoint_aliases,
        changed_during_scan,
    )
    write_csv(output / "coverage_checks.csv", coverage_checks)
    scan_finished = datetime.now(timezone.utc)

    unique_standard = [
        row for row in unique_evaluations if row["standard_grid_protocol"]
    ]
    counts = {
        "filesystem_artifacts": len(artifact_manifest),
        "project15_artifacts": sum(
            row["repo_id"] == repos[0][0] for row in artifact_manifest
        ),
        "sacn_worktree_artifacts": sum(
            row["repo_id"] == repos[1][0] for row in artifact_manifest
        ),
        "neighbor_only_artifacts": sum(
            row["relation"] == "neighbor_only" for row in reconciliation
        ),
        "exact_cross_repo_mirrors": sum(
            row["relation"] == "exact_mirror" for row in reconciliation
        ),
        "same_path_changed_cross_repo_files": sum(
            row["relation"] == "same_path_changed" for row in reconciliation
        ),
        "report_json_files": len(json_parse_audit),
        "report_json_parse_failures": sum(
            row["parse_status"] == "failed" for row in json_parse_audit
        ),
        "result_bearing_json_nodes": len(metric_json_rows),
        "result_bearing_json_files": len(
            {(row["repo_id"], row["json_path"]) for row in metric_json_rows}
        ),
        "tabular_result_rows": len(tabular_metric_rows),
        "tabular_result_files": len(
            {(row["repo_id"], row["csv_path"]) for row in tabular_metric_rows}
        ),
        "unique_tabular_result_rows_after_cross_repo_aliases": len(
            {
                row["canonical_tabular_row_id"]
                for row in tabular_metric_rows
            }
        ),
        "checkpoint_alias_instances": len(checkpoint_aliases),
        "checkpoint_alias_exact_copies": sum(
            row["alias_type"] == "byte_exact_checkpoint_copy"
            for row in checkpoint_aliases
        ),
        "checkpoint_alias_swa_without_manifest": sum(
            row["alias_type"] == "synthesized_actor_swa"
            for row in checkpoint_aliases
        ),
        "run_directory_instances": len(physical_runs),
        "physical_training_execution_instances": sum(
            row["metadata_kind"] != "checkpoint_or_swa_alias"
            for row in physical_runs
        ),
        "primary_run_instances": sum(
            row["repo_id"] == repos[0][0] for row in physical_runs
        ),
        "neighbor_run_instances": sum(
            row["repo_id"] == repos[1][0] for row in physical_runs
        ),
        "primary_run_families": len(families),
        "primary_config_parent_groups": len(config_parent_groups),
        "primary_config_parent_groups_under_five_seeds": sum(
            bool(row["less_than_5_seeds"]) for row in config_parent_groups
        ),
        "primary_configuration_signature_groups": len(
            configuration_signatures
        ),
        "primary_configuration_signature_groups_under_five_seeds": sum(
            bool(row["less_than_5_seeds"])
            for row in configuration_signatures
        ),
        "primary_completed_physical_training_instances": sum(
            row["repo_id"] == repos[0][0]
            and row["metadata_kind"] != "checkpoint_or_swa_alias"
            and bool(row["completed"])
            for row in physical_runs
        ),
        "primary_incomplete_or_unresolved_physical_training_instances": sum(
            row["repo_id"] == repos[0][0]
            and row["metadata_kind"] != "checkpoint_or_swa_alias"
            and not bool(row["completed"])
            for row in physical_runs
        ),
        "primary_online_rl_metrics_files_missing_or_corrupt": sum(
            row["repo_id"] == repos[0][0]
            and row["metadata_kind"] == "online_rl"
            and row["metrics_file_integrity"] != "ok"
            for row in physical_runs
        ),
        "evaluation_summary_aliases": len(alias_rows),
        "unique_evaluations": len(unique_evaluations),
        "unique_standard_grid_evaluations": len(unique_standard),
        "unique_five_or_more_seed_evaluations": sum(
            int(row["seed_count"]) >= 5 for row in unique_standard
        ),
        "unique_under_five_seed_evaluations": sum(
            int(row["seed_count"]) < 5 for row in unique_standard
        ),
        "clean_standard_evaluations": sum(
            row["cheating_status"] == "clean" for row in unique_standard
        ),
        "cheating_standard_evaluations": sum(
            row["cheating_status"] == "cheating" for row in unique_standard
        ),
        "unknown_legality_standard_evaluations": sum(
            row["cheating_status"] == "unknown" for row in unique_standard
        ),
        "pure_rl_standard_evaluations": sum(
            row["category"] == "pure RL" for row in unique_standard
        ),
        "rl_supervised_standard_evaluations": sum(
            row["category"] == "RL + supervised" for row in unique_standard
        ),
        "promising_wrong_budget_or_underreplicated": len(promising),
        "promising_local_eval_only_families": len(promising_local),
        "high_severity_exceptions": sum(
            row["severity"] == "high" for row in exceptions
        ),
        "all_exceptions": len(exceptions),
        "files_changed_during_scan": len(changed_during_scan),
        "coverage_check_failures": sum(
            row["status"] == "fail" for row in coverage_checks
        ),
        "coverage_check_warnings": sum(
            row["status"] == "warn" for row in coverage_checks
        ),
    }
    seed_distribution = Counter(int(row["seed_count"]) for row in unique_standard)
    category_distribution = Counter(str(row["category"]) for row in unique_standard)
    legality_distribution = Counter(
        str(row["cheating_status"]) for row in unique_standard
    )
    manifest = {
        "generated_date": "2026-07-23",
        "scan_started_utc": scan_started.isoformat(),
        "scan_finished_utc": scan_finished.isoformat(),
        "repositories": [
            {"repo_id": repo_id, "path": str(repo.resolve())}
            for repo_id, repo in repos
        ],
        "classification_policy": {
            "cheating": [
                "hardcoded state angle or velocity region used to select training data, resets, replay, rollouts, anchors, or shaping",
                "hardcoded state-range router at inference",
                "reference policy queried at inference",
            ],
            "allowed_but_separate_flags": [
                "Q-search",
                "automatic performance-based state discovery",
                "automatically learned ranges",
                "reference supervision during training",
                "model mixtures are flagged separately even when not range cheating",
            ],
            "pure_rl": "reward-only training throughout the deployed actor/critic lineage",
            "less_than_five_seeds": "stored standardized evaluation contains fewer than five distinct training seeds",
            "target_budget": TARGET_STEPS,
        },
        "counts": counts,
        "seed_count_distribution": dict(sorted(seed_distribution.items())),
        "category_distribution": dict(category_distribution),
        "legality_distribution": dict(legality_distribution),
        "output_files": [
            "artifact_manifest.csv",
            "repo_reconciliation.csv",
            "metric_json_inventory.csv",
            "json_parse_audit.csv",
            "tabular_metric_rows.csv",
            "tabular_metric_audit.csv",
            "checkpoint_alias_audit.csv",
            "run_instances.csv",
            "run_families.csv",
            "configuration_signature_inventory.csv",
            "config_parent_groups.csv",
            "evaluation_aliases.csv",
            "unique_standardized_evaluations.csv",
            "promising_wrong_budget_or_underreplicated.csv",
            "promising_local_eval_only_families.csv",
            "orphan_incomplete_and_missing_eval_audit.csv",
            "changed_during_scan.csv",
            "coverage_checks.csv",
        ],
    }
    (output / "coverage_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary",
        type=Path,
        default=Path.cwd(),
        help="Primary Project 15 worktree",
    )
    parser.add_argument(
        "--neighbor",
        type=Path,
        default=Path.cwd().parent / "Project 15-sac-n-experiments",
        help="Neighbor SAC-N worktree",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/two_repo_forensic_inventory_20260723"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary = args.primary.resolve()
    neighbor = args.neighbor.resolve()
    output = args.output
    if not output.is_absolute():
        output = primary / output
    manifest = build_inventory(
        [("project15", primary), ("sacn_worktree", neighbor)],
        output.resolve(),
    )
    print(json.dumps({"output": str(output), "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
