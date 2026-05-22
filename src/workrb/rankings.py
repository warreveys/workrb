"""Loading and validation utilities for ranking artifacts.

A ranking artifact is the JSON file produced by
:meth:`BenchmarkConfig.save_rankings_artifact` for one ``(task, dataset_id)``
pair. This module provides the read-side counterparts used by
:func:`workrb.evaluate_rankings` to replay metrics without re-running a model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from workrb.tasks.abstract.ranking_base import RankingDataset

logger = logging.getLogger(__name__)

# Current on-disk schema version for ranking artifacts.
#
# Bumped when the JSON shape itself changes (field renames, new required
# fields, restructuring of `scores`). Independent of ``workrb_version``:
# a workrb release that does not change the schema keeps the same value.
SCHEMA_VERSION = 1

# Schema versions this reader can parse. Artifacts with any other
# ``schema_version`` are hard-rejected by :func:`validate_header`.
SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({1})

# Pinned header field set per schema version.
#
# This is the source of truth that
# :func:`workrb.config.BenchmarkConfig.save_rankings_artifact` must agree
# with. A test (``test_writer_header_matches_pinned_schema``) compares
# freshly written headers against this mapping; mismatch means the writer
# changed without bumping :data:`SCHEMA_VERSION`. When evolving the schema:
#
# 1. Add a new entry ``SCHEMA_HEADER_FIELDS[N] = frozenset({...})``.
# 2. Bump :data:`SCHEMA_VERSION` to ``N``.
# 3. Decide whether to keep the old version in
#    :data:`SUPPORTED_SCHEMA_VERSIONS` (back-compat) or drop it
#    (force rewrite).
SCHEMA_HEADER_FIELDS: dict[int, frozenset[str]] = {
    1: frozenset(
        {
            "schema_version",
            "workrb_version",
            "model_name",
            "task_name",
            "dataset_id",
            "split",
            "num_queries",
            "num_targets",
            "first_query_text",
            "last_query_text",
            "first_target_text",
            "last_target_text",
        }
    ),
}


class RankingsArtifactMissing(FileNotFoundError):
    """Raised when a needed ranking artifact file is not on disk."""

    def __init__(self, path: Path, task_name: str, dataset_id: str):
        self.path = path
        self.task_name = task_name
        self.dataset_id = dataset_id
        super().__init__(
            f"No ranking artifact for task '{task_name}', dataset '{dataset_id}' at {path}"
        )


class RankingsArtifactInvalid(ValueError):
    """Raised when a ranking artifact does not match the current dataset."""

    def __init__(self, path: Path, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Invalid ranking artifact at {path}: {reason}")


def load_rankings_artifact(path: Path) -> tuple[dict, dict[int, dict[int, float]]]:
    """Load a ranking artifact and return ``(header, scores)`` with integer keys.

    JSON object keys are always strings, so both axes are cast back to ``int``.
    """
    with open(path) as f:
        payload = json.load(f)

    if "header" not in payload or "scores" not in payload:
        raise RankingsArtifactInvalid(path, "missing 'header' or 'scores' top-level keys")

    header = payload["header"]
    raw_scores = payload["scores"]
    scores: dict[int, dict[int, float]] = {
        int(q): {int(t): float(s) for t, s in row.items()} for q, row in raw_scores.items()
    }
    return header, scores


def materialize_prediction_matrix(
    scores: dict[int, dict[int, float]],
    num_queries: int,
    num_targets: int,
) -> np.ndarray:
    """Build a dense ``(num_queries, num_targets)`` matrix from sparse scores."""
    matrix = np.zeros((num_queries, num_targets), dtype=np.float32)
    for q_idx, row in scores.items():
        for t_idx, score in row.items():
            matrix[q_idx, t_idx] = score
    return matrix


def validate_header(
    header: dict,
    dataset: RankingDataset,
    task_name: str,
    dataset_id: str,
    split: str,
    path: Path,
    running_workrb_version: str,
) -> None:
    """Validate an artifact header against the live dataset.

    Checks are run in three tiers:

    1. ``schema_version`` must be in :data:`SUPPORTED_SCHEMA_VERSIONS`. This
       gates whether the reader can parse the file at all; an unknown schema
       version is a hard reject.
    2. Structural identity (task name, dataset id, split, sizes, canary
       strings) must match the live dataset. Any mismatch is a hard reject.
    3. ``workrb_version`` is informational: a difference logs a warning but
       does not block scoring. Cross-version replay is the intended use case
       for :func:`workrb.evaluate_rankings` (metrics may have changed); the
       real safety net is tier 2.
    """
    schema_version = header.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise RankingsArtifactInvalid(
            path,
            f"unsupported schema_version {schema_version!r}; this reader "
            f"supports {sorted(SUPPORTED_SCHEMA_VERSIONS)}",
        )

    if header.get("task_name") != task_name:
        raise RankingsArtifactInvalid(
            path,
            f"header task_name '{header.get('task_name')}' does not match task '{task_name}'",
        )
    if header.get("dataset_id") != dataset_id:
        raise RankingsArtifactInvalid(
            path,
            f"header dataset_id '{header.get('dataset_id')}' does not match dataset '{dataset_id}'",
        )
    if header.get("split") != split:
        raise RankingsArtifactInvalid(
            path,
            f"header split '{header.get('split')}' does not match task split '{split}'",
        )
    if header.get("num_queries") != len(dataset.query_texts):
        raise RankingsArtifactInvalid(
            path,
            f"num_queries mismatch: header={header.get('num_queries')}, "
            f"dataset={len(dataset.query_texts)}",
        )
    if header.get("num_targets") != len(dataset.target_space):
        raise RankingsArtifactInvalid(
            path,
            f"num_targets mismatch: header={header.get('num_targets')}, "
            f"dataset={len(dataset.target_space)}",
        )
    if header.get("first_query_text") != dataset.query_texts[0]:
        raise RankingsArtifactInvalid(path, "first_query_text canary mismatch")
    if header.get("last_query_text") != dataset.query_texts[-1]:
        raise RankingsArtifactInvalid(path, "last_query_text canary mismatch")
    if header.get("first_target_text") != dataset.target_space[0]:
        raise RankingsArtifactInvalid(path, "first_target_text canary mismatch")
    if header.get("last_target_text") != dataset.target_space[-1]:
        raise RankingsArtifactInvalid(path, "last_target_text canary mismatch")

    header_version = header.get("workrb_version")
    if header_version != running_workrb_version:
        logger.warning(
            "Ranking artifact %s was written by workrb %s, current is %s. "
            "Sizes and canaries match, proceeding.",
            path,
            header_version,
            running_workrb_version,
        )
