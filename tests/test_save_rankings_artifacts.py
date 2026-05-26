"""Tests for ranking artifact persistence and replay."""

import json
import logging
import shutil
from pathlib import Path

import pytest
import torch

import workrb
from workrb.models.base import ModelInterface
from workrb.rankings import (
    SCHEMA_HEADER_FIELDS,
    SCHEMA_VERSION,
    RankingsArtifactInvalid,
    RankingsArtifactMissing,
)
from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTask, RankingTaskGroup
from workrb.types import ModelInputType


class TinyRankingTask(RankingTask):
    """Minimal ranking task used to test ranking artifact persistence."""

    @property
    def name(self) -> str:
        return "Tiny Ranking Task"

    @property
    def description(self) -> str:
        return "Tiny in-memory ranking task for tests."

    @property
    def supported_query_languages(self) -> list[Language]:
        return [Language.EN]

    @property
    def supported_target_languages(self) -> list[Language]:
        return [Language.EN]

    @property
    def task_group(self) -> RankingTaskGroup:
        return RankingTaskGroup.SKILL_EXTRACTION

    @property
    def label_type(self) -> LabelType:
        return LabelType.MULTI_LABEL

    @property
    def query_input_type(self) -> ModelInputType:
        return ModelInputType.SKILL_SENTENCE

    @property
    def target_input_type(self) -> ModelInputType:
        return ModelInputType.SKILL_NAME

    def load_dataset(self, dataset_id: str, split: DatasetSplit) -> RankingDataset:
        return RankingDataset(
            query_texts=["query one", "query two"],
            target_indices=[[0], [1]],
            target_space=["target_a", "target_b", "target_c"],
            dataset_id=dataset_id,
        )


class TinyDeterministicModel(ModelInterface):
    """Deterministic model with fixed ranking scores for tests."""

    @property
    def name(self) -> str:
        return "tiny-deterministic-model"

    @property
    def description(self) -> str:
        return "Tiny deterministic model for ranking artifact tests."

    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type: ModelInputType,
        target_input_type: ModelInputType,
    ) -> torch.Tensor:
        # 2 queries x 3 targets, including 0.0 and a negative score to confirm
        # both roundtrip exactly (no sparsity).
        return torch.tensor(
            [
                [0.1, 0.0, -0.3],
                [0.4, 0.5, 0.6],
            ],
            dtype=torch.float32,
        )

    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type: ModelInputType,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        return torch.zeros((len(texts), len(targets)))

    @property
    def classification_label_space(self) -> list[str] | None:
        return None


def _fresh_dir(name: str) -> Path:
    path = Path("tmp") / name
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    return path


def _run_and_get_rankings_dir(output_folder: Path) -> Path:
    model = TinyDeterministicModel()
    tasks = [TinyRankingTask(split=DatasetSplit.TEST, languages=[Language.EN])]
    workrb.evaluate(
        model=model,
        tasks=tasks,
        output_folder=str(output_folder),
        force_restart=True,
        save_rankings=True,
    )
    return output_folder / "rankings" / model.name


def test_evaluate_saves_rankings_artifact_with_new_schema():
    """evaluate(save_rankings=True) writes one JSON per ranking dataset with header + sparse int-keyed scores."""
    output_folder = _fresh_dir("rankings_artifact_test_enabled")
    rankings_dir = _run_and_get_rankings_dir(output_folder)

    ranking_files = list(rankings_dir.glob("*.json"))
    assert len(ranking_files) == 1

    with open(ranking_files[0]) as f:
        payload = json.load(f)

    assert set(payload.keys()) == {"header", "scores"}

    header = payload["header"]
    assert header["schema_version"] == 1
    assert header["model_name"] == "tiny-deterministic-model"
    assert header["task_name"] == "Tiny Ranking Task"
    assert header["dataset_id"] == "en"
    assert header["split"] == "test"
    assert header["num_queries"] == 2
    assert header["num_targets"] == 3
    assert header["first_query_text"] == "query one"
    assert header["last_query_text"] == "query two"
    assert header["first_target_text"] == "target_a"
    assert header["last_target_text"] == "target_c"
    assert isinstance(header["workrb_version"], str) and header["workrb_version"]

    scores = payload["scores"]
    assert set(scores.keys()) == {"0", "1"}
    # Every (q, t) cell is stored, including 0.0 and negatives.
    assert set(scores["0"].keys()) == {"0", "1", "2"}
    assert scores["0"]["0"] == pytest.approx(0.1)
    assert scores["0"]["1"] == pytest.approx(0.0)
    assert scores["0"]["2"] == pytest.approx(-0.3)
    assert set(scores["1"].keys()) == {"0", "1", "2"}
    assert scores["1"]["0"] == pytest.approx(0.4)
    assert scores["1"]["1"] == pytest.approx(0.5)
    assert scores["1"]["2"] == pytest.approx(0.6)


def test_evaluate_does_not_save_rankings_artifact_by_default():
    """evaluate() without save_rankings does not create a rankings/ directory."""
    output_folder = _fresh_dir("rankings_artifact_test_disabled")
    model = TinyDeterministicModel()
    tasks = [TinyRankingTask(split=DatasetSplit.TEST, languages=[Language.EN])]
    workrb.evaluate(
        model=model,
        tasks=tasks,
        output_folder=str(output_folder),
        force_restart=True,
    )

    assert not (output_folder / "rankings").exists()


def test_evaluate_rankings_roundtrip_parity():
    """evaluate_rankings on saved artifacts reproduces metrics from evaluate(save_rankings=True)."""
    write_dir = _fresh_dir("rankings_artifact_roundtrip_write")
    rankings_dir = _run_and_get_rankings_dir(write_dir)

    with open(write_dir / "results.json") as f:
        write_results = json.load(f)
    write_metrics = write_results["task_results"]["Tiny Ranking Task"]["datasetid_results"]["en"][
        "metrics_dict"
    ]

    read_output = _fresh_dir("rankings_artifact_roundtrip_read")
    tasks = [TinyRankingTask(split=DatasetSplit.TEST, languages=[Language.EN])]
    replay = workrb.evaluate_rankings(
        rankings_dir=rankings_dir,
        tasks=tasks,
        output_folder=str(read_output),
    )

    replay_metrics = replay.task_results["Tiny Ranking Task"].datasetid_results["en"].metrics_dict
    assert set(replay_metrics.keys()) == set(write_metrics.keys())
    for key, value in write_metrics.items():
        assert replay_metrics[key] == pytest.approx(value)

    # Replay records which workrb version wrote the source artifacts;
    # the original evaluate() run leaves the field unset.
    assert write_results["metadata"].get("replayed_from_workrb_version") is None
    artifact_header = json.loads(next(rankings_dir.glob("*.json")).read_text())["header"]
    assert replay.metadata.replayed_from_workrb_version == artifact_header["workrb_version"]


def test_evaluate_rankings_missing_artifact_raises():
    """A missing artifact file raises RankingsArtifactMissing."""
    write_dir = _fresh_dir("rankings_artifact_missing_write")
    rankings_dir = _run_and_get_rankings_dir(write_dir)

    # Remove the only artifact, then add a stub so the directory peek succeeds
    # but the per-task lookup fails.
    files = list(rankings_dir.glob("*.json"))
    assert len(files) == 1
    stub_payload = json.loads(files[0].read_text())
    files[0].unlink()
    decoy = rankings_dir / "Some_Other_Task__xx.json"
    decoy.write_text(json.dumps(stub_payload))

    tasks = [TinyRankingTask(split=DatasetSplit.TEST, languages=[Language.EN])]
    read_output = _fresh_dir("rankings_artifact_missing_read")
    with pytest.raises(RankingsArtifactMissing):
        workrb.evaluate_rankings(
            rankings_dir=rankings_dir,
            tasks=tasks,
            output_folder=str(read_output),
        )


def _hand_edit_header(rankings_dir: Path, **header_overrides) -> Path:
    files = list(rankings_dir.glob("*.json"))
    assert len(files) == 1
    path = files[0]
    payload = json.loads(path.read_text())
    payload["header"].update(header_overrides)
    path.write_text(json.dumps(payload))
    return path


def test_evaluate_rankings_size_mismatch_raises():
    write_dir = _fresh_dir("rankings_artifact_size_mismatch_write")
    rankings_dir = _run_and_get_rankings_dir(write_dir)
    _hand_edit_header(rankings_dir, num_queries=99)

    tasks = [TinyRankingTask(split=DatasetSplit.TEST, languages=[Language.EN])]
    read_output = _fresh_dir("rankings_artifact_size_mismatch_read")
    with pytest.raises(RankingsArtifactInvalid, match="num_queries"):
        workrb.evaluate_rankings(
            rankings_dir=rankings_dir,
            tasks=tasks,
            output_folder=str(read_output),
        )


def test_evaluate_rankings_canary_mismatch_raises():
    write_dir = _fresh_dir("rankings_artifact_canary_write")
    rankings_dir = _run_and_get_rankings_dir(write_dir)
    _hand_edit_header(rankings_dir, first_query_text="this is not the original query")

    tasks = [TinyRankingTask(split=DatasetSplit.TEST, languages=[Language.EN])]
    read_output = _fresh_dir("rankings_artifact_canary_read")
    with pytest.raises(RankingsArtifactInvalid, match="first_query_text"):
        workrb.evaluate_rankings(
            rankings_dir=rankings_dir,
            tasks=tasks,
            output_folder=str(read_output),
        )


def test_evaluate_rankings_split_mismatch_raises():
    """An artifact written for one split cannot be replayed against another."""
    write_dir = _fresh_dir("rankings_artifact_split_write")
    rankings_dir = _run_and_get_rankings_dir(write_dir)
    _hand_edit_header(rankings_dir, split="val")

    tasks = [TinyRankingTask(split=DatasetSplit.TEST, languages=[Language.EN])]
    read_output = _fresh_dir("rankings_artifact_split_read")
    with pytest.raises(RankingsArtifactInvalid, match="split"):
        workrb.evaluate_rankings(
            rankings_dir=rankings_dir,
            tasks=tasks,
            output_folder=str(read_output),
        )


def test_writer_header_matches_pinned_schema():
    """The writer must emit exactly the header fields pinned in SCHEMA_HEADER_FIELDS.

    This is the gate against silently changing the on-disk schema. If you add,
    remove, or rename a header field, bump SCHEMA_VERSION and add a new entry
    to SCHEMA_HEADER_FIELDS rather than editing the current one in place.
    """
    write_dir = _fresh_dir("rankings_artifact_pinned_schema")
    rankings_dir = _run_and_get_rankings_dir(write_dir)
    files = list(rankings_dir.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())

    written_fields = frozenset(payload["header"].keys())
    pinned_fields = SCHEMA_HEADER_FIELDS[SCHEMA_VERSION]
    assert written_fields == pinned_fields, (
        f"writer header drift from SCHEMA_VERSION={SCHEMA_VERSION}: "
        f"added={sorted(written_fields - pinned_fields)}, "
        f"removed={sorted(pinned_fields - written_fields)}. "
        "Bump SCHEMA_VERSION and add a new entry in SCHEMA_HEADER_FIELDS."
    )


def test_evaluate_rankings_unknown_schema_version_raises():
    """An unknown schema_version is a hard reject (independent of workrb_version)."""
    write_dir = _fresh_dir("rankings_artifact_schema_write")
    rankings_dir = _run_and_get_rankings_dir(write_dir)
    _hand_edit_header(rankings_dir, schema_version=999)

    tasks = [TinyRankingTask(split=DatasetSplit.TEST, languages=[Language.EN])]
    read_output = _fresh_dir("rankings_artifact_schema_read")
    with pytest.raises(RankingsArtifactInvalid, match="schema_version"):
        workrb.evaluate_rankings(
            rankings_dir=rankings_dir,
            tasks=tasks,
            output_folder=str(read_output),
        )


def test_save_rankings_rejects_non_finite_scores():
    """The writer refuses NaN/inf because JSON cannot represent them safely."""
    import numpy as np

    from workrb.config import BenchmarkConfig

    dataset = TinyRankingTask(
        split=DatasetSplit.TEST, languages=[Language.EN]
    ).datasets["en"]
    config = BenchmarkConfig(
        model_name="tiny",
        output_folder=str(_fresh_dir("rankings_artifact_non_finite_write")),
    )
    matrix = np.array([[0.1, float("nan"), 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        config.save_rankings_artifact(
            task_name="Tiny Ranking Task",
            dataset_id="en",
            split="test",
            dataset=dataset,
            prediction_matrix=matrix,
        )


def test_evaluate_rankings_rejects_non_finite_scores_in_artifact():
    """A hand-edited NaN in the artifact is rejected at load time."""
    write_dir = _fresh_dir("rankings_artifact_non_finite_read")
    rankings_dir = _run_and_get_rankings_dir(write_dir)
    files = list(rankings_dir.glob("*.json"))
    assert len(files) == 1
    # Inject NaN by editing the parsed structure and re-emitting with
    # allow_nan=True. json.load (used by the reader) accepts the non-standard
    # NaN token, which is exactly what validate_header/load_rankings_artifact
    # must catch.
    payload = json.loads(files[0].read_text())
    payload["scores"]["0"]["0"] = float("nan")
    files[0].write_text(json.dumps(payload, allow_nan=True))

    tasks = [TinyRankingTask(split=DatasetSplit.TEST, languages=[Language.EN])]
    read_output = _fresh_dir("rankings_artifact_non_finite_read_out")
    with pytest.raises(RankingsArtifactInvalid, match="non-finite"):
        workrb.evaluate_rankings(
            rankings_dir=rankings_dir,
            tasks=tasks,
            output_folder=str(read_output),
        )


def test_evaluate_rankings_rejects_out_of_bounds_target_index():
    """A target_index outside [0, num_targets) is rejected with IndexError."""
    write_dir = _fresh_dir("rankings_artifact_oob_write")
    rankings_dir = _run_and_get_rankings_dir(write_dir)
    files = list(rankings_dir.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    # num_targets is 3 in TinyRankingTask, so 999 is out of bounds.
    payload["scores"]["0"]["999"] = 0.42
    files[0].write_text(json.dumps(payload))

    tasks = [TinyRankingTask(split=DatasetSplit.TEST, languages=[Language.EN])]
    read_output = _fresh_dir("rankings_artifact_oob_read")
    with pytest.raises(IndexError, match="target_index 999"):
        workrb.evaluate_rankings(
            rankings_dir=rankings_dir,
            tasks=tasks,
            output_folder=str(read_output),
        )


def test_evaluate_rankings_version_mismatch_warns_but_proceeds():
    """workrb_version mismatch only logs a warning; metrics still computed."""
    write_dir = _fresh_dir("rankings_artifact_version_write")
    rankings_dir = _run_and_get_rankings_dir(write_dir)
    _hand_edit_header(rankings_dir, workrb_version="0.0.0-test-fixture")

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    capture = _Capture(level=logging.WARNING)
    rankings_logger = logging.getLogger("workrb.rankings")
    rankings_logger.addHandler(capture)
    try:
        tasks = [TinyRankingTask(split=DatasetSplit.TEST, languages=[Language.EN])]
        read_output = _fresh_dir("rankings_artifact_version_read")
        replay = workrb.evaluate_rankings(
            rankings_dir=rankings_dir,
            tasks=tasks,
            output_folder=str(read_output),
        )
    finally:
        rankings_logger.removeHandler(capture)

    assert any("0.0.0-test-fixture" in r.getMessage() for r in records)
    assert "Tiny Ranking Task" in replay.task_results
