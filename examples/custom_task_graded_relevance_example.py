"""
Custom Graded-Relevance Ranking Task

Extends ``custom_task_example.py`` with graded relevance: each (query, target)
pair carries an ordinal grade instead of a binary relevant/not-relevant flag.

How a graded task differs from a binary one:

1. Pass an additional ``target_relevance`` list to ``RankingDataset``, aligned
   1-to-1 with ``target_indices``. Items NOT listed in ``target_indices`` are
   implicitly grade 0 (irrelevant).
2. Add ``ndcg@k`` to ``default_metrics``. The (2^rel - 1) gain in nDCG uses the
   grades.
3. Optionally override ``binary_relevance_threshold`` on the task to control
   which graded items count as positives for binary metrics (``map``, ``mrr``,
   ``recall@k``, ``hit@k``, ``rp@k``). Items with relevance >= threshold are
   positives; items below are dropped from the binary positive set but still
   contribute to nDCG. Default is ``1e-9``, which keeps every listed item as
   a positive (matches the binary-only behavior).

The grading scale is up to the task. Common choices: {1, 2, 3} (this example),
{1, 2, 3, 4} (TREC-style), or fractional in [0, 1]. Values must be non-negative.
"""

import workrb
from workrb.registry import register_task
from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTaskGroup
from workrb.types import ModelInputType


@register_task()
class GradedJob2SkillTask(workrb.tasks.RankingTask):
    """Job-to-skill ranking with three relevance grades.

    Grade 3 — primary skill (the role is defined by it).
    Grade 2 — secondary skill (clearly expected).
    Grade 1 — nice-to-have (mentioned but not required).
    Grade 0 — irrelevant; never appears in ``target_indices`` (implicit).
    """

    @property
    def name(self) -> str:
        return "GradedJob2SkillTask"

    @property
    def description(self) -> str:
        return "Job-to-skill ranking with graded relevance for nDCG evaluation."

    @property
    def query_input_type(self) -> ModelInputType:
        return ModelInputType.JOB_TITLE

    @property
    def target_input_type(self) -> ModelInputType:
        return ModelInputType.SKILL

    @property
    def default_metrics(self) -> list[str]:
        # nDCG@k is the headline metric here. MAP/MRR/recall are kept as a
        # binary sanity check on the high-grade subset (see the threshold
        # below): they treat every passing graded positive as relevant=1.
        return ["ndcg@5", "ndcg@10", "map", "mrr", "recall@5"]

    @property
    def binary_relevance_threshold(self) -> float:
        # Only "secondary" (grade 2) and "primary" (grade 3) skills count as
        # positives for binary metrics; "nice-to-have" (grade 1) items drop
        # out of MAP/MRR/recall but still contribute to nDCG.
        return 2.0

    @property
    def task_group(self) -> RankingTaskGroup:
        return RankingTaskGroup.JOB2SKILL

    @property
    def label_type(self) -> LabelType:
        return LabelType.MULTI_LABEL

    @property
    def supported_query_languages(self) -> list[Language]:
        return [Language.EN]

    @property
    def supported_target_languages(self) -> list[Language]:
        return [Language.EN]

    def load_dataset(self, dataset_id: str, split: DatasetSplit) -> RankingDataset:
        """Load a tiny in-memory graded dataset.

        target_indices and target_relevance are aligned 1-to-1: position i in
        ``target_relevance[q]`` is the grade for ``target_indices[q][i]``.
        """
        queries = [
            "Machine learning engineer",
            "Data scientist",
            "Backend software developer",
        ]

        targets = [
            "Python programming",  # 0
            "Machine learning",  # 1
            "Data analysis",  # 2
            "Software engineering",  # 3
            "Statistics",  # 4
            "Deep learning",  # 5
            "Web development",  # 6
            "Database management",  # 7
        ]

        # Each list pairs (skill_idx -> grade). Skills not listed for a query
        # are implicitly grade 0.
        target_indices = [
            [1, 5, 0, 3],  # ML engineer: ML(3), DL(3), Python(2), SWE(1)
            [1, 2, 4, 5],  # Data scientist: ML(3), DA(3), Stats(2), DL(2)
            [0, 3, 7, 6],  # Backend dev: Python(3), SWE(3), DB(2), Web(1)
        ]
        target_relevance = [
            [3.0, 3.0, 2.0, 1.0],
            [3.0, 3.0, 2.0, 2.0],
            [3.0, 3.0, 2.0, 1.0],
        ]

        return RankingDataset(
            query_texts=queries,
            target_indices=target_indices,
            target_space=targets,
            dataset_id=dataset_id,
            target_relevance=target_relevance,
        )


if __name__ == "__main__":
    print("🚀 Custom Graded-Relevance Task Example")
    print("=" * 50)

    model = workrb.models.BiEncoderModel("all-MiniLM-L6-v2")
    tasks = [GradedJob2SkillTask(languages=["en"], split="test")]

    # Run with the task's default_metrics, which mixes nDCG (graded) with
    # MAP/MRR/recall (binary). The same dataset feeds both — graded metrics
    # consult target_relevance, binary metrics ignore it.
    results = workrb.evaluate(
        model,
        tasks,
        output_folder="results/graded_task_demo",
        description="Custom ranking task with graded relevance.",
        force_restart=True,
    )
    print(results)
