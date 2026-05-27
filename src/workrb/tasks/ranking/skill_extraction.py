"""Skill Extraction Ranking Tasks."""

import pandas as pd
from datasets import Dataset, load_dataset

from workrb.data.esco import ESCO
from workrb.registry import register_task
from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language
from workrb.tasks.abstract.ranking_base import (
    RankingDataset,
    RankingTask,
    RankingTaskGroup,
)
from workrb.types import ModelInputType


class BaseESCOSkillExtractRanking(RankingTask):
    """Base ESCO Skill Extraction Ranking Task."""

    def __init__(
        self,
        hf_name: str,
        esco_version: str = "1.1.0",
        orig_esco_version: str = "1.1.0",
        **kwargs,
    ):
        """
        Initialize ESCO Skill Extraction Ranking Task.

        Args:
            hf_name: Name of the Hugging Face dataset
            esco_version: Target version of ESCO to use
            orig_esco_version: Version of ESCO used for tagging the original data
            **kwargs: Additional arguments for base class
        """
        self.esco_version = esco_version
        self.hf_name = hf_name
        self.orig_esco_version = orig_esco_version
        super().__init__(**kwargs)

    @property
    def task_group(self) -> RankingTaskGroup:
        """Skill extraction house task group."""
        return RankingTaskGroup.SKILL_EXTRACTION

    @property
    def supported_query_languages(self) -> list[Language]:
        """Supported query languages are always English."""
        return [Language.EN]

    @property
    def supported_target_languages(self) -> list[Language]:
        """Supported target languages for the configured ESCO version."""
        return list(ESCO.get_supported_languages(self.esco_version))

    @property
    def label_type(self) -> LabelType:
        """Label type is multi-label."""
        return LabelType.MULTI_LABEL

    @property
    def query_input_type(self) -> ModelInputType:
        """Query input type for skill extraction sentences."""
        return ModelInputType.SKILL_SENTENCE

    @property
    def target_input_type(self) -> ModelInputType:
        """Target input type for ESCO skills."""
        return ModelInputType.SKILL_NAME

    def load_dataset(self, dataset_id: str, split: DatasetSplit) -> RankingDataset:
        """Load skill extraction data for a specific split and dataset.

        Args:
            dataset_id: Dataset identifier (language code for this task)
            split: Dataset split to load

        Returns
        -------
            RankingDataset object
        """
        language = Language(dataset_id)
        # Load data
        split_names = {DatasetSplit.TEST: "test", DatasetSplit.VAL: "validation"}
        dataset = load_dataset(self.hf_name, split=split_names[split])
        assert isinstance(dataset, Dataset)
        df = dataset.to_pandas()
        assert isinstance(df, pd.DataFrame)

        # If ESCO version is not 1.1.0 and / or language is not en, we need to translate the skills
        if self.esco_version != self.orig_esco_version or language != Language.EN:
            original_esco = ESCO(version=self.orig_esco_version, language=Language.EN)
            original_skill_uris = original_esco.get_skills_uris()
            original_uris_to_skill = {v: k for k, v in original_skill_uris.items()}

            target_esco = ESCO(version=self.esco_version, language=language)
            target_skill_uris = target_esco.get_skills_uris()
            target_uris_to_skill = {v: k for k, v in target_skill_uris.items()}

            original_skill_to_target_skill = {}
            for uri, orig_skill in original_uris_to_skill.items():
                if uri in target_uris_to_skill:
                    original_skill_to_target_skill[orig_skill] = target_uris_to_skill[uri]

            df["label"] = df["label"].apply(original_skill_to_target_skill.get)
            # Drop rows where label is None
            df = df[df["label"].notna()].reset_index(drop=True).copy()

        grouped_df = df.groupby("sentence")["label"].apply(list).reset_index()

        # Load ESCO skill vocabulary for target version/language
        esco = ESCO(version=self.esco_version, language=language)
        skill_vocab = esco.get_skills_vocabulary()
        skill2label = {skill: i for i, skill in enumerate(skill_vocab)}

        # Filter skills that exist in vocabulary (Excludes "LABEL NOT PRESENT" and "UNDERSPECIFIED")
        filtered_queries = []
        filtered_labels = []
        for query, skill_list in zip(grouped_df["sentence"], grouped_df["label"], strict=True):
            filtered_skill_list = [skill for skill in skill_list if skill in skill2label]
            if len(filtered_skill_list) == 0:
                continue
            filtered_queries.append(query)
            filtered_labels.append([skill2label[skill] for skill in filtered_skill_list])

        return RankingDataset(
            query_texts=filtered_queries,
            target_indices=filtered_labels,
            target_space=skill_vocab,
            dataset_id=dataset_id,
        )


@register_task()
class HouseSkillExtractRanking(BaseESCOSkillExtractRanking):
    """Skill Extraction from House Dataset Ranking Task."""

    orig_esco_version = "1.1.0"

    def __init__(self, esco_version: str = "1.1.0", **kwargs):
        self.esco_version = esco_version
        super().__init__(hf_name="TechWolf/skill-extraction-house", **kwargs)

    @property
    def name(self) -> str:
        """Skill extraction house task name."""
        return "Skill Extraction House"

    @property
    def description(self) -> str:
        """Skill extraction house task description."""
        return "Extract skills from general text descriptions in the HOUSE subset of CAREER."

    @property
    def citation(self) -> str:
        """Skill extraction house task citation."""
        return """@inproceedings{decorte2022design,
  articleno    = {{4}},
  author       = {{Decorte, Jens-Joris and Van Hautte, Jeroen and Deleu, Johannes and Develder, Chris and Demeester, Thomas}},
  booktitle    = {{Proceedings of the 2nd Workshop on Recommender Systems for Human Resources (RecSys-in-HR 2022)}},
  editor       = {{Kaya, Mesut and Bogers, Toine and Graus, David and Mesbah, Sepideh and Johnson, Chris and Gutiérrez, Francisco}},
  isbn         = {{9781450398565}},
  issn         = {{1613-0073}},
  language     = {{eng}},
  location     = {{Seatle, USA}},
  pages        = {{7}},
  publisher    = {{CEUR}},
  title        = {{Design of negative sampling strategies for distantly supervised skill extraction}},
  url          = {{https://ceur-ws.org/Vol-3218/RecSysHR2022-paper_4.pdf}},
  volume       = {{3218}},
  year         = {{2022}},
}
"""


@register_task()
class TechSkillExtractRanking(BaseESCOSkillExtractRanking):
    """Skill Extraction from Tech Dataset Ranking Task."""

    orig_esco_version = "1.1.0"

    def __init__(self, esco_version: str = "1.1.0", **kwargs):
        self.esco_version = esco_version
        super().__init__(hf_name="TechWolf/skill-extraction-tech", **kwargs)

    @property
    def name(self) -> str:
        """Skill extraction tech task name."""
        return "Skill Extraction Tech"

    @property
    def description(self) -> str:
        """Skill extraction tech task description."""
        return "Extract skills from technical text descriptions in the TECH subset of CAREER."

    @property
    def citation(self) -> str:
        """Skill extraction tech task citation."""
        return """@inproceedings{decorte2022design,
  articleno    = {{4}},
  author       = {{Decorte, Jens-Joris and Van Hautte, Jeroen and Deleu, Johannes and Develder, Chris and Demeester, Thomas}},
  booktitle    = {{Proceedings of the 2nd Workshop on Recommender Systems for Human Resources (RecSys-in-HR 2022)}},
  editor       = {{Kaya, Mesut and Bogers, Toine and Graus, David and Mesbah, Sepideh and Johnson, Chris and Gutiérrez, Francisco}},
  isbn         = {{9781450398565}},
  issn         = {{1613-0073}},
  language     = {{eng}},
  location     = {{Seatle, USA}},
  pages        = {{7}},
  publisher    = {{CEUR}},
  title        = {{Design of negative sampling strategies for distantly supervised skill extraction}},
  url          = {{https://ceur-ws.org/Vol-3218/RecSysHR2022-paper_4.pdf}},
  volume       = {{3218}},
  year         = {{2022}},
}
"""


@register_task()
class TechWolfSkillExtractRanking(BaseESCOSkillExtractRanking):
    """Skill Extraction from TechWolf Dataset Ranking Task."""

    orig_esco_version = "1.1.0"

    def __init__(self, esco_version: str = "1.1.0", **kwargs):
        self.esco_version = esco_version
        super().__init__(hf_name="TechWolf/skill-extraction-techwolf", **kwargs)

    @property
    def name(self) -> str:
        """Skill extraction TechWolf task name."""
        return "Skill Extraction TechWolf"

    @property
    def description(self) -> str:
        """Skill extraction TechWolf task description."""
        return (
            "Extract skills from text descriptions in a generic distribution of job descriptions."
        )

    @property
    def citation(self) -> str:
        """Skill extraction TechWolf task citation."""
        return """@article{decorte2023extreme,
  title={Extreme multi-label skill extraction training using large language models},
  author={Decorte, Jens-Joris and Verlinden, Severine and Van Hautte, Jeroen and Deleu, Johannes and Develder, Chris and Demeester, Thomas},
  journal={arXiv preprint arXiv:2307.10778},
  year={2023}
}
"""


@register_task()
class SkillXLSkillExtractRanking(BaseESCOSkillExtractRanking):
    """Skill Extraction from SkillXL Dataset Ranking Task."""

    orig_esco_version = "1.1.0"

    def __init__(self, esco_version: str = "1.1.0", **kwargs):
        self.esco_version = esco_version
        super().__init__(hf_name="TechWolf/Skill-XL", **kwargs)

    def load_dataset(self, dataset_id: str, split: DatasetSplit) -> RankingDataset:
        """Load SkillXL data, filtering to relevant rows.

        SkillXL differs from the base class in two ways: it includes
        irrelevant sentences (filtered via the ``relevant`` boolean column),
        and the skill column is named ``skill`` instead of ``label``.
        """
        language = Language(dataset_id)
        # Load data
        split_names = {DatasetSplit.TEST: "test", DatasetSplit.VAL: "validation"}
        dataset = load_dataset(self.hf_name, split=split_names[split])
        assert isinstance(dataset, Dataset)
        df = dataset.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert "relevant" in df.columns, "Expected 'relevant' column in the dataset."

        # Only keep rows where relevant is True
        df = df[df["relevant"]].reset_index(drop=True).copy()
        assert isinstance(df, pd.DataFrame)

        # If ESCO version is not 1.1.0 and / or language is not en, we need to translate the skills
        if self.esco_version != self.orig_esco_version or language != Language.EN:
            original_esco = ESCO(version=self.orig_esco_version, language=Language.EN)
            original_skill_uris = original_esco.get_skills_uris()
            original_uris_to_skill = {v: k for k, v in original_skill_uris.items()}

            target_esco = ESCO(version=self.esco_version, language=language)
            target_skill_uris = target_esco.get_skills_uris()
            target_uris_to_skill = {v: k for k, v in target_skill_uris.items()}

            original_skill_to_target_skill = {}
            for uri, orig_skill in original_uris_to_skill.items():
                if uri in target_uris_to_skill:
                    original_skill_to_target_skill[orig_skill] = target_uris_to_skill[uri]

            df["skill"] = df["skill"].apply(original_skill_to_target_skill.get)
            # Drop rows where skill is None
            df = df[df["skill"].notna()].reset_index(drop=True).copy()

        grouped_df = df.groupby("sentence")["skill"].apply(list).reset_index()

        # Load ESCO skill vocabulary for target version/language
        esco = ESCO(version=self.esco_version, language=language)
        skill_vocab = esco.get_skills_vocabulary()
        skill2label = {skill: i for i, skill in enumerate(skill_vocab)}

        # Filter skills that exist in vocabulary (Excludes "LABEL NOT PRESENT" and "UNDERSPECIFIED")
        filtered_queries = []
        filtered_labels = []
        for query, skill_list in zip(grouped_df["sentence"], grouped_df["skill"], strict=True):
            filtered_skill_list = [skill for skill in skill_list if skill in skill2label]
            if len(filtered_skill_list) == 0:
                continue
            filtered_queries.append(query)
            filtered_labels.append([skill2label[skill] for skill in filtered_skill_list])

        return RankingDataset(
            query_texts=filtered_queries,
            target_indices=filtered_labels,
            target_space=skill_vocab,
            dataset_id=dataset_id,
        )

    @property
    def name(self) -> str:
        """Skill extraction SkillXL task name."""
        return "Skill Extraction SkillXL"

    @property
    def description(self) -> str:
        """Skill extraction SkillXL task description."""
        return (
            "Extract skills from text descriptions in a generic distribution of job descriptions."
        )

    @property
    def citation(self) -> str:
        """Skill extraction SkillXL task citation."""
        return """@ARTICLE{contextmatch_2025,
  author={Decorte, Jens-Joris and van Hautte, Jeroen and Develder, Chris and Demeester, Thomas},
  journal={IEEE Access},
  title={Efficient Text Encoders for Labor Market Analysis},
  year={2025},
  volume={13},
  number={},
  pages={133596-133608},
  keywords={Taxonomy;Contrastive learning;Training;Annotations;Benchmark testing;Training data;Large language models;Computational efficiency;Accuracy;Terminology;Labor market analysis;text encoders;skill extraction;job title normalization},
  doi={10.1109/ACCESS.2025.3589147}
}
"""


class BaseGradedSkillExtractRanking(BaseESCOSkillExtractRanking):
    """Base class for BEIR-layout graded skill-extraction tasks.

    Reads the ``queries``, ``corpus`` and ``qrels`` configs published on the
    Hugging Face Hub via ``load_dataset``. The target_space is the corpus's
    ``title`` column (ESCO preferred labels, in corpus order); qrels are
    expected to contain only non-zero judgments (absent items are implicit
    grade 0). Only the validation split is released at this stage; the test
    split will be released alongside the workrb.ai challenge results.

    Subclasses set ``hf_name`` via ``__init__``.
    """

    @property
    def default_metrics(self) -> list[str]:
        """Default metrics include nDCG to leverage the graded labels.

        ``ndcg`` without a cutoff scores the full ranking (k = |target_space|).
        """
        return ["ndcg", "ndcg@5", "ndcg@10", "map", "rp@10", "mrr"]

    def load_dataset(self, dataset_id: str, split: DatasetSplit) -> RankingDataset:
        """Load BEIR-style graded annotations and convert to a RankingDataset."""
        if split != DatasetSplit.VAL:
            raise ValueError(
                f"Split '{split}' not supported for {type(self).__name__}: only the "
                f"validation split is annotated at this stage."
            )

        queries_ds = load_dataset(self.hf_name, "queries", split="validation")
        corpus_ds = load_dataset(self.hf_name, "corpus", split="corpus")
        qrels_ds = load_dataset(self.hf_name, "qrels", split="validation")
        assert isinstance(queries_ds, Dataset)
        assert isinstance(corpus_ds, Dataset)
        assert isinstance(qrels_ds, Dataset)
        queries_df = queries_ds.to_pandas()
        corpus_df = corpus_ds.to_pandas()
        qrels_df = qrels_ds.to_pandas()

        # target_space is the corpus titles in corpus order; URIs map by row index.
        target_space = corpus_df["title"].tolist()
        uri_to_idx = {uri: i for i, uri in enumerate(corpus_df["_id"])}

        qrels_df["target_idx"] = qrels_df["corpus-id"].map(uri_to_idx)
        # Every qrel should resolve; if any don't, surface the issue rather than silently dropping.
        unresolved = qrels_df["target_idx"].isna().sum()
        assert unresolved == 0, (
            f"{unresolved} qrel rows reference corpus-ids not present in the corpus config"
        )
        qrels_df["target_idx"] = qrels_df["target_idx"].astype(int)

        id_to_query = dict(zip(queries_df["_id"], queries_df["text"], strict=True))
        qrels_df["sentence"] = qrels_df["query-id"].map(id_to_query)

        grouped = qrels_df.groupby("sentence")
        filtered_queries: list[str] = []
        filtered_indices: list[list[int]] = []
        filtered_relevance: list[list[float]] = []
        for sentence, group in grouped:
            filtered_queries.append(str(sentence))
            filtered_indices.append(group["target_idx"].tolist())
            filtered_relevance.append([float(s) for s in group["score"].tolist()])

        return RankingDataset(
            query_texts=filtered_queries,
            target_indices=filtered_indices,
            target_space=target_space,
            dataset_id=dataset_id,
            target_relevance=filtered_relevance,
        )


@register_task()
class HouseGradedSkillExtractRanking(BaseGradedSkillExtractRanking):
    """Skill Extraction from HOUSE Dataset with Graded Relevance.

    Re-annotates the sentences from ``TechWolf/skill-extraction-house`` against
    the full ESCO v1.1.0 skill taxonomy with a 0-4 relevance scale, following the
    BEIR layout (``queries``, ``corpus``, ``qrels``). Score scale: 0 unrelated,
    1 plausible in domain, 2 recommendable but off-granularity, 3 strongly implied,
    4 explicitly demonstrated.
    """

    def __init__(self, **kwargs):
        super().__init__(hf_name="TechWolf/skill-extraction-house-with-relevancy", **kwargs)

    @property
    def name(self) -> str:
        """Skill extraction HOUSE graded task name."""
        return "Skill Extraction House Graded"

    @property
    def description(self) -> str:
        """Skill extraction HOUSE graded task description."""
        return (
            "Extract skills from general text descriptions in the HOUSE subset of CAREER, "
            "with graded 0-4 relevance against the full ESCO v1.1.0 taxonomy."
        )

    @property
    def citation(self) -> str:
        """Skill extraction HOUSE graded task citation."""
        return """@inproceedings{decorte2022design,
  articleno    = {{4}},
  author       = {{Decorte, Jens-Joris and Van Hautte, Jeroen and Deleu, Johannes and Develder, Chris and Demeester, Thomas}},
  booktitle    = {{Proceedings of the 2nd Workshop on Recommender Systems for Human Resources (RecSys-in-HR 2022)}},
  editor       = {{Kaya, Mesut and Bogers, Toine and Graus, David and Mesbah, Sepideh and Johnson, Chris and Gutiérrez, Francisco}},
  isbn         = {{9781450398565}},
  issn         = {{1613-0073}},
  language     = {{eng}},
  location     = {{Seatle, USA}},
  pages        = {{7}},
  publisher    = {{CEUR}},
  title        = {{Design of negative sampling strategies for distantly supervised skill extraction}},
  url          = {{https://ceur-ws.org/Vol-3218/RecSysHR2022-paper_4.pdf}},
  volume       = {{3218}},
  year         = {{2022}},
}
"""


@register_task()
class TechGradedSkillExtractRanking(BaseGradedSkillExtractRanking):
    """Skill Extraction from TECH Dataset with Graded Relevance.

    Re-annotates the sentences from ``TechWolf/skill-extraction-tech`` against
    the full ESCO v1.1.0 skill taxonomy with a 0-4 relevance scale, following the
    BEIR layout (``queries``, ``corpus``, ``qrels``). Score scale: 0 unrelated,
    1 plausible in domain, 2 recommendable but off-granularity, 3 strongly implied,
    4 explicitly demonstrated.
    """

    def __init__(self, **kwargs):
        super().__init__(hf_name="TechWolf/skill-extraction-tech-with-relevancy", **kwargs)

    @property
    def name(self) -> str:
        """Skill extraction TECH graded task name."""
        return "Skill Extraction Tech Graded"

    @property
    def description(self) -> str:
        """Skill extraction TECH graded task description."""
        return (
            "Extract skills from technical text descriptions in the TECH subset of CAREER, "
            "with graded 0-4 relevance against the full ESCO v1.1.0 taxonomy."
        )

    @property
    def citation(self) -> str:
        """Skill extraction TECH graded task citation."""
        return """@inproceedings{decorte2022design,
  articleno    = {{4}},
  author       = {{Decorte, Jens-Joris and Van Hautte, Jeroen and Deleu, Johannes and Develder, Chris and Demeester, Thomas}},
  booktitle    = {{Proceedings of the 2nd Workshop on Recommender Systems for Human Resources (RecSys-in-HR 2022)}},
  editor       = {{Kaya, Mesut and Bogers, Toine and Graus, David and Mesbah, Sepideh and Johnson, Chris and Gutiérrez, Francisco}},
  isbn         = {{9781450398565}},
  issn         = {{1613-0073}},
  language     = {{eng}},
  location     = {{Seatle, USA}},
  pages        = {{7}},
  publisher    = {{CEUR}},
  title        = {{Design of negative sampling strategies for distantly supervised skill extraction}},
  url          = {{https://ceur-ws.org/Vol-3218/RecSysHR2022-paper_4.pdf}},
  volume       = {{3218}},
  year         = {{2022}},
}
"""


@register_task()
class SkillSkapeExtractRanking(BaseESCOSkillExtractRanking):
    """Skill Extraction from SkillSkape Ranking Task."""

    orig_esco_version = "1.1.0"

    def __init__(self, esco_version: str = "1.1.0", **kwargs):
        self.esco_version = esco_version
        super().__init__(hf_name="jjzha/skillskape", **kwargs)

    @property
    def name(self) -> str:
        """Skill extraction SkillSkape task name."""
        return "Skill Extraction SkillSkape"

    @property
    def description(self) -> str:
        """Skill extraction from SkillSkape task description."""
        return "Extract skills from text descriptions."

    @property
    def citation(self) -> str:
        """Skill extraction SkillSkape task citation."""
        return """@inproceedings{magron-etal-2024-jobskape,
  title     = {{JobSkape: A Framework for Generating Synthetic Job Postings to Enhance Skill Matching}},
  author    = {{Magron, Antoine and Dai, Anna and Zhang, Mike and Montariol, Syrielle and Bosselut, Antoine}},
  editor    = {{Hruschka, Estevam and Lake, Thom and Otani, Naoki and Mitchell, Tom}},
  booktitle = {{Proceedings of the First Workshop on Natural Language Processing for Human Resources (NLP4HR 2024)}},
  month     = {{mar}},
  year      = {{2024}},
  address   = {{St. Julian's, Malta}},
  publisher = {{Association for Computational Linguistics}},
  url       = {{https://aclanthology.org/2024.nlp4hr-1.4/}},
  pages     = {{43--58}}
}
"""
