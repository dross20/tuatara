from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

from loguru import logger

from tuatara.pipeline import PipelineStep

if TYPE_CHECKING:
    from tuatara.pair_generation import FineTuningPair


class Filter(PipelineStep):
    """Abstract class that defines the interface for filters."""

    def forward(self, data: list[FineTuningPair]) -> list[FineTuningPair]:
        logger.debug(f"Filtering {len(data)} fine-tuning pairs")

        filtered_pairs = self._filter(data)

        logger.debug(f"{len(filtered_pairs)} pairs remaining after filtering")

        return self._filter(data)

    @abstractmethod
    def _filter(self, pairs: list[FineTuningPair]) -> list[FineTuningPair]:
        """
        Applies a filter to fine-tuning pairs, returning only those pairs that pass
        some criteria.

        Args:
            pairs: The list of fine-tuning pairs to filter.
        Returns:
            The list of fine-tuning pairs that make it through the filter.
        """
        ...


class SemanticSimilarityFilter(Filter):
    """Filter for removing semantically similar pairs."""

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        representative_strategy: Literal["first", "random", "centroid"] = "first",
        clustering_args: dict | None = None,
    ):
        """
        Args:
            model: The ID of the encoder model to use.
            representative_strategy: The strategy to use for selecting a representative
                                    pair from a cluster.
            clustering_args: The keyword arguments to use when instantiating the
                            `sklearn.cluster.DBSCAN` object. Defaults to
                            `{"metric": "cosine"}`.
        """
        try:
            import numpy as np
            from sklearn.cluster import DBSCAN

            self.dbscan = DBSCAN
            self.np = np
        except ImportError:
            raise ImportError(
                "The `sklearn` library must be installed to use"
                "`SemanticSimilarityFilter.` To install it, run the following command:"
                "`pip install scikit-learn`"
            )
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model)
        except ImportError:
            raise ImportError(
                "The `sentence-transformers` library must be installed to use"
                "`SemanticSimilarityFilter`. To install it, run the following command:"
                "`pip install sentence-transformers`"
            )
        self.representative_strategy = representative_strategy
        self.clustering_args = (
            clustering_args if clustering_args is not None else {"metric": "cosine"}
        )

    def _filter(self, pairs: list[FineTuningPair]) -> list[FineTuningPair]:
        # Embed pairs
        pair_texts = [f"{pair.prompt} {pair.response}" for pair in pairs]
        embeddings = self.model.encode(pair_texts, convert_to_numpy=True)

        # Create a new `DBSCAN` instance
        dbscan = self.dbscan(**self.clustering_args)

        # Cluster pair embeddings
        clustering = dbscan.fit(embeddings)
        labels = clustering.labels_
        unique_labels = set(labels)

        # Handle noise points (unique pairs)
        noise_mask = labels == -1
        noise_indices = self.np.where(noise_mask)[0]

        filtered_pairs = [pairs[index] for index in noise_indices]

        if -1 in unique_labels:
            unique_labels.remove(-1)

        # Handle clustered points (similar pairs)
        for label in unique_labels:
            label_mask = labels == label
            label_indices = self.np.where(label_mask)[0]

            # Choose first pair in the cluster to be the representative
            if self.representative_strategy == "first":
                representative = pairs[label_indices[0]]
            # Choose representative arbitrarily from within the cluster
            elif self.representative_strategy == "random":
                random_index = self.np.random.choice(label_indices)
                representative = pairs[random_index]
            # Choose representative closest to the cluster's centroid
            elif self.representative_strategy == "centroid":
                cluster_embeddings = embeddings[label_indices]
                centroid = self.np.mean(cluster_embeddings, axis=0)
                distances = self.np.linalg.norm(cluster_embeddings - centroid, axis=1)
                closest_index = label_indices[self.np.argmin(distances)]
                representative = pairs[closest_index]

            filtered_pairs.append(representative)

        return filtered_pairs


class NLISourceGroundingFilter(Filter):
    """Filter for removing pairs without grounding in their source chunks."""

    def __init__(
        self,
        model: str = "cross-encoder/nli-deberta-v3-base",
        entailment_threshold: float = 0.5,
    ):
        """
        Args:
            model: The ID of the NLI model to use.
            entailment_threshold: The minimum required predicted entailment score for a
                                  pair to be considered sufficiently grounded.
        """
        try:
            from sentence_transformers import CrossEncoder

            self.cross_encoder = CrossEncoder(model)
        except ImportError:
            raise ImportError(
                "The `sentence-transformers` library must be installed to use"
                "`NLISourceGroundingFilter`. To install it, run the following command:"
                "`pip install sentence-transformers`"
            )
        self.entailment_threshold = entailment_threshold

    def _filter(self, pairs: list[FineTuningPair]) -> list[FineTuningPair]:
        pair_texts = [f"{pair.prompt} {pair.response}" for pair in pairs]
        sources = [" ".join(pair.source_chunks) for pair in pairs]
        grounding_inputs = list(zip(pair_texts, sources))

        scores = self.cross_encoder.predict(grounding_inputs)

        # The predicted entailment probability is stored in axis 1, index 1 of
        # `CrossEncoder.predict`'s return array
        entailment_scores = scores[:, 1]

        filtered_pairs = [
            pair
            for pair, score in zip(pairs, entailment_scores)
            if score >= self.entailment_threshold
        ]

        return filtered_pairs
