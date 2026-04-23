"""Contains all classes related to fine-tuning pair filtering."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Any
import inspect
import ast
import textwrap
import warnings

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

        return filtered_pairs

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
        model_id: str = "all-MiniLM-L6-v2",
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

            self.model_id = model_id
            self.model = SentenceTransformer(self.model_id)
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


class PredicateFilter(Filter):
    """
    Filter for removing pairs using a custom predicate.

    Example usage for filtering fine-tuning pairs by the combined length of their
    prompt and response:
    ```python
    from tuatara.filtering import PredicateFilter

    threshold = 100
    length_predicate = lambda x: len(x) < threshold
    length_filter = PredicateFilter(combined_predicate=length_predicate)
    ```
    """

    def __init__(
        self,
        prompt_predicate: Callable[[str], bool] | None = None,
        response_predicate: Callable[[str], bool] | None = None,
        combined_predicate: Callable[[str], bool] | None = None,
    ):
        """
        Args:
            prompt_predicate: The filtering predicate applied to the prompt of each
                              fine-tuning pair. If the predicate returns `True` if the
                              pair should be kept, and `False` otherwise.
            response_predicate: The filtering predicate applied to the response of each
                                fine-tuning pair.
            combined_predicate: The filtering predicate applied to the the combined
                                prompt and response.
        """
        self.prompt_predicate = prompt_predicate
        self.response_predicate = response_predicate
        self.combined_predicate = combined_predicate

        # Find the frame in the callstack where the user defined this object
        creation_frame = None

        for frame_info in inspect.stack():
            module = inspect.getmodule(frame_info.frame)
            if module and not "tuatara" in module.__name__:
                creation_frame = frame_info
                break
            
        self._creation_frame_info = creation_frame

    def _filter(self, pairs: list[FineTuningPair]) -> list[FineTuningPair]:
        filtered_pairs = []
        for pair in pairs:
            if self.prompt_predicate and not self.prompt_predicate(pair.prompt):
                continue
            if self.response_predicate and not self.response_predicate(pair.response):
                continue
            if self.combined_predicate and not self.combined_predicate(
                f"{pair.prompt} {pair.response}"
            ):
                continue
            filtered_pairs.append(pair)
        return filtered_pairs
    
    def _capture_source(self, attr_name: str) -> str:
        """
        Find the definition of the lambda and return it as a string.

        Args:
            attr_name: The name of the attribute on this object for which to find the
                       lambda.
        Returns:
            The definition of the lambda corresponding with `attr_name`, as a string.
        """
        try:
            frame = self._creation_frame_info
            call_site_line_no = frame.lineno
            module = inspect.getmodule(frame.frame)
            source = textwrap.dedent(inspect.getsource(module))
            module_source = textwrap.dedent(source)
            
            tree = ast.parse(module_source)

            for node in ast.walk(tree):
                if hasattr(node, "lineno") and node.lineno == call_site_line_no:
                    if isinstance(node, ast.Call) or (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                        call_node = node if isinstance(node, ast.Call) else node.value

                        arg_node = None

                        for kw in call_node.keywords:
                            if kw.arg == attr_name:
                                arg_node = kw.value
                                break

                        if arg_node is None:
                            sig = inspect.signature(self.__init__)
                            params = list(sig.parameters.keys())
                            if attr_name in params:
                                index = params.index(attr_name)
                                if index < len(call_node.args):
                                    arg_node = call_node.args[index]

                        if arg_node is None:
                            return None

                        if isinstance(arg_node, ast.Lambda):
                            return ast.unparse(arg_node)
                        
                        if isinstance(arg_node, ast.Name):
                            var_name = arg_node.id
                            return self._find_assignment(tree, var_name, call_site_line_no)
        except Exception as e:
            warnings.warn(e)
            return None
        
    def _find_assignment(self, tree, var_name, before_lineno):
        """
        Find the most recent assignment of a given variable in an AST.

        Args:
            tree: The AST in which to find the assignment.
            var_name: The name of the variable for which to find the assignment.
            before_lineno: The line number before which the assignment must be located.
        Returns:
            The RHS of the most recent assignment, as a string.
        """
        last_val = None
        best_lineno = -1
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        if best_lineno < node.lineno < before_lineno:
                            best_lineno = node.lineno
                            last_val = ast.unparse(node.value)
        return last_val

    def _build_env(self, lambd: Callable[[str], Any]) -> dict[str, Any]:
        """
        Build a complete environment from the creation frame for this instance.

        Args:
            lambd: The function currently being serialized.
        Returns:
            A dictionary containing the full environment at the site of the lambda
            definition, prioritizing closure variables, then local variables, and
            finally global variables.
        """
        env = {}
        frame = self._creation_frame_info.frame

        closure = getattr(lambd, "__closure__", None)
        freevars = getattr(lambd.__code__, "co_freevars", ())

        if closure and freevars:
            for name, cell in zip(freevars, closure):
                env[name] = cell.cell_contents

        for k, v in frame.f_locals.items():
            if k not in env:
                env[k] = v

        for k, v in frame.f_globals.items():
            if k not in env:
                env[k] = v

        return env

    def _to_config(self):
        cfg = {}
        predicates = list(inspect.signature(self.__init__).parameters.keys())

        for pred in predicates:
            lambd = getattr(self, pred)
            source = self._capture_source(pred)
            if source:
                env = self._build_env(lambd)
                filtered_env = {
                    k: v
                    for k, v in env.items() if k in lambd.__code__.co_names
                }

                cfg[pred] = {
                    "source": source,
                    "env": filtered_env,
                }

        return cfg
    
    @classmethod
    def _from_config(cls, cfg):
        new_cfg = {}
        predicates = list(inspect.signature(cls.__init__).parameters.keys())
        for key, value in cfg.items():
            if key in predicates and "source" in value:
                source = value["source"]
                if isinstance(source, str) and source.strip().startswith("lambda"):
                    env = value.get("env", {})
                    try:
                        new_cfg[key] = eval(source, env)
                    except Exception as e:
                        raise ValueError(f"Failed to evaluate predicate for '{key}: {value}'") from e
        return cls(**new_cfg)
