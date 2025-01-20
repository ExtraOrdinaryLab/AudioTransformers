import os
import csv
from contextlib import nullcontext
from typing import List, Optional, Union, Literal

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances

from .. import logging, AudioTransformer
from .AudioEvaluator import AudioEvaluator
from ..similarity_functions import SimilarityFunction

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class EmbeddingSimilarityEvaluator(AudioEvaluator):

    def __init__(
        self, 
        audio_1: List[str], 
        audio_2: List[str], 
        scores: List[float], 
        batch_size: int = 16, 
        main_similarity: Union[str, SimilarityFunction] = None, 
        similarity_fn_names: List[Literal["cosine", "euclidean", "manhattan", "dot"]] = None,
        name: str = "",
        show_progress_bar: bool = False,
        write_csv: bool = True,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = None,
        truncate_dim: int = None,
    ):
        super().__init__()
        self.audio_1 = audio_1
        self.audio_2 = audio_2
        self.scores = scores
        self.write_csv = write_csv
        self.precision = precision
        self.truncate_dim = truncate_dim

        assert len(self.audio_1) == len(self.audio_2)
        assert len(self.audio_1) == len(self.scores)

        self.main_similarity = SimilarityFunction(main_similarity) if main_similarity else None
        self.similarity_fn_names = similarity_fn_names or []
        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file = (
            "similarity_evaluation"
            + ("_" + name if name else "")
            + ("_" + precision if precision else "")
            + "_results.csv"
        )
        self.csv_headers = [
            "epoch",
            "steps",
        ]

        self._append_csv_headers(self.similarity_fn_names)

    def _append_csv_headers(self, similarity_fn_names: List[str]) -> None:
        metrics = ["pearson", "spearman"]

        for v in similarity_fn_names:
            for m in metrics:
                self.csv_headers.append(f"{v}_{m}")

    def __call__(
        self, model: AudioTransformer, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(f"EmbeddingSimilarityEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")

        with nullcontext() if self.truncate_dim is None else model.truncate_audio_embeddings(self.truncate_dim):
            embeddings1 = model.encode(
                self.audio_1,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
                precision=self.precision,
                normalize_embeddings=bool(self.precision),
            )
            embeddings2 = model.encode(
                self.audio_2,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
                precision=self.precision,
                normalize_embeddings=bool(self.precision),
            )
        # Binary and ubinary embeddings are packed, so we need to unpack them for the distance metrics
        if self.precision == "binary":
            embeddings1 = (embeddings1 + 128).astype(np.uint8)
            embeddings2 = (embeddings2 + 128).astype(np.uint8)
        if self.precision in ("ubinary", "binary"):
            embeddings1 = np.unpackbits(embeddings1, axis=1)
            embeddings2 = np.unpackbits(embeddings2, axis=1)

        labels = self.scores

        if not self.similarity_fn_names:
            self.similarity_fn_names = [model.similarity_fn_name]
            self._append_csv_headers(self.similarity_fn_names)

        similarity_functions = {
            "cosine": lambda x, y: 1 - paired_cosine_distances(x, y),
            "manhattan": lambda x, y: -paired_manhattan_distances(x, y),
            "euclidean": lambda x, y: -paired_euclidean_distances(x, y),
            "dot": lambda x, y: [np.dot(emb1, emb2) for emb1, emb2 in zip(x, y)],
        }

        metrics = {}
        for fn_name in self.similarity_fn_names:
            if fn_name in similarity_functions:
                scores = similarity_functions[fn_name](embeddings1, embeddings2)
                eval_pearson, _ = pearsonr(labels, scores)
                eval_spearman, _ = spearmanr(labels, scores)
                metrics[f"pearson_{fn_name}"] = eval_pearson
                metrics[f"spearman_{fn_name}"] = eval_spearman
                logger.info(
                    f"{fn_name.capitalize()}-Similarity :\tPearson: {eval_pearson:.4f}\tSpearman: {eval_spearman:.4f}"
                )

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline="", mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow(
                    [
                        epoch,
                        steps,
                    ]
                    + [
                        metrics[f"{metric}_{fn_name}"]
                        for fn_name in self.similarity_fn_names
                        for metric in ["pearson", "spearman"]
                    ]
                )

        if len(self.similarity_fn_names) > 1:
            metrics["pearson_max"] = max(metrics[f"pearson_{fn_name}"] for fn_name in self.similarity_fn_names)
            metrics["spearman_max"] = max(metrics[f"spearman_{fn_name}"] for fn_name in self.similarity_fn_names)

        if self.main_similarity:
            self.primary_metric = {
                SimilarityFunction.COSINE: "spearman_cosine",
                SimilarityFunction.EUCLIDEAN: "spearman_euclidean",
                SimilarityFunction.MANHATTAN: "spearman_manhattan",
                SimilarityFunction.DOT_PRODUCT: "spearman_dot",
            }.get(self.main_similarity)
        else:
            if len(self.similarity_fn_names) > 1:
                self.primary_metric = "spearman_max"
            else:
                self.primary_metric = f"spearman_{self.similarity_fn_names[0]}"

        metrics = self.prefix_name_to_metrics(metrics, self.name)
        # self.store_metrics_in_model_card_data(model, metrics)
        return metrics

    @property
    def description(self) -> str:
        return "Semantic Similarity"