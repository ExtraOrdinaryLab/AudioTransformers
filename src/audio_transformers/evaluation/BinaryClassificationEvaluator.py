import csv
import os
from contextlib import nullcontext
from typing import Literal, Union, List, Dict

import numpy as np
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances

from .. import logging
from .. import AudioTransformer
from .AudioEvaluator import AudioEvaluator
from ..similarity_functions import SimilarityFunction

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class BinaryClassificationEvaluator(AudioEvaluator):

    def __init__(
        self,
        audio_1: List[str],
        audio_2: List[str],
        labels: List[int],
        name: str = "",
        batch_size: int = 32,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        truncate_dim: int = None,
        similarity_fn_names: List[Literal["cosine", "dot", "euclidean", "manhattan"]] = None,
    ):
        super().__init__()
        self.audio_1 = audio_1
        self.audio_2 = audio_2
        self.labels = labels
        self.truncate_dim = truncate_dim
        self.similarity_fn_names = similarity_fn_names or []

        assert len(self.audio_1) == len(self.audio_2)
        assert len(self.audio_2) == len(self.labels)

        for label in labels:
            assert label == 0 or label == 1

        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file = "binary_classification_evaluation" + ("_" + name if name else "") + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        self._append_csv_headers(self.similarity_fn_names)

    def _append_csv_headers(self, similarity_fn_names: list[str]) -> None:
        metrics = [
            "accuracy",
            "accuracy_threshold",
            "f1",
            "precision",
            "recall",
            "f1_threshold",
            "ap",
            "mcc",
        ]

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

        logger.info(f"Binary Accuracy Evaluation of the model on the {self.name} dataset{out_txt}:")

        if not self.similarity_fn_names:
            self.similarity_fn_names = [model.similarity_fn_name]
            self._append_csv_headers(self.similarity_fn_names)
        scores = self.compute_metrices(model)

        file_output_data = [epoch, steps]

        for header_name in self.csv_headers:
            if "_" in header_name:
                sim_fct, metric = header_name.split("_", maxsplit=1)
                file_output_data.append(scores[sim_fct][metric])

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow(file_output_data)
            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(file_output_data)

        metrics = {
            f"{short_name}_{metric}": value
            for short_name, values in scores.items()
            for metric, value in values.items()
        }
        if len(self.similarity_fn_names) > 1:
            metrics.update(
                {
                    f"max_{metric}": max(scores[short_name][metric] for short_name in scores)
                    for metric in scores["cosine"]
                }
            )
            self.primary_metric = "max_ap"
        else:
            self.primary_metric = f"{self.similarity_fn_names[0]}_ap"

        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics)
        return metrics

    def compute_metrices(self, model: AudioTransformer) -> Dict[str, Dict[str, float]]:
        with nullcontext() if self.truncate_dim is None else model.truncate_audio_embeddings(self.truncate_dim):
            try:
                audios = list(set(self.audio_1 + self.audio_2))
            except TypeError:
                embeddings = model.encode(
                    self.audio_1 + self.audio_2,
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar,
                    convert_to_numpy=True,
                )
                embeddings1 = embeddings[: len(self.audio_1)]
                embeddings2 = embeddings[len(self.audio_2) :]
            else:
                embeddings = model.encode(
                    audios,
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar,
                    convert_to_numpy=True,
                )
                emb_dict = {audio: emb for audio, emb in zip(audios, embeddings)}
                embeddings1 = [emb_dict[audio] for audio in self.audio_1]
                embeddings2 = [emb_dict[audio] for audio in self.audio_2]

        similarity_fns = {
            SimilarityFunction.COSINE.value: {
                "score_fn": lambda x, y: 1 - paired_cosine_distances(x, y),
                "name": "Cosine-Similarity",
                "greater_is_better": True,
            },
            SimilarityFunction.DOT_PRODUCT.value: {
                "score_fn": lambda x, y: np.sum(np.asarray(x) * np.asarray(y), axis=-1),
                "name": "Dot-Product",
                "greater_is_better": True,
            },
            SimilarityFunction.MANHATTAN.value: {
                "score_fn": lambda x, y: -paired_manhattan_distances(x, y),
                "name": "Manhattan-Distance",
                "greater_is_better": False,
            },
            SimilarityFunction.EUCLIDEAN.value: {
                "score_fn": lambda x, y: -paired_euclidean_distances(x, y),
                "name": "Euclidean-Distance",
                "greater_is_better": False,
            },
        }

        labels = np.asarray(self.labels)
        output_scores = {}
        for similarity_fn_name in self.similarity_fn_names:
            similarity_fn = similarity_fns[similarity_fn_name]
            scores = similarity_fn["score_fn"](embeddings1, embeddings2)
            greater_is_better = similarity_fn["greater_is_better"]
            name = similarity_fn["name"]

            acc, acc_threshold = self.find_best_acc_and_threshold(scores, labels, greater_is_better)
            f1, precision, recall, f1_threshold = self.find_best_f1_and_threshold(scores, labels, greater_is_better)
            ap = average_precision_score(labels, scores * (1 if greater_is_better else -1))

            predicted_labels = (scores >= f1_threshold) if greater_is_better else (scores <= f1_threshold)
            mcc = matthews_corrcoef(labels, predicted_labels)

            logger.info(f"Accuracy with {name}:             {acc * 100:.2f}\t(Threshold: {acc_threshold:.4f})")
            logger.info(f"F1 with {name}:                   {f1 * 100:.2f}\t(Threshold: {f1_threshold:.4f})")
            logger.info(f"Precision with {name}:            {precision * 100:.2f}")
            logger.info(f"Recall with {name}:               {recall * 100:.2f}")
            logger.info(f"Average Precision with {name}:    {ap * 100:.2f}")
            logger.info(f"Matthews Correlation with {name}: {mcc * 100:.2f}\n")

            output_scores[similarity_fn_name] = {
                "accuracy": acc,
                "accuracy_threshold": acc_threshold,
                "f1": f1,
                "f1_threshold": f1_threshold,
                "precision": precision,
                "recall": recall,
                "ap": ap,
                "mcc": mcc,
            }

        return output_scores

    @staticmethod
    def find_best_acc_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)
        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        max_acc = 0
        best_threshold = -1

        positive_so_far = 0
        remaining_negatives = sum(labels == 0)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > max_acc:
                max_acc = acc
                best_threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return max_acc, best_threshold

    @staticmethod
    def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)

        scores = np.asarray(scores)
        labels = np.asarray(labels)

        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        best_f1 = best_precision = best_recall = 0
        threshold = 0
        nextract = 0
        ncorrect = 0
        total_num_duplicates = sum(labels)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            nextract += 1

            if label == 1:
                ncorrect += 1

            if ncorrect > 0:
                precision = ncorrect / nextract
                recall = ncorrect / total_num_duplicates
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return best_f1, best_precision, best_recall, threshold