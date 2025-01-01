from itertools import zip_longest
from typing import Dict, Generator, Iterable, List, Optional, Union

import numpy as np

import torch
from torch.utils.data import IterableDataset

from . import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def shuffle_combinations(iterable: Iterable, replacement: bool = True) -> Generator:
    """Generates shuffled pair combinations for any iterable data provided.

    Args:
        iterable: data to generate pair combinations from
        replacement: enable to include combinations of same samples,
            equivalent to itertools.combinations_with_replacement

    Returns:
        Generator of shuffled pairs as a tuple
    """
    n = len(iterable)
    k = 1 if not replacement else 0
    idxs = np.stack(np.triu_indices(n, k), axis=-1)
    for i in np.random.RandomState(seed=914).permutation(len(idxs)):
        _idx, idx = idxs[i, :]
        yield iterable[_idx], iterable[idx]


class ContrastiveDataset(IterableDataset):

    def __init__(
        self,
        audios: List[str],
        labels: List[Union[int, float]],
        multilabel: bool,
        num_iterations: Optional[int] = None,  # Fixed the type hint
        sampling_strategy: str = "oversampling",
        max_pairs: int = -1,
    ) -> None:
        super().__init__()
        self.pos_index = 0
        self.neg_index = 0
        self.pos_pairs = []
        self.neg_pairs = []
        self.audios = audios
        self.labels = labels
        self.audios_labels = list(zip(self.audios, self.labels))
        self.max_pos_or_neg = -1 if max_pairs == -1 else max_pairs // 2

        if multilabel:
            self.generate_multilabel_pairs()
        else:
            self.generate_pairs()

        # Adjust len_pos_pairs and len_neg_pairs based on max_pairs
        total_pairs = len(self.pos_pairs) + len(self.neg_pairs)
        if max_pairs != -1:
            if total_pairs > max_pairs:
                ratio = max_pairs / total_pairs
                self.len_pos_pairs = int(len(self.pos_pairs) * ratio)
                self.len_neg_pairs = int(len(self.neg_pairs) * ratio)
            else:
                self.len_pos_pairs = len(self.pos_pairs)
                self.len_neg_pairs = len(self.neg_pairs)
        elif num_iterations is not None and num_iterations > 0:
            # Base on num_iterations if max_pairs is not set
            self.len_pos_pairs = num_iterations * len(self.audios)
            self.len_neg_pairs = num_iterations * len(self.audios)
        elif sampling_strategy == "unique":
            self.len_pos_pairs = len(self.pos_pairs)
            self.len_neg_pairs = len(self.neg_pairs)
        elif sampling_strategy == "undersampling":
            self.len_pos_pairs = min(len(self.pos_pairs), len(self.neg_pairs))
            self.len_neg_pairs = min(len(self.pos_pairs), len(self.neg_pairs))
        elif sampling_strategy == "oversampling":
            self.len_pos_pairs = max(len(self.pos_pairs), len(self.neg_pairs))
            self.len_neg_pairs = max(len(self.pos_pairs), len(self.neg_pairs))
        else:
            raise ValueError("Invalid sampling strategy. Must be one of 'unique', 'oversampling', or 'undersampling'.")

    def generate_pairs(self) -> None:
        for (_audio, _label), (audio, label) in shuffle_combinations(self.audios_labels):
            is_positive = _label == label
            is_positive_full = self.max_pos_or_neg != -1 and len(self.pos_pairs) >= self.max_pos_or_neg
            is_negative_full = self.max_pos_or_neg != -1 and len(self.neg_pairs) >= self.max_pos_or_neg

            if is_positive:
                if not is_positive_full:
                    self.pos_pairs.append({"audio_1": _audio, "audio_2": audio, "label": 1.0})
            elif not is_negative_full:
                self.neg_pairs.append({"audio_1": _audio, "audio_2": audio, "label": 0.0})

            if is_positive_full and is_negative_full:
                break

    def generate_multilabel_pairs(self) -> None:
        for (_audio, _label), (audio, label) in shuffle_combinations(self.audios_labels):
            # logical_and checks if labels are both set for each class
            is_positive = any(np.logical_and(_label, label))
            is_positive_full = self.max_pos_or_neg != -1 and len(self.pos_pairs) >= self.max_pos_or_neg
            is_negative_full = self.max_pos_or_neg != -1 and len(self.neg_pairs) >= self.max_pos_or_neg

            if is_positive:
                if not is_positive_full:
                    self.pos_pairs.append({"audio_1": _audio, "audio_2": audio, "label": 1.0})
            elif not is_negative_full:
                self.neg_pairs.append({"audio_1": _audio, "audio_2": audio, "label": 0.0})

            if is_positive_full and is_negative_full:
                break

    def get_positive_pairs(self) -> List[Dict[str, Union[str, float]]]:
        pairs = []
        for _ in range(self.len_pos_pairs):
            if self.pos_index >= len(self.pos_pairs):
                self.pos_index = 0
            pairs.append(self.pos_pairs[self.pos_index])
            self.pos_index += 1
        return pairs

    def get_negative_pairs(self) -> List[Dict[str, Union[str, float]]]:
        pairs = []
        for _ in range(self.len_neg_pairs):
            if self.neg_index >= len(self.neg_pairs):
                self.neg_index = 0
            pairs.append(self.neg_pairs[self.neg_index])
            self.neg_index += 1
        return pairs

    def __iter__(self):
        for pos_pair, neg_pair in zip_longest(self.get_positive_pairs(), self.get_negative_pairs()):
            if pos_pair is not None:
                yield pos_pair
            if neg_pair is not None:
                yield neg_pair

    def __len__(self) -> int:
        return self.len_pos_pairs + self.len_neg_pairs
