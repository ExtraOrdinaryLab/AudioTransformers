from __future__ import annotations

import logging
from dataclasses import dataclass, field

from transformers.utils import ExplicitEnum
from transformers.training_args import ParallelMode
from transformers import TrainingArguments as TransformersTrainingArguments

logger = logging.getLogger(__name__)


class BatchSamplers(ExplicitEnum):

    BATCH_SAMPLER = "batch_sampler"
    NO_DUPLICATES = "no_duplicates"
    GROUP_BY_LABEL = "group_by_label"


class MultiDatasetBatchSamplers(ExplicitEnum):
    
    ROUND_ROBIN = "round_robin"  # Round-robin sampling from each dataset
    PROPORTIONAL = "proportional"  # Sample from each dataset in proportion to its size [default]


@dataclass
class AudioTransformerTrainingArguments(TransformersTrainingArguments):

    prompts: dict[str, dict[str, str]] | dict[str, str] | str | None = None
    batch_sampler: BatchSamplers | str = field(
        default=BatchSamplers.BATCH_SAMPLER, metadata={"help": "The batch sampler to use."}
    )
    multi_dataset_batch_sampler: MultiDatasetBatchSamplers | str = field(
        default=MultiDatasetBatchSamplers.PROPORTIONAL, metadata={"help": "The multi-dataset batch sampler to use."}
    )

    def __post_init__(self):
        super().__post_init__()

        self.batch_sampler = BatchSamplers(self.batch_sampler)
        self.multi_dataset_batch_sampler = MultiDatasetBatchSamplers(self.multi_dataset_batch_sampler)

        # The `compute_loss` method in `SentenceTransformerTrainer` is overridden to only compute the prediction loss,
        # so we set `prediction_loss_only` to `True` here to avoid
        self.prediction_loss_only = True

        # Disable broadcasting of buffers to avoid `RuntimeError: one of the variables needed for gradient computation
        # has been modified by an inplace operation.` when training with DDP & a BertModel-based model.
        self.ddp_broadcast_buffers = False

        if self.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
            # If output_dir is "unused", then this instance is created to compare training arguments vs the defaults,
            # so we don't have to warn.
            if self.output_dir != "unused":
                logger.warning(
                    "Currently using DataParallel (DP) for multi-gpu training, while DistributedDataParallel (DDP) is recommended for faster training. "
                    "See https://sbert.net/docs/sentence_transformer/training/distributed.html for more information."
                )

        elif self.parallel_mode == ParallelMode.DISTRIBUTED and not self.dataloader_drop_last:
            # If output_dir is "unused", then this instance is created to compare training arguments vs the defaults,
            # so we don't have to warn.
            if self.output_dir != "unused":
                logger.warning(
                    "When using DistributedDataParallel (DDP), it is recommended to set `dataloader_drop_last=True` to avoid hanging issues with an uneven last batch. "
                    "Setting `dataloader_drop_last=True`."
                )
            self.dataloader_drop_last = True