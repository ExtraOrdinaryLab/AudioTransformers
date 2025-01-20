import os
import math
import random
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import List, Union, Dict, Any, Callable, Tuple, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import huggingface_hub.utils as hf_hub_utils
from sklearn.linear_model import LogisticRegression
from packaging.version import parse as parse_version
from transformers import __version__ as transformers_version
from transformers import Trainer, TrainerCallback
from transformers.data.data_collator import DataCollator
from transformers.integrations import WandbCallback
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_sagemaker_mp_enabled
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.trainer_utils import (
    EvalPrediction,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    set_seed,
)
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, Value
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, SubsetRandomSampler

from . import logging, losses
from . import AudioTransformer
from .util import disable_logging, batch_to_device
from .evaluation import AudioEvaluator
from .sampler import NoDuplicatesBatchSampler
from .training_args import AudioTransformerTrainingArguments, BatchSamplers

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

if TYPE_CHECKING:
    import optuna


@dataclass
class AudioTransformerDataCollator(object):

    featurizer: Callable
    valid_label_columns: list[str] = field(default_factory=lambda: ["label", "score"])

    # def __init__(self, audio_transformer: AudioTransformer):
    #     self.audio_transformer = audio_transformer
    #     self.valid_label_columns = ['label', 'labels', 'audio_1', 'audio_2']

    # def __call__(self, batch: List[Dict[str, Any]]):
    #     features = {}
    #     columns = list(batch[0].keys())
    #     for column in columns:
    #         if column in ['label', 'labels']:
    #             features['label'] = [example[column] for example in batch]
    #         else:
    #             audios_batch = [example[column] for example in batch]
    #             features[column] = self.audio_transformer.transformer_model.featurize(audios_batch)
    #     return features

    def __call__(self, features: List[Dict[str, Any]]):
        column_names = list(features[0].keys())

        batch = {}

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        # Extract the label column if it exists
        for label_column in self.valid_label_columns:
            if label_column in column_names:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                column_names.remove(label_column)
                break

        for column_name in column_names:
            audios_batch = [row[column_name] for row in features]
            batch[column_name] = self.featurizer(audios_batch)
        
        return batch


class AudioTransformerTrainer(Trainer):

    def __init__(
        self, 
        model: AudioTransformer,
        args: AudioTransformerTrainingArguments = None, 
        train_dataset: Union[Dataset, DatasetDict, IterableDataset, Dict[str, Dataset]] = None,
        eval_dataset: Union[Dataset, DatasetDict, IterableDataset, Dict[str, Dataset]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction, Dict[str, Any]], Dict[str, float]]] = None,
        data_collator: Callable = None, 
        loss_fn: Union[nn.Module, Callable[[AudioTransformer], nn.Module], Dict[str, Callable[[AudioTransformer], nn.Module]]] = None, 
        evaluator = None,
        callbacks: List[TrainerCallback] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")

        if model is None:
            raise RuntimeError("`Trainer` requires a `model` argument")

        if compute_metrics is not None:
            logger.warning(
                "`compute_metrics` is currently not compatible with the AudioTransformerTrainer. Please use the "
                "`evaluator` argument instead for detailed evaluation metrics, or the `eval_dataset` argument for "
                "the evaluation loss."
            )

        # Get a dictionary of the default training arguments, so we can determine which arguments have been changed
        # for the model card
        default_args_dict = AudioTransformerTrainingArguments(output_dir="unused").to_dict()

        if data_collator is None:
            data_collator = AudioTransformerDataCollator(
                featurizer=model.transformer_model.featurize
            )

        for dataset_name, dataset in zip(["train", "eval"], [train_dataset, eval_dataset]):
            if isinstance(dataset, IterableDataset) and dataset.column_names is None:
                sample = next(iter(dataset))
                naive_type_mapping = {str: "string", int: "int64", float: "float32", bool: "bool"}
                example_features = {
                    key: Value(naive_type_mapping.get(type(value), "null")) for key, value in sample.items()
                }
                raise ValueError(
                    f"The provided `{dataset_name}_dataset` must have Features. Specify them with e.g.:\n"
                    f"{dataset_name}_dataset = {dataset_name}_dataset.cast(Features({example_features}))\n"
                    "or by providing the Features to the IterableDataset initialization method. See the Datasets "
                    "documentation for more information on dataset Features: "
                    "https://huggingface.co/docs/datasets/en/about_dataset_features"
                )

        if isinstance(train_dataset, dict) and not isinstance(train_dataset, DatasetDict):
            train_dataset = DatasetDict(train_dataset)
        if isinstance(eval_dataset, dict) and not isinstance(eval_dataset, DatasetDict):
            eval_dataset = DatasetDict(eval_dataset)

        # Transformers v4.46.0 introduced a ValueError if `eval_dataset` is None while eval_strategy is not "no",
        # but in Sentence Transformers you can also evaluate without an eval_dataset via an evaluator, so we set
        # it to "dummy" in that case to avoid the ValueError
        super_kwargs = {
            "model": model,
            "args": args,
            "data_collator": data_collator,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset if eval_dataset is not None else "dummy",
            "model_init": None,
            "compute_metrics": compute_metrics,
            "callbacks": callbacks,
            "optimizers": optimizers,
            "preprocess_logits_for_metrics": preprocess_logits_for_metrics,
        }
        # Transformers v4.46.0 changed the `tokenizer` argument to a more general `processing_class` argument
        if parse_version(transformers_version) >= parse_version("4.46.0"):
            super_kwargs["processing_class"] = model.transformer_model.feature_extractor
        else:
            super_kwargs["tokenizer"] = model.transformer_model.feature_extractor

        super().__init__(**super_kwargs)
        # If the eval_dataset is "dummy", then we set it back to None
        if self.eval_dataset == "dummy":
            self.eval_dataset = None

        self.model: AudioTransformer
        self.args: AudioTransformerTrainingArguments
        self.data_collator: AudioTransformerDataCollator
        # Set the W&B project via environment variables if it's not already set
        if any([isinstance(callback, WandbCallback) for callback in self.callback_handler.callbacks]):
            os.environ.setdefault("WANDB_PROJECT", "audio-transformers")

        if loss_fn is None:
            logger.info("No `loss` passed, using `losses.MultipleNegativesRankingLoss` as a default option.")
            loss_fn = losses.MultipleNegativesRankingLoss(self.model)
        self.loss_fn = self.prepare_loss(loss_fn, model)

        self.evaluator = evaluator

    def prepare_loss(
        self,
        loss: Union[Callable[[AudioTransformer], torch.nn.Module], torch.nn.Module],
        model: AudioTransformer,
    ) -> torch.nn.Module:
        if isinstance(loss, torch.nn.Module):
            return loss.to(model.device)
        return loss(model).to(model.device)

    def compute_loss(
        self,
        model: AudioTransformer,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        features, labels = self.collect_features(inputs)
        loss = self.loss_fn(features, labels)
        if return_outputs:
            return loss, {}
        return loss

    def collect_features(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor]:
        labels = inputs.pop('label')
        labels = torch.tensor(labels, device=self.model.device)
        features = [batch_to_device(v, self.model.device) for k, v in inputs.items()]
        return features, labels

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Training requires specifying a train_dataset to the AudioTransformerTrainer.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        generator = torch.Generator()
        if self.args.seed:
            generator.manual_seed(self.args.seed)

        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
            # "batch_size": self.args.train_batch_size,
            # "drop_last": self.args.dataloader_drop_last,
        }

        batch_sampler = NoDuplicatesBatchSampler(
            dataset=train_dataset,
            batch_size=self.args.train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            valid_label_columns=self.data_collator.valid_label_columns,
            generator=generator,
        )
        dataloader_params["batch_sampler"] = batch_sampler

        # If 'even_batches' is True, it will use the initial few samples to pad out the last sample. This can
        # cause issues with multi-dataset training, so we want to set this to False.
        # For evaluation, setting 'even_batches' to False results in hanging, so we keep it as True there.
        self.accelerator.even_batches = False
        self._train_dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        return self._train_dataloader

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        self.train_embeddings(
            resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs
        )

    def train_embeddings(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments used to hide deprecated arguments
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint)
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None:
                self._train_batch_size = state.train_batch_size

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)
        len_dataloader = len(train_dataloader)
        num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        num_examples = self.num_examples(train_dataloader)
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                args.max_steps % num_update_steps_per_epoch > 0
            )
            # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
            # the best we can do.
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(args.num_train_epochs)
            num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs

        logger.info("***** Running training *****")
        logger.info(f"  Num unique pairs = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Num GPUs = {args.n_gpu}")
        logger.info(f"  World size = {args.world_size}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self._train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
        if args.push_to_hub:
            try:
                # Disable progress bars when uploading models during checkpoints to avoid polluting stdout
                hf_hub_utils.disable_progress_bars()
                return inner_training_loop(
                    args=args,
                    resume_from_checkpoint=resume_from_checkpoint,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                )
            finally:
                hf_hub_utils.enable_progress_bars()
        else:
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

    def evaluate(
        self,
        eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
        ignore_keys: List[str] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        eval_dataset = self.eval_dataset
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool = None,
        ignore_keys: list[str] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        output = super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # If the evaluator is not defined, we can just return the output
        if self.evaluator is None:
            return output

        # If we are training and eval_dataset is a DatasetDict, then we should
        # 1) only run the evaluator for the first dataset
        # 2) prefix that only run as "eval", rather than e.g. "eval_multi_nli"
        if self.is_in_train and isinstance(self.eval_dataset, dict) and metric_key_prefix.startswith("eval_"):
            if metric_key_prefix[5:] == list(self.eval_dataset.keys())[0]:
                metric_key_prefix = "eval"
            else:
                return output

        with nullcontext() if self.is_local_process_zero() else disable_logging(logging.INFO):
            output_path = self.args.output_dir
            if output_path is not None:
                output_path = os.path.join(output_path, "eval")
                os.makedirs(output_path, exist_ok=True)
            evaluator_metrics = self.evaluator(
                self.model, output_path=output_path, epoch=self.state.epoch, steps=self.state.global_step
            )
        if not isinstance(evaluator_metrics, dict):
            evaluator_metrics = {"evaluator": evaluator_metrics}

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(evaluator_metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                evaluator_metrics[f"{metric_key_prefix}_{key}"] = evaluator_metrics.pop(key)

        output.metrics.update(evaluator_metrics)

        return output
