import os
import json
import logging
import argparse
import warnings

import numpy as np
from rich import print
from rich.logging import RichHandler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset, Dataset

from audio_transformers.logging import get_logger
from audio_transformers.sampler import PositivePairsDataset, ContrastiveDataset
from audio_transformers.trainer import AudioTransformerTrainer
from audio_transformers.evaluation import BinaryClassificationEvaluator
from audio_transformers.similarity_functions import SimilarityFunction
from audio_transformers.training_args import AudioTransformerTrainingArguments
from audio_transformers import AudioTransformer, losses

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO, 
    # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    # datefmt="%Y-%m-%d %H:%M:%S", 
    # filename="confit.log", 
    # handlers=[RichHandler()]
)
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch ConFit Training")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/mnt/data4_HDD_14TB/yang/confit-checkpoints/gtzan", 
        help="Output directory.", 
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default='facebook/wav2vec2-base', 
    )
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        default=None, 
    )
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default='confit/gtzan', 
        help="Name of a dataset from the datasets package"
    )
    parser.add_argument(
        "--dataset_config_name", 
        type=str, 
        default=None, 
        help="The configuration name of the dataset to use (via the datasets library)."
    )
    parser.add_argument(
        "--num_iterations", 
        type=int, 
        default=10, 
        help="Number of iterations to generate positive pairs.",
    )
    parser.add_argument(
        "--max_pairs", 
        type=int, 
        default=-1, 
        help="Maximum number to generate positive pairs.",
    )
    parser.add_argument(
        "--eval_num_iterations", 
        type=int, 
        default=1, 
        help="Number of iterations to generate positive pairs for validation set.",
    )
    parser.add_argument(
        "--max_length_seconds", 
        type=float, 
        default=10, 
        help="Length (in seconds) of the audio to train.",
    )
    parser.add_argument(
        "--return_attention_mask", 
        action="store_true", 
        help="Return attention_mask.",
    )
    parser.add_argument(
        "--pooling_mode", 
        type=str, 
        default='mean', 
        help="Pooling methods.",
    )
    parser.add_argument(
        "--mini_batch_size", 
        type=int, 
        default=None, 
        help="Mini batch size used in GradCache.",
    )
    parser.add_argument(
        "--per_device_train_batch_size", 
        type=int, 
        default=None, 
        help="Train batch size.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size", 
        type=int, 
        default=None, 
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--dataloader_num_workers", 
        type=int, 
        default=0, 
        help="Number of workers for dataloader.",
    )
    parser.add_argument(
        "--gradient_checkpointing", 
        action="store_true", 
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--fp16", 
        action="store_true", 
        help="Enable mixed precision training.",
    )
    parser.add_argument(
        "--bf16", 
        action="store_true", 
        help="Enable bfloat16 training.",
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=1, 
        help="Enable bfloat16 training.",
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-5, 
        help="Learning rate.",
    )
    parser.add_argument(
        "--report_to", 
        type=str, 
        default="none", 
        help="The list of integrations to report the results and logs to.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    output_dir = args.output_dir

    raw_datasets = load_dataset(
        args.dataset_name, 
        args.dataset_config_name, 
        trust_remote_code=True
    )
    logger.info(raw_datasets)

    train_sampler = PositivePairsDataset(
        audios=[example['file'] for example in raw_datasets['train']], 
        labels=[example['label'] for example in raw_datasets['train']], 
        multilabel=False, 
        num_iterations=args.num_iterations, 
        sampling_strategy='oversampling', 
        max_pairs=args.max_pairs,
    )
    train_dataset = Dataset.from_list(list(train_sampler))

    if 'validation' in raw_datasets:
        val_audios = [example['file'] for example in raw_datasets['validation']]
        val_labels = [example['label'] for example in raw_datasets['validation']]
    elif 'test' in raw_datasets and 'validation' not in raw_datasets:
        val_audios = [example['file'] for example in raw_datasets['test']]
        val_labels = [example['label'] for example in raw_datasets['test']]

    val_sampler = ContrastiveDataset(
        audios=val_audios, 
        labels=val_labels, 
        multilabel=False, 
        num_iterations=args.eval_num_iterations, 
        sampling_strategy='oversampling', 
    )
    val_dataset = Dataset.from_list(list(val_sampler))

    audio_transformer = AudioTransformer(
        model_name_or_path=args.model_name_or_path, 
        max_length_seconds=args.max_length_seconds, 
        return_attention_mask=args.return_attention_mask, 
        pooling_mode=args.pooling_mode, 
    )

    if args.mini_batch_size is not None:
        loss_fn = losses.CachedMultipleNegativesRankingLoss(
            audio_transformer, 
            mini_batch_size=args.mini_batch_size
        )
    else:
        loss_fn = losses.MultipleNegativesRankingLoss(audio_transformer)

    val_evaluator = BinaryClassificationEvaluator(
        audio_1=val_dataset["audio_1"],
        audio_2=val_dataset["audio_2"],
        labels=val_dataset["label"],
        similarity_fn_names=['cosine'], 
        name=f"{str(args.dataset_name).split('/')[1]}-val",
    )

    training_args = AudioTransformerTrainingArguments(
        output_dir=output_dir, 
        do_train=True, 
        do_eval=True, 
        per_device_train_batch_size=args.per_device_train_batch_size, 
        per_device_eval_batch_size=args.per_device_eval_batch_size, 
        dataloader_num_workers=args.dataloader_num_workers, 
        gradient_checkpointing=args.gradient_checkpointing, 
        fp16=args.bf16, 
        bf16=args.bf16, 
        num_train_epochs=args.num_train_epochs, 
        learning_rate=args.learning_rate, 
        logging_steps=10, 
        report_to=args.report_to, 
    )
    trainer = AudioTransformerTrainer(
        model=audio_transformer, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=val_dataset, 
        loss_fn=loss_fn, 
        evaluator=val_evaluator,
    )
    trainer.train()

    if 'test' in raw_datasets:
        test_audios = [example['file'] for example in raw_datasets['test']]
        test_labels = [example['label'] for example in raw_datasets['test']]
    elif 'validation' in raw_datasets and 'test' not in raw_datasets:
        test_audios = [example['file'] for example in raw_datasets['validation']]
        test_labels = [example['label'] for example in raw_datasets['validation']]

    test_sampler = ContrastiveDataset(
        audios=test_audios, 
        labels=test_labels, 
        multilabel=False, 
        num_iterations=args.eval_num_iterations, 
        sampling_strategy='oversampling', 
    )
    test_dataset = Dataset.from_list(list(test_sampler))

    test_evaluator = BinaryClassificationEvaluator(
        audio_1=test_dataset["audio_1"],
        audio_2=test_dataset["audio_2"],
        labels=test_dataset["label"],
        similarity_fn_names=['cosine'], 
        name=f"{str(args.dataset_name).split('/')[1]}-test",
    )
    results = test_evaluator(audio_transformer)
    logger.info(results)

    final_output_dir = f"{output_dir}/final"
    audio_transformer.save(final_output_dir)

    x_train = audio_transformer.encode(
        audios=[example['file'] for example in raw_datasets['train']], 
        batch_size=1, 
        normalize_embeddings=True, 
        show_progress_bar=True
    )
    y_train = np.array([example['label'] for example in raw_datasets['train']])
    classifier = LogisticRegression(max_iter=500)
    classifier.fit(x_train, y_train)

    x_test = audio_transformer.encode(
        audios=test_audios, 
        batch_size=1, 
        normalize_embeddings=True, 
        show_progress_bar=True
    )
    y_test = np.array(test_labels)
    y_pred = classifier.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy:.4f}")

    with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
        all_results = {
            'predict_accuracy': accuracy, **results
        }
        json.dump(all_results, f, indent=4)


if __name__ == '__main__':
    main()