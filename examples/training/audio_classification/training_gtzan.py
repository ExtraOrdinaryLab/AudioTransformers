import warnings

from datasets import load_dataset, Dataset

from audio_transformers.sampler import ContrastiveDataset
from audio_transformers.trainer import AudioTransformerTrainer
from audio_transformers.training_args import AudioTransformerTrainingArguments
from audio_transformers import (
    AudioTransformer, 
    logging, 
    losses
)

warnings.filterwarnings('ignore')
logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def main():
    raw_datasets = load_dataset('confit/gtzan', trust_remote_code=True)
    logger.info(raw_datasets)

    data_sampler = ContrastiveDataset(
        audios=[example['file'] for example in raw_datasets['train']], 
        labels=[example['label'] for example in raw_datasets['train']], 
        multilabel=False, 
        num_iterations=20, 
        sampling_strategy='oversampling', 
        # max_pairs=10000,
    )
    train_dataset = Dataset.from_list(list(data_sampler))

    audio_transformer = AudioTransformer(
        model_name_or_path='facebook/wav2vec2-base', 
        max_length_seconds=10, 
        return_attention_mask=True, 
        pooling_mode='mean', 
    )

    loss_fn = losses.CachedMultipleNegativesRankingLoss(audio_transformer, mini_batch_size=8)

    training_args = AudioTransformerTrainingArguments(
        output_dir='/mnt/data4_HDD_14TB/yang/confit-checkpoints/gtzan', 
        do_train=True, 
        per_device_train_batch_size=32, 
        gradient_checkpointing=True, 
        fp16=True, 
        bf16=False, 
        num_train_epochs=3, 
        learning_rate=1e-5, 
        logging_steps=100, 
        report_to='none'
    )
    trainer = AudioTransformerTrainer(
        model=audio_transformer, 
        args=training_args, 
        train_dataset=train_dataset, 
        loss_fn=loss_fn
    )
    trainer.train()


if __name__ == '__main__':
    main()