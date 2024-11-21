import os
import logging
from random import randint
from typing import (
    Literal, 
    Any, 
    Union, 
    Dict, 
    Optional, 
    List
)

import numpy as np
from tqdm.autonotebook import trange

import torch
import torchaudio
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

from transformers import (
    AutoConfig, 
    AutoModel, 
    AutoFeatureExtractor, 
    is_torch_npu_available, 
)

logger = logging.getLogger(__name__)


class Transformer(nn.Module):
    """HuggingFace AutoModel to generate frame-level embeddings.
    """
    def __init__(
        self, 
        model_name_or_path: str, 
        feature_extractor_name: str = None, 
        config_args: Dict[str, Any] = None, 
        model_args: Dict[str, Any] = None, 
        feature_extractor_args: Dict[str, Any] = None, 
        max_length_seconds: float = None, 
        return_attention_mask: bool = False, 
        token: Union[str, bool] = None, 
        trust_remote_code: bool = False, 
        revision: str = None, 
        cache_dir: str = None,  # Path to store cached data and models, 
    ):
        super().__init__()
        shared_kwargs = {
            "token": token, 
            "trust_remote_code": trust_remote_code, 
            "revision": revision, 
        }
        model_args = shared_kwargs if model_args is None else {**shared_kwargs, **model_args}
        feature_extractor_args = shared_kwargs if feature_extractor_args is None else {**shared_kwargs, **feature_extractor_args}
        config_args = shared_kwargs if config_args is None else {**shared_kwargs, **config_args}

        self.max_length_seconds = max_length_seconds
        
        self.config = AutoConfig.from_pretrained(
            model_name_or_path, 
            **config_args, 
            cache_dir=cache_dir
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            feature_extractor_name or model_name_or_path,
            **feature_extractor_args,
            return_attention_mask=return_attention_mask, 
            cache_dir=cache_dir,
        )
        self.auto_model = AutoModel.from_pretrained(
            model_name_or_path, 
            config=self.config, 
            cache_dir=cache_dir, 
            **model_args
        )

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """Returns frame_embeddings"""
        output_states = self.auto_model(**features, **kwargs, return_dict=False)
        output_frames = output_states[0]
        features["frame_embeddings"] = output_frames
        if 'attention_mask' in features:
            padding_mask = self.auto_model._get_feature_vector_attention_mask(
                output_frames.shape[1], 
                features['attention_mask']
            )
            features['padding_mask'] = padding_mask
        return features

    def featurize(self, audios: List[str]):

        def load_audio(audio_filename):
            waveform, sample_rate = torchaudio.load(audio_filename)
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=self.feature_extractor.sampling_rate
            )
            return waveform.squeeze(dim=0)

        audios = [load_audio(audio) for audio in audios]

        def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
            """Randomly sample chunks of `max_length` seconds from the input audio"""
            sample_length = int(round(sample_rate * max_length))
            if len(wav) <= sample_length:
                return wav
            random_offset = randint(0, len(wav) - sample_length - 1)
            return wav[random_offset : random_offset + sample_length]

        if self.max_length_seconds is not None:
            subsampled_wavs = []
            for audio in audios:
                wav = random_subsample(
                    audio, 
                    max_length=self.max_length_seconds, 
                    sample_rate=self.feature_extractor.sampling_rate
                )
                subsampled_wavs.append(wav)
            inputs = self.feature_extractor(
                [to_numpy(wav) for wav in subsampled_wavs], 
                sampling_rate=self.feature_extractor.sampling_rate, 
                return_tensors='pt'
            )
            model_input_name = self.feature_extractor.model_input_names[0]
            outputs = {model_input_name: inputs.get(model_input_name)}
        else:
            wavs = [to_numpy(audio) for audio in audios]
            inputs = self.feature_extractor(
                wavs, 
                sampling_rate=self.feature_extractor.sampling_rate, 
                return_tensors='pt'
            )
            outputs = {model_input_name: inputs.get(model_input_name)}
            
        return outputs


class Pooling(nn.Module):

    POOLING_MODES = (
        'statistic', 
        'mean', 
        'weighted_mean', 
        'max', 
    )

    def __init__(self, pooling_mode: str = None, hidden_size: int = None):
        super().__init__()
        self.pooling_mode = pooling_mode
        self.hidden_size = hidden_size

        if pooling_mode is not None:  # Set pooling mode by string
            pooling_mode = pooling_mode.lower()

            if pooling_mode not in self.POOLING_MODES:
                raise ValueError(
                    f"Set invalid pooling mode: {pooling_mode}. Valid pooling modes are: {self.POOLING_MODES}."
                )
            
    def get_pooling_output_dimension(self) -> int:
        if self.pooling_mode == 'statistic':
            return self.hidden_size * 2  # For mean and std
        elif self.pooling_mode == 'mean':
            return self.hidden_size
        elif self.pooling_mode == 'weighted_mean':
            return self.hidden_size
        elif self.pooling_mode == 'max':
            return self.hidden_size
        
    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden_states = features["frame_embeddings"]
        padding_mask = (
            features['padding_mask'] 
            if 'padding_mask' in features 
            else None
        )

        # Pooling strategy
        if self.pooling_mode == 'mean':
            if padding_mask is None:
                pooled_output = hidden_states.mean(dim=1)
            else:
                hidden_states[~padding_mask] = 0.0
                pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        elif self.pooling_mode == 'weighted_mean':
            if padding_mask is None:
                batch_size, max_length_seconds, _ = hidden_states.shape
                padding_mask = torch.ones((batch_size, max_length_seconds)).to(hidden_states.device)
            input_mask_expanded = (
                padding_mask.unsqueeze(-1).expand(hidden_states.size()).to(hidden_states.dtype)
            )
            # token_embeddings shape: bs, seq, hidden_dim
            weights = (
                torch.arange(start=1, end=hidden_states.shape[1] + 1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(hidden_states.size())
                .to(hidden_states.dtype)
                .to(hidden_states.device)
            )
            assert weights.shape == hidden_states.shape == input_mask_expanded.shape
            input_mask_expanded = input_mask_expanded * weights

            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if "token_weights_sum" in features:
                sum_mask = features["token_weights_sum"].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_output = sum_embeddings / sum_mask

        elif self.pooling_mode =='statistic':
            # Statistic Pooling
            if padding_mask is None:
                mean_features = hidden_states.mean(dim=1)
                std_features = hidden_states.std(dim=1)
            else:
                mean_features = []
                std_features = []
                for i, length in enumerate(padding_mask.sum(dim=1)):
                    mean_features.append(hidden_states[i, :length].mean(dim=0))
                    std_features.append(hidden_states[i, :length].std(dim=0))
                mean_features = torch.stack(mean_features)
                std_features = torch.stack(std_features)
            pooled_output = torch.cat([mean_features, std_features], dim=-1)

        elif self.pooling_mode == 'max':
            if padding_mask is None:
                batch_size, max_length_seconds, _ = hidden_states.shape
                padding_mask = torch.ones((batch_size, max_length_seconds)).to(hidden_states.device)
            input_mask_expanded = (
                padding_mask.unsqueeze(-1).expand(hidden_states.size()).to(hidden_states.dtype)
            )
            hidden_states[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            pooled_output = torch.max(hidden_states, 1)[0] # Max over time

        features["segment_embedding"] = pooled_output
        return features


class AudioTransformer(nn.Module):

    def __init__(
        self, 
        model_name_or_path: str, 
        feature_extractor_name: str = None, 
        config_args: Dict[str, Any] = None, 
        model_args: Dict[str, Any] = None, 
        feature_extractor_args: Dict[str, Any] = None, 
        max_length_seconds: float = None, 
        return_attention_mask: bool = False, 
        pooling_mode: str = 'mean', 
        token: Union[str, bool] = None, 
        trust_remote_code: bool = False, 
        revision: str = None, 
        cache_dir: str = None,  # Path to store cached data and models, 
        device: str = None,  # Set to None to automatically choose device, 
    ):
        super().__init__()

        if device is None:
            self.device = self.get_device_name()
            logger.info(f"Use pytorch device_name: {self.device}")

        self.model_name_or_path = model_name_or_path
        if model_name_or_path is not None and model_name_or_path != "":
            logger.info(f"Load pre-trained AudioTransformer: {model_name_or_path}")

            if not os.path.exists(model_name_or_path):
                # Not a path, load from hub
                if "\\" in model_name_or_path or model_name_or_path.count("/") > 1:
                    raise ValueError(f"Path {model_name_or_path} not found")
                
                self.transformer_model = Transformer(
                    model_name_or_path, 
                    feature_extractor_name=feature_extractor_name, 
                    config_args=config_args, 
                    model_args=model_args, 
                    feature_extractor_args=feature_extractor_args, 
                    max_length_seconds=max_length_seconds, 
                    return_attention_mask=return_attention_mask, 
                    token=token, 
                    trust_remote_code=trust_remote_code, 
                    revision=revision, 
                    cache_dir=cache_dir,
                )
                self.pooling_model = Pooling(
                    pooling_mode=pooling_mode, 
                    hidden_size=self.transformer_model.config.hidden_size
                )

    def encode(
        self, 
        audios: Union[str, List[str]], 
        batch_size: int = 16, 
        device: str = None, 
        normalize_embeddings: bool = False,
        output_value: Literal["segment_embedding", "frame_embeddings"] = None, 
        show_progress_bar: bool = False, 
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ):
        """
        Computes audio embeddings.
        """
        self.eval()

        input_was_string = False
        if isinstance(audios, str) or not hasattr(audios, "__len__"):
            audios = [audios]
            input_was_string = True

        if device is None:
            device = self.device
        self.to(device)

        all_embeddings = []
        for start_index in trange(0, len(audios), batch_size, desc="Batches", disable=not show_progress_bar):
            audios_batch = audios[start_index : start_index + batch_size]
            features = self.transformer_model.featurize(audios_batch)
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.transformer_model.forward(features)
                out_features = self.pooling_model.forward(out_features)

                if output_value is None:
                    output_value = 'segment_embedding'

                if output_value == 'frame_embeddings':
                    embeddings = []
                    for frame_emb, attention in zip(out_features[output_value], out_features["padding_mask"]):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1
                        embeddings.append(frame_emb[0 : last_mask_id + 1])
                elif output_value == 'segment_embedding':
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.extend(embeddings)
            
        if convert_to_tensor:
            if len(all_embeddings):
                if isinstance(all_embeddings, np.ndarray):
                    all_embeddings = torch.from_numpy(all_embeddings)
                else:
                    all_embeddings = torch.stack(all_embeddings)
            else:
                all_embeddings = torch.Tensor()
        elif convert_to_numpy:
            if not isinstance(all_embeddings, np.ndarray):
                if all_embeddings and all_embeddings[0].dtype == torch.bfloat16:
                    all_embeddings = np.asarray([emb.float().numpy() for emb in all_embeddings])
                else:
                    all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        elif isinstance(all_embeddings, np.ndarray):
            all_embeddings = [torch.from_numpy(embedding) for embedding in all_embeddings]

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def get_device_name(self) -> Literal["mps", "cuda", "npu", "hpu", "cpu"]:
        """
        Returns the name of the device where this module is running on.

        It's a simple implementation that doesn't cover cases when more powerful GPUs are available and
        not a primary device ('cuda:0') or MPS device is available, but not configured properly.

        Returns:
            str: Device name, like 'cuda' or 'cpu'
        """
        if torch.cuda.is_available():
            return "cuda"
        elif is_torch_npu_available():
            return "npu"
        return "cpu"
    

def batch_to_device(batch: Dict[str, Any], target_device: str) -> dict[str, Any]:
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


def to_numpy(X):
    if isinstance(X, np.ndarray):
        return X

    if is_pandas_ndframe(X):
        return X.values

    if not is_torch_data_type(X):
        raise TypeError("Cannot convert this data type to a numpy array.")

    if X.is_cuda:
        X = X.cpu()

    if X.requires_grad:
        X = X.detach()

    return X.numpy()


def is_torch_data_type(x):
    # pylint: disable=protected-access
    return isinstance(x, (torch.Tensor, PackedSequence))


def is_pandas_ndframe(x):
    # the sklearn way of determining this
    return hasattr(x, 'iloc')


if __name__ == "__main__":
    audios = [
        '/mnt/data1_HDD_14TB/yang/corpus/audio/VoxCeleb1/test/wav/id10270/x6uYqmx31kE/00001.wav', 
        '/mnt/data1_HDD_14TB/yang/corpus/audio/VoxCeleb1/test/wav/id10270/x6uYqmx31kE/00002.wav', 
        '/mnt/data1_HDD_14TB/yang/corpus/audio/VoxCeleb1/test/wav/id10270/x6uYqmx31kE/00003.wav', 
        '/mnt/data1_HDD_14TB/yang/corpus/audio/VoxCeleb1/test/wav/id10270/x6uYqmx31kE/00004.wav', 
    ]
    audio_transformer = AudioTransformer(
        model_name_or_path='facebook/wav2vec2-base', 
        max_length_seconds=3, 
        return_attention_mask=True, 
        pooling_mode='mean', 
    )
    embeddings = audio_transformer.encode(audios, batch_size=2, show_progress_bar=True, convert_to_tensor=True)
    print(embeddings)
    print(embeddings.shape)