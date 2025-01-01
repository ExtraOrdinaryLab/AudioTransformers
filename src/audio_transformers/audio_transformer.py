import os
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
from rich import print
from tqdm.autonotebook import trange

import torch
import torchaudio
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

from datasets import Audio
from transformers import (
    AutoConfig, 
    AutoModel, 
    AutoFeatureExtractor, 
    is_torch_npu_available, 
)

from . import util
from . import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


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
        self.return_attention_mask = return_attention_mask
        
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

    def featurize(self, audios: Union[Dict[str, List[Any]], List[Any]]):
        model_input_name = self.feature_extractor.model_input_names[0]

        def load_audio(audio_filename):
            waveform, sample_rate = torchaudio.load(audio_filename)
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=self.feature_extractor.sampling_rate
            )
            return waveform.squeeze(dim=0)

        if isinstance(audios, list) and isinstance(audios[0], str):
            audios = [load_audio(audio) for audio in audios]
        elif isinstance(audios, list) and isinstance(audios[0], dict) and ('array' in audios[0]):
            audios = [audio['array'] for audio in audios]
        # elif isinstance(audios, dict):
        #     column = list(audios.keys())[0]
        #     audios = [load_audio(audio) for audio in audios[column]]

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
                padding=True, 
                return_tensors='pt'
            )
            outputs = {model_input_name: inputs.get(model_input_name)}
        else:
            wavs = [to_numpy(audio) for audio in audios]
            inputs = self.feature_extractor(
                wavs, 
                sampling_rate=self.feature_extractor.sampling_rate, 
                return_tensors='pt'
            )
            outputs = {model_input_name: inputs.get(model_input_name)}

        if self.return_attention_mask and 'attention_mask' in inputs:
            outputs['attention_mask'] = inputs['attention_mask']
            
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
        else:
            self.device = device
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
        self.to(self.device)

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
        Computes embeddings for audio files or audio data.

        This method processes audio data to compute embeddings, which can represent either 
        entire segments or individual frames, based on the `output_value` parameter. 
        The embeddings can optionally be normalized, converted to tensors or NumPy arrays, 
        and returned in a format suitable for downstream tasks such as classification or retrieval.

        Args:
            audios (Union[str, List[str]]): The input audio(s), either as a single audio file path 
                or a list of file paths.
            batch_size (int, optional): The number of audio files to process in a single batch. 
                Defaults to 16.
            device (str, optional): The computation device to use (`'cpu'`, `'cuda'`, etc.). 
                If not specified, uses the device associated with the model.
            normalize_embeddings (bool, optional): Whether to normalize the computed embeddings 
                to unit length. Defaults to False.
            output_value (Literal["segment_embedding", "frame_embeddings"], optional): Specifies the type 
                of embeddings to return:
                - `'segment_embedding'`: Single embedding per audio segment.
                - `'frame_embeddings'`: Sequence of embeddings for individual audio frames.
                Defaults to `'segment_embedding'`.
            show_progress_bar (bool, optional): Whether to display a progress bar during processing. 
                Defaults to False.
            convert_to_numpy (bool, optional): If True, converts the embeddings to NumPy arrays. 
                Defaults to True.
            convert_to_tensor (bool, optional): If True, converts the embeddings to PyTorch tensors. 
                If both `convert_to_numpy` and `convert_to_tensor` are True, PyTorch tensors are prioritized. 
                Defaults to False.

        Returns:
            Union[torch.Tensor, np.ndarray, List[torch.Tensor]]: The computed embeddings, depending on the 
            specified options:
                - A single embedding if `audios` is a string.
                - A list of embeddings if `audios` is a list of file paths.
                - The format of embeddings (PyTorch tensor, NumPy array, or list) depends on the values of 
                `convert_to_numpy` and `convert_to_tensor`.

        Raises:
            ValueError: If both `convert_to_numpy` and `convert_to_tensor` are set to False.

        Examples:
            >>> embeddings = model.encode("example.wav", output_value="segment_embedding")
            >>> embeddings = model.encode(["audio1.wav", "audio2.wav"], batch_size=8, normalize_embeddings=True)
            >>> tensor_embeddings = model.encode("example.wav", convert_to_tensor=True)
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
                    all_embeddings = np.asarray([to_numpy(emb.float()) for emb in all_embeddings])
                else:
                    all_embeddings = np.asarray([to_numpy(emb) for emb in all_embeddings])
        elif isinstance(all_embeddings, np.ndarray):
            all_embeddings = [torch.from_numpy(embedding) for embedding in all_embeddings]

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def forward(
        self,
        features: Union[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for training or inference.
        
        Args:
            features: Audio features.
            batch_size: Batch size for processing.
            device: Device to run the computation on.
            normalize_embeddings: If True, normalize embeddings to unit length.
            output_value: Determines the output type - "segment_embedding" or "frame_embeddings".
            return_loss: If True, computes and returns loss (requires labels).
            labels: Target labels for supervised training.

        Returns:
            Dict containing model outputs and optionally the loss if `return_loss` is True.
        """
        features = batch_to_device(features, self.device)

        out_features = self.transformer_model.forward(features)
        out_features = self.pooling_model.forward(out_features)

        return out_features

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

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        # Propagate the gradient checkpointing to the transformer model
        if isinstance(self.transformer_model, Transformer):
            return self.transformer_model.auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    

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
