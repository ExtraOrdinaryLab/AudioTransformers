import os
import sys
import json
import shutil
from pathlib import Path
from random import randint
from contextlib import contextmanager
from typing import (
    Literal, 
    Any, 
    Union, 
    Dict, 
    Optional, 
    List, 
    Iterator
)

import numpy as np
from rich import print
from tqdm.autonotebook import trange

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model

import transformers
from transformers.dynamic_module_utils import get_relative_import_files
from transformers import (
    AutoConfig, 
    AutoModel, 
    AutoFeatureExtractor, 
    is_torch_npu_available, 
)

from . import logging
from .util import batch_to_device, fullname, import_from_string
from .similarity_functions import SimilarityFunction

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
        self.config_keys = ["max_length_seconds"]
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

    def get_config_dict(self) -> dict[str, Any]:
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str, safe_serialization: bool = True) -> None:
        self.auto_model.save_pretrained(output_path, safe_serialization=safe_serialization)
        self.feature_extractor.save_pretrained(output_path)

        with open(os.path.join(output_path, "audio_transformer_config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @classmethod
    def load(cls, input_path: str):
        config_path = os.path.join(input_path, "audio_transformer_config.json")

        with open(config_path) as fIn:
            config = json.load(fIn)
        # Don't allow configs to set trust_remote_code
        if "model_args" in config and "trust_remote_code" in config["model_args"]:
            config["model_args"].pop("trust_remote_code")
        if "feature_extractor_args" in config and "trust_remote_code" in config["feature_extractor_args"]:
            config["feature_extractor_args"].pop("trust_remote_code")
        if "config_args" in config and "trust_remote_code" in config["config_args"]:
            config["config_args"].pop("trust_remote_code")
        return cls(model_name_or_path=input_path, **config)


class Pooling(nn.Module):

    POOLING_MODES = (
        'statistic', 
        'mean', 
        'weighted_mean', 
        'max', 
    )

    def __init__(self, pooling_mode: str = None, hidden_size: int = None):
        super().__init__()
        self.config_keys = [
            "pooling_mode",
            "hidden_size"
        ]

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

    def get_config_dict(self) -> dict[str, Any]:
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path) -> None:
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)
        return Pooling(**config)


class Dense(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_function = nn.Tanh(),
        init_weight: torch.Tensor = None,
        init_bias: torch.Tensor = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation_function = activation_function
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if init_weight is not None:
            self.linear.weight = nn.Parameter(init_weight)

        if init_bias is not None:
            self.linear.bias = nn.Parameter(init_bias)

    def forward(self, features: dict[str, torch.Tensor]):
        features.update(
            {"segment_embedding": self.activation_function(self.linear(features["segment_embedding"]))}
        )
        return features

    def get_audio_embedding_dimension(self) -> int:
        return self.out_features

    def get_config_dict(self):
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "activation_function": fullname(self.activation_function),
        }

    def save(self, output_path, safe_serialization: bool = True) -> None:
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut)

        if safe_serialization:
            save_safetensors_model(self, os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    def __repr__(self):
        return f"Dense({self.get_config_dict()})"

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        config["activation_function"] = import_from_string(config["activation_function"])()
        model = Dense(**config)
        if os.path.exists(os.path.join(input_path, "model.safetensors")):
            load_safetensors_model(model, os.path.join(input_path, "model.safetensors"))
        else:
            model.load_state_dict(
                torch.load(
                    os.path.join(input_path, "pytorch_model.bin"), map_location=torch.device("cpu"), weights_only=True
                )
            )
        return model


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
        similarity_fn_name: Union[str, SimilarityFunction] = None,
    ):
        super().__init__()
        self._model_config = {}
        self.similarity_fn_name = similarity_fn_name

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
                self.projector_model = nn.Sequential(
                    Dense(
                        in_features=self.pooling_model.get_pooling_output_dimension(), 
                        out_features=128, 
                        activation_function=nn.LeakyReLU()
                    ), 
                    Dense(
                        in_features=128, 
                        out_features=128, 
                        activation_function=nn.Identity()
                    )
                )
        self.to(self.device)

    def encode(
        self, 
        audios: Union[str, List[str]], 
        batch_size: int = 16, 
        device: str = None, 
        normalize_embeddings: bool = False,
        output_value: Literal["segment_embedding", "frame_embeddings"] = None, 
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
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
            
        if precision and precision != "float32":
            all_embeddings = quantize_embeddings(all_embeddings, precision=precision)

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

    @contextmanager
    def truncate_audio_embeddings(self, truncate_dim: int = None) -> Iterator[None]:
        original_output_dim = self.truncate_dim
        try:
            self.truncate_dim = truncate_dim
            yield
        finally:
            self.truncate_dim = original_output_dim

    def forward(
        self,
        features: Union[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for training or inference.
        """
        features = batch_to_device(features, self.device)

        out_features = self.transformer_model.forward(features)
        out_features = self.pooling_model.forward(out_features)
        out_features = self.projector_model.forward(out_features)
        out_features['segment_embedding'] = F.normalize(out_features['segment_embedding'], p=2, dim=1)

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
    
    def save(
        self,
        path: str,
        safe_serialization: bool = True,
    ) -> None:
        """
        Saves a model and its configuration files to a directory, so that it can be loaded
        with ``SentenceTransformer(path)`` again.

        Args:
            path (str): Path on disc where the model will be saved.
            model_name (str, optional): Optional model name.
            create_model_card (bool, optional): If True, create a README.md with basic information about this model.
            train_datasets (List[str], optional): Optional list with the names of the datasets used to train the model.
            safe_serialization (bool, optional): If True, save the model using safetensors. If False, save the model
                the traditional (but unsafe) PyTorch way.
        """
        if path is None:
            return

        os.makedirs(path, exist_ok=True)

        logger.info(f"Save model to {path}")
        modules_config = []

        # Save some model info
        self._model_config["__version__"] = {
            "transformers": transformers.__version__,
            "pytorch": torch.__version__,
        }

        with open(os.path.join(path, "config_audio_transformers.json"), "w") as fOut:
            config = self._model_config.copy()
            config["similarity_fn_name"] = self.similarity_fn_name
            json.dump(config, fOut, indent=2)

        # Save modules
        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            if type(module).__name__ not in ['Transformer', 'Pooling']:
                continue
            if idx == 0 and hasattr(module, "save_in_root"):  # Save first module in the main folder
                model_path = path + "/"
            else:
                model_path = os.path.join(path, str(idx) + "_" + type(module).__name__)

            os.makedirs(model_path, exist_ok=True)
            # Try to save with safetensors, but fall back to the traditional PyTorch way if the module doesn't support it
            try:
                module.save(model_path, safe_serialization=safe_serialization)
            except TypeError:
                module.save(model_path)

            # "module" only works for Audio Transformers as the modules have the same names as the classes
            class_ref = type(module).__module__
            # For remote modules, we want to remove "transformers_modules.{repo_name}":
            if class_ref.startswith("transformers_modules."):
                class_file = sys.modules[class_ref].__file__

                # Save the custom module file
                dest_file = Path(model_path) / (Path(class_file).name)
                shutil.copy(class_file, dest_file)

                # Save all files importeed in the custom module file
                for needed_file in get_relative_import_files(class_file):
                    dest_file = Path(model_path) / (Path(needed_file).name)
                    shutil.copy(needed_file, dest_file)

                # For remote modules, we want to ignore the "transformers_modules.{repo_id}" part,
                # i.e. we only want the filename
                class_ref = f"{class_ref.split('.')[-1]}.{type(module).__name__}"
            # For other cases, we want to add the class name:
            elif not class_ref.startswith("audio_transformers."):
                class_ref = f"{class_ref}.{type(module).__name__}"
            modules_config.append({"idx": idx, "name": name, "path": os.path.basename(model_path), "type": class_ref})

        with open(os.path.join(path, "modules.json"), "w") as fOut:
            json.dump(modules_config, fOut, indent=2)

    def save_pretrained(
        self,
        path: str,
        safe_serialization: bool = True,
    ) -> None:
        """
        Saves a model and its configuration files to a directory, so that it can be loaded
        with ``AudioTransformer(path)`` again.

        Args:
            path (str): Path on disc where the model will be saved.
            model_name (str, optional): Optional model name.
            create_model_card (bool, optional): If True, create a README.md with basic information about this model.
            train_datasets (List[str], optional): Optional list with the names of the datasets used to train the model.
            safe_serialization (bool, optional): If True, save the model using safetensors. If False, save the model
                the traditional (but unsafe) PyTorch way.
        """
        self.save(
            path,
            safe_serialization=safe_serialization,
        )


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


def quantize_embeddings(
    embeddings: Union[torch.Tensor, np.ndarray],
    precision: Literal["float32", "int8", "uint8", "binary", "ubinary"],
    ranges: np.ndarray = None,
    calibration_embeddings: np.ndarray = None,
) -> np.ndarray:
    """
    Quantizes embeddings to a lower precision. This can be used to reduce the memory footprint and increase the
    speed of similarity search. The supported precisions are "float32", "int8", "uint8", "binary", and "ubinary".

    Args:
        embeddings: Unquantized (e.g. float) embeddings with to quantize
            to a given precision
        precision: The precision to convert to. Options are "float32",
            "int8", "uint8", "binary", "ubinary".
        ranges (Optional[np.ndarray]): Ranges for quantization of
            embeddings. This is only used for int8 quantization, where
            the ranges refers to the minimum and maximum values for each
            dimension. So, it's a 2D array with shape (2,
            embedding_dim). Default is None, which means that the ranges
            will be calculated from the calibration embeddings.
        calibration_embeddings (Optional[np.ndarray]): Embeddings used
            for calibration during quantization. This is only used for
            int8 quantization, where the calibration embeddings can be
            used to compute ranges, i.e. the minimum and maximum values
            for each dimension. Default is None, which means that the
            ranges will be calculated from the query embeddings. This is
            not recommended.

    Returns:
        Quantized embeddings with the specified precision
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    elif isinstance(embeddings, list):
        if isinstance(embeddings[0], torch.Tensor):
            embeddings = [embedding.cpu().numpy() for embedding in embeddings]
        embeddings = np.array(embeddings)
    if embeddings.dtype in (np.uint8, np.int8):
        raise Exception("Embeddings to quantize must be float rather than int8 or uint8.")

    if precision == "float32":
        return embeddings.astype(np.float32)

    if precision.endswith("int8"):
        # Either use the 1. provided ranges, 2. the calibration dataset or 3. the provided embeddings
        if ranges is None:
            if calibration_embeddings is not None:
                ranges = np.vstack((np.min(calibration_embeddings, axis=0), np.max(calibration_embeddings, axis=0)))
            else:
                if embeddings.shape[0] < 100:
                    logger.warning(
                        f"Computing {precision} quantization buckets based on {len(embeddings)} embedding{'s' if len(embeddings) != 1 else ''}."
                        f" {precision} quantization is more stable with `ranges` calculated from more embeddings "
                        "or a `calibration_embeddings` that can be used to calculate the buckets."
                    )
                ranges = np.vstack((np.min(embeddings, axis=0), np.max(embeddings, axis=0)))
        starts = ranges[0, :]
        steps = (ranges[1, :] - ranges[0, :]) / 255

        if precision == "uint8":
            return ((embeddings - starts) / steps).astype(np.uint8)
        elif precision == "int8":
            return ((embeddings - starts) / steps - 128).astype(np.int8)

    if precision == "binary":
        return (np.packbits(embeddings > 0).reshape(embeddings.shape[0], -1) - 128).astype(np.int8)

    if precision == "ubinary":
        return np.packbits(embeddings > 0).reshape(embeddings.shape[0], -1)

    raise ValueError(f"Precision {precision} is not supported")