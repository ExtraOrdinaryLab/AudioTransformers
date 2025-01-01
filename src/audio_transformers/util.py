from typing import Union, Any

import numpy as np

import torch
from torch import Tensor


def _convert_to_tensor(a: Union[list, np.ndarray, Tensor]) -> Tensor:
    """
    Converts the input `a` to a PyTorch tensor if it is not already a tensor.

    Args:
        a (Union[list, np.ndarray, Tensor]): The input array or tensor.

    Returns:
        Tensor: The converted tensor.
    """
    if not isinstance(a, Tensor):
        a = torch.tensor(a)
    return a


def _convert_to_batch(a: Tensor) -> Tensor:
    """
    If the tensor `a` is 1-dimensional, it is unsqueezed to add a batch dimension.

    Args:
        a (Tensor): The input tensor.

    Returns:
        Tensor: The tensor with a batch dimension.
    """
    if a.dim() == 1:
        a = a.unsqueeze(0)
    return a


def _convert_to_batch_tensor(a: Union[list, np.ndarray, Tensor]) -> Tensor:
    """
    Converts the input data to a tensor with a batch dimension.

    Args:
        a (Union[list, np.ndarray, Tensor]): The input data to be converted.

    Returns:
        Tensor: The converted tensor with a batch dimension.
    """
    a = _convert_to_tensor(a)
    a = _convert_to_batch(a)
    return a


def pytorch_cos_sim(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the cosine similarity between two tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = cos_sim(a[i], b[j])
    """
    return cos_sim(a, b)


def cos_sim(a: Union[list, np.ndarray, Tensor], b: Union[list, np.ndarray, Tensor]) -> Tensor:
    """
    Computes the cosine similarity between two tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = cos_sim(a[i], b[j])
    """
    a = _convert_to_batch_tensor(a)
    b = _convert_to_batch_tensor(b)

    a_norm = normalize_embeddings(a)
    b_norm = normalize_embeddings(b)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def pairwise_cos_sim(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the pairwise cosine similarity cos_sim(a[i], b[i]).

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Vector with res[i] = cos_sim(a[i], b[i])
    """
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    return pairwise_dot_score(normalize_embeddings(a), normalize_embeddings(b))


def dot_score(a: Union[list, np.ndarray, Tensor], b: Union[list, np.ndarray, Tensor]) -> Tensor:
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = dot_prod(a[i], b[j])
    """
    a = _convert_to_batch_tensor(a)
    b = _convert_to_batch_tensor(b)

    return torch.mm(a, b.transpose(0, 1))


def pairwise_dot_score(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the pairwise dot-product dot_prod(a[i], b[i]).

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Vector with res[i] = dot_prod(a[i], b[i])
    """
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    return (a * b).sum(dim=-1)


def manhattan_sim(a: Union[list, np.ndarray, Tensor], b: Union[list, np.ndarray, Tensor]) -> Tensor:
    """
    Computes the manhattan similarity (i.e., negative distance) between two tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = -manhattan_distance(a[i], b[j])
    """
    a = _convert_to_batch_tensor(a)
    b = _convert_to_batch_tensor(b)

    return -torch.cdist(a, b, p=1.0)


def pairwise_manhattan_sim(a: Union[list, np.ndarray, Tensor], b: Union[list, np.ndarray, Tensor]):
    """
    Computes the manhattan similarity (i.e., negative distance) between pairs of tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Vector with res[i] = -manhattan_distance(a[i], b[i])
    """
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    return -torch.sum(torch.abs(a - b), dim=-1)


def euclidean_sim(a: Union[list, np.ndarray, Tensor], b: Union[list, np.ndarray, Tensor]) -> Tensor:
    """
    Computes the euclidean similarity (i.e., negative distance) between two tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = -euclidean_distance(a[i], b[j])
    """
    a = _convert_to_batch_tensor(a)
    b = _convert_to_batch_tensor(b)

    return -torch.cdist(a, b, p=2.0)


def pairwise_euclidean_sim(a: Union[list, np.ndarray, Tensor], b: Union[list, np.ndarray, Tensor]):
    """
    Computes the euclidean distance (i.e., negative distance) between pairs of tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Vector with res[i] = -euclidean_distance(a[i], b[i])
    """
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    return -torch.sqrt(torch.sum((a - b) ** 2, dim=-1))


def pairwise_angle_sim(x: Tensor, y: Tensor) -> Tensor:
    """
    Computes the absolute normalized angle distance. See :class:`~sentence_transformers.losses.AnglELoss`
    or https://arxiv.org/abs/2309.12871v1 for more information.

    Args:
        x (Tensor): The first tensor.
        y (Tensor): The second tensor.

    Returns:
        Tensor: Vector with res[i] = angle_sim(a[i], b[i])
    """

    x = _convert_to_tensor(x)
    y = _convert_to_tensor(y)

    # modified from https://github.com/SeanLee97/AnglE/blob/main/angle_emb/angle.py
    # chunk both tensors to obtain complex components
    a, b = torch.chunk(x, 2, dim=1)
    c, d = torch.chunk(y, 2, dim=1)

    z = torch.sum(c**2 + d**2, dim=1, keepdim=True)
    re = (a * c + b * d) / z
    im = (b * c - a * d) / z

    dz = torch.sum(a**2 + b**2, dim=1, keepdim=True) ** 0.5
    dw = torch.sum(c**2 + d**2, dim=1, keepdim=True) ** 0.5
    re /= dz / dw
    im /= dz / dw

    norm_angle = torch.sum(torch.concat((re, im), dim=1), dim=1)
    return torch.abs(norm_angle)


def normalize_embeddings(embeddings: Tensor) -> Tensor:
    """
    Normalizes the embeddings matrix, so that each sentence embedding has unit length.

    Args:
        embeddings (Tensor): The input embeddings matrix.

    Returns:
        Tensor: The normalized embeddings matrix.
    """
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def batch_to_device(batch: dict[str, Any], target_device: torch.device) -> dict[str, Any]:
    """
    Send a PyTorch batch (i.e., a dictionary of string keys to Tensors) to a device (e.g. "cpu", "cuda", "mps").

    Args:
        batch (Dict[str, Tensor]): The batch to send to the device.
        target_device (torch.device): The target device (e.g. "cpu", "cuda", "mps").

    Returns:
        Dict[str, Tensor]: The batch with tensors sent to the target device.
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch