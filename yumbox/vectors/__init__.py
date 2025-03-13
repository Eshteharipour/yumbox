from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def topk(search_func: Callable, queries: np.ndarray, k, is_faiss_kmeans_index=False):
    nn_d = []
    nn = []
    batch_size = 512

    for i in tqdm(range(0, len(queries), batch_size)):
        batch = queries[i : i + batch_size]
        distances, indices = search_func(batch, k=k)

        nn_d.append(distances)
        nn.append(indices)

    if (k == 1 or queries.squeeze().ndim == 1) and is_faiss_kmeans_index == False:
        nn_d = np.array(nn_d).flatten()
        nn = np.array(nn).flatten()
    else:
        nn_d = np.concatenate(nn_d)
        nn = np.concatenate(nn)

    return nn_d, nn


def id2label(
    df: pd.DataFrame,
    nn_d: np.ndarray,
    nn: np.ndarray,
    label_col: str,
    confidence: float,
):
    index2label = dict(zip(df.index, df[label_col]))
    predictions = np.array([index2label[index] for index in nn])
    if confidence:
        certain = nn_d >= confidence
        predictions = np.where(certain, predictions, "-1")
    return predictions


def nested_topk(
    create_index_func: callable,
    search_func_name: str,
    topk_candids: np.ndarray,
    queries: np.ndarray,
):
    nn_d = []
    nn = []

    for c, q in zip(topk_candids, queries):
        index = create_index_func(c)
        q = np.expand_dims(q, axis=0)
        distances, indices = topk(getattr(index, search_func_name), q, k=1)

        nn_d.append(distances)
        nn.append(indices)

    nn_d = np.array(nn_d).flatten()
    nn = np.array(nn).flatten()

    return nn_d, nn


def normalize_vector(v: np.ndarray | torch.Tensor):
    if isinstance(v, np.ndarray):
        assert 0 < v.ndim < 3
        return v / np.linalg.norm(v, axis=-1, keepdims=True)
    elif isinstance(v, torch.Tensor):
        assert 0 < v.dim() < 3
        return v / v.norm(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Expected Numpy Array or Pytorch Tensor, got {type(v)}")


def full_feats(
    df: pd.DataFrame, feats: dict[str, np.ndarray], colname: str
) -> dict[str, np.ndarray]:
    return {x: feats[x] for x in df[colname].values}


def partial_feats(
    df: pd.DataFrame, feats: dict[str, np.ndarray], colname: str
) -> dict[str, np.ndarray]:
    return {x: feats[x] for x in df[colname].values if bool(x) and pd.notna(x)}


def reconstruct_original_index(
    target: np.ndarray | list, missing_indices: np.ndarray | list, fill_value=None
):
    target = list(target)
    for index in missing_indices:
        target.insert(index, fill_value)
    return np.array(target, dtype=object)
