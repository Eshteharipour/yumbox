from collections.abc import Callable, Iterable
from typing import Literal

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

no_op = lambda x: x


def topk(
    search_func: Callable,
    queries: np.ndarray | list,
    k: int,
    keepdims=False,
    search_size: int | None = None,
):
    # keepdims: for faiss kmeans clusters topk, pass keepdims=True
    # renamed is_faiss_kmeans_index arg to keepdims

    # sparse array length issue
    try:
        queries_len = len(queries)
    except TypeError:
        queries_len = queries.shape[0]

    if queries_len == 1:
        if not hasattr(queries[0], "__iter__"):
            if isinstance(queries, np.ndarray):
                queries = np.expand_dims(queries, axis=0)
            else:
                queries = [queries]

    nn_d = []
    nn = []
    batch_size = 512
    if search_size:
        batch_size = search_size // k

    for i in tqdm(range(0, queries_len, batch_size)):
        batch = queries[i : i + batch_size]
        distances, indices = search_func(batch, k=k)

        nn_d.append(distances)
        nn.append(indices)

    if keepdims == False and (k == 1 or queries_len == 1):
        nn_d = np.concatenate(nn_d).flatten()
        nn = np.concatenate(nn).flatten()
    else:
        nn_d = np.concatenate(nn_d)
        nn = np.concatenate(nn)

    return nn_d, nn


def id2label(
    df: pd.DataFrame,
    nn_d: np.ndarray,
    nn: np.ndarray,
    label_col: str,
    confidence: int | float,
):
    if hasattr(nn[0], "__iter__"):
        k = 2
    else:
        k = 1

    index2label = dict(zip(np.arange(len(df)).tolist(), df[label_col]))
    if k == 1:
        predictions = np.array([index2label[index] for index in nn])
    else:
        predictions = np.array([[index2label[index] for index in topk] for topk in nn])
    if isinstance(confidence, (int, float)):
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


def notfona(x):
    if bool(x) and pd.notna(x):
        return True
    return False


def full_featdict(
    df: pd.DataFrame, feats: dict[str, np.ndarray], colname: str
) -> dict[str, np.ndarray]:
    # colname is id col
    return {x: feats[x] for x in df[colname].values}


def partial_featdict(
    df: pd.DataFrame, feats: dict[str, np.ndarray], colname: str
) -> dict[str, np.ndarray]:
    # colname is id col
    return {x: feats[x] for x in df[colname].values if notfona(x)}


def sum_feats(
    df: pd.DataFrame,
    feats_a: dict[str, np.ndarray],
    feats_b: dict[str, np.ndarray],
    colname_a: str,
    colname_b: str,
    normalize: Literal["before", "after", None] = None,
) -> np.ndarray:
    # colname_a is id col for feats_a
    # colname_b is id col for feats_b
    # allows missing value on either col a or col b
    if normalize == None or normalize == "after":
        x = np.array(
            [
                (
                    feats_a[r[colname_a]] + feats_b[r[colname_b]]
                    if (notfona(r[colname_a]) and notfona(r[colname_b]))
                    else (
                        feats_a[r[colname_a]]
                        if (notfona(r[colname_a]))
                        else feats_b[r[colname_b]]
                    )
                )
                for _, r in df.iterrows()
            ]
        )
        if normalize == None:
            return x
        else:
            return normalize_vector(x)
    elif normalize == "before":
        return np.array(
            [
                (
                    normalize_vector(feats_a[r[colname_a]])
                    + normalize_vector(feats_b[r[colname_b]])
                    if (notfona(r[colname_a]) and notfona(r[colname_b]))
                    else (
                        normalize_vector(feats_a[r[colname_a]])
                        if (notfona(r[colname_a]))
                        else normalize_vector(feats_b[r[colname_b]])
                    )
                )
                for _, r in df.iterrows()
            ]
        )
    else:
        raise ValueError(normalize)


def cat_feats(
    df: pd.DataFrame,
    feats_a: dict[str, np.ndarray],
    feats_b: dict[str, np.ndarray],
    colname_a: str,
    colname_b: str,
    zeros_a: np.ndarray | None = None,
    zeros_b: np.ndarray | None = None,
    normalize: Literal["before", "after", None] = None,
    pca_a: Callable | None = no_op,
    pca_b: Callable | None = no_op,
) -> np.ndarray:
    # colname_a is id col for feats_a
    # colname_b is id col for feats_b
    # expects feats_a and feats_b to not have missing values if zeros_a and zero_b not provided
    if pca_a is None:
        pca_a = no_op
    if pca_b is None:
        pca_b = no_op
    if zeros_a is None and zeros_b is None:
        if normalize == None or normalize_vector == "after":
            x = np.array(
                np.concatenate(
                    [pca_a(feats_a[r[colname_a]]), pca_b(feats_b[r[colname_b]])]
                )
                for _, r in df.iterrows()
            )
            if normalize == None:
                return x
            else:
                return normalize_vector(x)
        elif normalize == "before":
            return np.array(
                np.concatenate(
                    [
                        pca_a(normalize_vector(feats_a[r[colname_a]])),
                        pca_b(normalize_vector(feats_b[r[colname_b]])),
                    ]
                )
                for _, r in df.iterrows()
            )
        else:
            raise ValueError(normalize)
    else:
        if normalize == None or normalize == "after":
            x = np.array(
                [
                    (
                        np.concatenate(
                            [pca_a(feats_a[r[colname_a]]), pca_b(feats_b[r[colname_b]])]
                        )
                        if (notfona(r[colname_a]) and notfona(r[colname_b]))
                        else (
                            [
                                pca_a(feats_a[r[colname_a]]),
                                zeros_b,
                            ]
                            if (notfona(r[colname_a]))
                            else [
                                zeros_a,
                                pca_b(feats_b[r[colname_b]]),
                            ]
                        )
                    )
                    for _, r in df.iterrows()
                ]
            )
            if normalize == None:
                return x
            else:
                return normalize_vector(x)
        elif normalize == "before":
            return np.array(
                [
                    (
                        np.concatenate(
                            [
                                pca_a(normalize_vector(feats_a[r[colname_a]])),
                                pca_b(normalize_vector(feats_b[r[colname_b]])),
                            ]
                        )
                        if (notfona(r[colname_a]) and notfona(r[colname_b]))
                        else (
                            np.concatenate(
                                [
                                    pca_a(normalize_vector(feats_a[r[colname_a]])),
                                    zeros_b,
                                ]
                            )
                            if (notfona(r[colname_a]))
                            else np.concatenate(
                                [
                                    zeros_a,
                                    pca_b(normalize_vector(feats_b[r[colname_b]])),
                                ]
                            )
                        )
                    )
                    for _, r in df.iterrows()
                ]
            )

        else:
            raise ValueError(normalize)


def full_feats(keys: Iterable[str], feats: dict[str]) -> dict[str]:
    return np.array([feats[k] for k in keys])


def partial_feats(keys: Iterable[str], feats: dict[str]) -> dict[str]:
    return np.array([feats[k] for k in keys if notfona(k)])


def reconstruct_original_index(
    target: np.ndarray | list, missing_indices: np.ndarray | list, fill_value=None
):
    target = list(target)
    for index in missing_indices:
        target.insert(index, fill_value)
    return np.array(target, dtype=object)
