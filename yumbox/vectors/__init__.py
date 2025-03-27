from collections.abc import Callable, Iterable
from typing import Literal

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def topk(
    search_func: Callable,
    queries: np.ndarray,
    k: int,
    keepdims=False,
    search_size: int | None = None,
):
    # keepdims: for faiss kmeans clusters topk, pass keepdims=True
    # renamed is_faiss_kmeans_index arg to keepdims

    nn_d = []
    nn = []
    batch_size = 512
    if search_size:
        batch_size = search_size // k

    for i in tqdm(range(0, queries.shape[0], batch_size)):
        batch = queries[i : i + batch_size]
        distances, indices = search_func(batch, k=k)

        nn_d.append(distances)
        nn.append(indices)

    if (k == 1 or queries.squeeze().ndim == 1) and keepdims == False:
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
    confidence: float,
):
    if hasattr(nn[0], "__iter__"):
        k = len(nn[0])
    else:
        k = 1

    index2label = dict(zip(df.index, df[label_col]))
    if k == 1:
        predictions = np.array([index2label[index] for index in nn])
    else:
        predictions = np.array([[index2label[index] for index in topk] for topk in nn])
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
) -> dict[str, np.ndarray]:
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
        ValueError(normalize)


def cat_feats(
    df: pd.DataFrame,
    feats_a: dict[str, np.ndarray],
    feats_b: dict[str, np.ndarray],
    colname_a: str,
    colname_b: str,
    zeros_a: int | None = None,
    zeros_b: int | None = None,
    normalize: Literal["before", "after", None] = None,
    pca: Callable = lambda x: x,
) -> dict[str, np.ndarray]:
    # colname_a is id col for feats_a
    # colname_b is id col for feats_b
    # expects feats_a and feats_b to not have missing values if fill_size not provided
    if zeros_a == None and zeros_b == None:
        if normalize == None or normalize_vector == "after":
            x = np.concatenate(
                np.array([pca(feats_a[r[colname_a]]), pca(feats_b[r[colname_b]])])
                for _, r in df.iterrows()
            )
            if normalize == None:
                return x
            else:
                return normalize_vector(x)
        elif normalize == "before":
            return np.concatenate(
                np.array(
                    [
                        pca(normalize_vector(feats_a[r[colname_a]])),
                        pca(normalize_vector(feats_b[r[colname_b]])),
                    ]
                )
                for _, r in df.iterrows()
            )
        else:
            ValueError(normalize)
    else:
        if normalize == None or normalize == "after":
            x = np.concatenate(
                [
                    (
                        [pca(feats_a[r[colname_a]]), pca(feats_b[r[colname_b]])]
                        if (notfona(r[colname_a]) and notfona(r[colname_b]))
                        else (
                            [
                                pca(feats_a[r[colname_a]]),
                                zeros_b,
                            ]
                            if (notfona(r[colname_a]))
                            else [
                                zeros_a,
                                pca(feats_b[r[colname_b]]),
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
            return np.concatenate(
                [
                    (
                        [
                            pca(normalize_vector(feats_a[r[colname_a]])),
                            pca(normalize_vector(feats_b[r[colname_b]])),
                        ]
                        if (notfona(r[colname_a]) and notfona(r[colname_b]))
                        else (
                            [
                                pca(normalize_vector(feats_a[r[colname_a]])),
                                zeros_b,
                            ]
                            if (notfona(r[colname_a]))
                            else [
                                zeros_a,
                                pca(normalize_vector(feats_b[r[colname_b]])),
                            ]
                        )
                    )
                    for _, r in df.iterrows()
                ]
            )

        else:
            ValueError(normalize)


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
