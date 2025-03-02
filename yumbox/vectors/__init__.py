import numpy as np
import pandas as pd
from tqdm import tqdm


def topk(search_func: callable, queries: np.ndarray, k, is_faiss_kmeans_index=False):
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
