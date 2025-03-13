import numpy as np
from tqdm import tqdm

from .faiss_indexes import FaissIndexBuilder


def pca(features: np.ndarray, n_components=128):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=362)
    transformed = pca.fit_transform(features)

    return transformed


def pca_faiss(features: np.ndarray, n_components=128):
    import faiss

    embed_size = features[0].shape[-1]
    mat = faiss.PCAMatrix(embed_size, n_components)
    mat.train(features)
    assert mat.is_trained
    transformed = mat.apply(features)
    return transformed


def kmeans_faiss(features: np.ndarray, ncentroids=1024, niter=20, verbose=False):
    import faiss

    embed_size = features[0].shape[-1]
    kmeans = faiss.Kmeans(embed_size, ncentroids, niter=niter, verbose=verbose)
    kmeans.train(features)
    return kmeans


def build_index(features: np.ndarray):
    import faiss

    batch_size = 32 * 16
    embed_size = features[0].shape[-1]
    index = faiss.IndexFlatIP(embed_size)
    # index = faiss.index_factory(embed_size, "NSG,Flat") # error: NNDescent.build cannot build a graph smaller than 100
    # index = faiss.index_factory(embed_size, "HNSW,Flat")
    # GPU
    # resources = faiss.StandardGpuResources()
    # index = faiss.index_binary_cpu_to_gpu(resources, 0, index)

    # Add to index in batch
    # for i in tqdm(range(0, len(features), batch_size)):
    #     batch = features[i : i + batch_size]
    #     index.add(batch)

    # Add to index in one go
    index.add(features)

    return index


def build_index_l2(features: np.ndarray):
    import faiss

    embed_size = features[0].shape[-1]
    index = faiss.IndexFlatL2(embed_size)
    index.add(features)

    return index


def self_similarity(x: np.ndarray, batch_size=2048):
    """
    Compute similarity matrix for x@x.T using FAISS IP index with optimizations.

    Args:
        x (np.ndarray): Input array of shape (n_samples, n_features)

    Returns:
        np.ndarray: Similarity matrix of shape (n_samples, n_samples)

    >>> np.random.seed(42)

    >>> x = np.random.randn(5, 3).astype(np.float32)

    >>> x = x / np.linalg.norm(x, axis=1)[:, np.newaxis]

    >>> assert np.max(np.abs(self_similarity(x) - x@x.T)) < 1e-6

    """

    # Ensure x is float32 for FAISS
    # x = np.ascontiguousarray(x, dtype=np.float32)
    n_samples, n_features = x.shape

    sim_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
    np.fill_diagonal(sim_matrix, 1.0)

    for i in tqdm(range(0, n_samples - 1)):
        q = x[i : i + 1]
        agg_distances = []
        agg_indices = []
        for j in range(i + 1, n_samples, batch_size):
            c = x[j : j + batch_size]
            ci = build_index(c)
            D, I = ci.search(q, k=len(c))
            D = D.reshape(-1)
            I = I.reshape(-1)
            agg_distances.append(D)
            I = I + 1 + i
            agg_indices.append(I)

        agg_distances = np.concatenate(agg_distances)
        agg_indices = np.concatenate(agg_indices)

        sim_matrix[i, agg_indices] = agg_distances
        sim_matrix[agg_indices, i] = agg_distances

    return sim_matrix
