import numpy as np

# def pca(features: dict[str, np.ndarray], n_components=128):
#     from sklearn.decomposition import PCA

#     pca = PCA(n_components=n_components, random_state=362)
#     transformed = pca.fit_transform(np.stack(list(features.values())))


#     features = dict(zip(features.keys(), transformed))
#     return features


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
