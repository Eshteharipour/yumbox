from time import time

import faiss
import numpy as np


class FaissIndexBuilder:
    """
    A class containing static methods to build different FAISS indexes for testing search speeds.
    All indexes are configured for inner product similarity (METRIC_INNER_PRODUCT).
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the FaissIndexBuilder with a verbose flag.

        Args:
            verbose (bool): If True, print detailed information (default: False)
        """
        self.verbose = verbose

    def normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalizes vectors to unit length (L2 normalization) for cosine similarity.

        Args:
            vectors (np.ndarray): Input vectors [n_samples, embed_size]

        Returns:
            np.ndarray: Normalized vectors
        """
        normalized = vectors.copy()
        faiss.normalize_L2(normalized)
        return normalized

    def build_flat_ip_index(
        self, features: np.ndarray, use_gpu: bool = False
    ) -> faiss.Index:
        """
        Builds a FlatIP index - exact search baseline.
        Pros: Highest accuracy
        Cons: Slowest search, highest memory usage

        Args:
            features (np.ndarray): Feature vectors [n_samples, embed_size]
            use_gpu (bool): Use GPU acceleration if available (default: False)

        Returns:
            faiss.Index: Built FlatIP index
        """
        embed_size = features.shape[1]
        index = faiss.IndexFlatIP(embed_size)

        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        start_time = time()
        index.add(features)
        build_time = time() - start_time

        if self.verbose:
            print(f"FlatIP Index built in {build_time:.3f} seconds")
            print(f"Index size: {index.ntotal} vectors")
            print(f"Dimension: {embed_size}")
        else:
            print(f"FlatIP Index built in {build_time:.3f} seconds")

        return index

    def build_ivf_index(
        self,
        features: np.ndarray,
        nlist: int = 100,
        nprobe: int = 1,
        use_gpu: bool = False,
    ) -> faiss.Index:
        """
        Builds an IVF (Inverted File) index - partitions space into cells.
        Pros: Good speed-accuracy balance
        Cons: Requires training, sensitive to nlist

        Args:
            features (np.ndarray): Feature vectors [n_samples, embed_size]
            nlist (int): Number of cells/clusters (default: 100)
            nprobe (int): Number of cells to probe during search (default: 1)
            use_gpu (bool): Use GPU acceleration if available (default: False)

        Returns:
            faiss.Index: Built IVF index
        """
        embed_size = features.shape[1]
        quantizer = faiss.IndexFlatIP(embed_size)
        index = faiss.IndexIVFFlat(
            quantizer, embed_size, nlist, faiss.METRIC_INNER_PRODUCT
        )
        index.nprobe = nprobe

        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        start_time = time()
        index.train(features)
        index.add(features)
        build_time = time() - start_time

        if self.verbose:
            print(f"IVF Index (nlist={nlist}) built in {build_time:.3f} seconds")
            print(f"Index size: {index.ntotal} vectors")
            print(f"Number of clusters: {nlist}")
            print(f"Probes per query: {nprobe}")
        else:
            print(f"IVF Index (nlist={nlist}) built in {build_time:.3f} seconds")

        return index

    def build_hnsw_index(
        self,
        features: np.ndarray,
        M: int = 32,
        efConstruction: int = 200,
        use_gpu: bool = False,
    ) -> faiss.Index:
        """
        Builds an HNSW (Hierarchical Navigable Small World) index - graph-based.
        Pros: Fast search, good accuracy-memory trade-off
        Cons: Slower to build

        Args:
            features (np.ndarray): Feature vectors [n_samples, embed_size]
            M (int): Number of neighbors per node (default: 32)
            efConstruction (int): Size of dynamic list during construction (default: 200)
            use_gpu (bool): Use GPU acceleration if available (default: False)

        Returns:
            faiss.Index: Built HNSW index
        """
        embed_size = features.shape[1]
        index = faiss.IndexHNSWFlat(embed_size, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = efConstruction

        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        start_time = time()
        index.add(features)
        build_time = time() - start_time

        if self.verbose:
            print(
                f"HNSW Index (M={M}, efConstruction={efConstruction}) built in {build_time:.3f} seconds"
            )
            print(f"Index size: {index.ntotal} vectors")
            print(f"Neighbors per node: {M}")
            print(f"Construction search size: {efConstruction}")
        else:
            print(
                f"HNSW Index (M={M}, efConstruction={efConstruction}) built in {build_time:.3f} seconds"
            )

        return index

    def build_pq_index(
        self, features: np.ndarray, m: int = 8, nbits: int = 8, use_gpu: bool = False
    ) -> faiss.Index:
        """
        Builds a PQ (Product Quantization) index - compresses vectors.
        Pros: Memory efficient, fast search
        Cons: Lower accuracy due to compression

        Args:
            features (np.ndarray): Feature vectors [n_samples, embed_size]
            m (int): Number of subquantizers (default: 8)
            nbits (int): Number of bits per subquantizer (default: 8)
            use_gpu (bool): Use GPU acceleration if available (default: False)

        Returns:
            faiss.Index: Built PQ index
        """
        embed_size = features.shape[1]
        index = faiss.IndexPQ(embed_size, m, nbits, faiss.METRIC_INNER_PRODUCT)

        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        start_time = time()
        index.train(features)
        index.add(features)
        build_time = time() - start_time

        if self.verbose:
            print(f"PQ Index (m={m}, nbits={nbits}) built in {build_time:.3f} seconds")
            print(f"Index size: {index.ntotal} vectors")
            print(f"Subquantizers: {m}")
            print(f"Bits per subquantizer: {nbits}")
        else:
            print(f"PQ Index (m={m}, nbits={nbits}) built in {build_time:.3f} seconds")

        return index

    def build_ivfpq_index(
        self,
        features: np.ndarray,
        nlist: int = 100,
        m: int = 8,
        nbits: int = 8,
        nprobe: int = 1,
        use_gpu: bool = False,
    ) -> faiss.Index:
        """
        Builds an IVFPQ (Inverted File with Product Quantization) index - combines IVF and PQ.
        Pros: Scalable, memory efficient
        Cons: Approximate results, requires training

        Args:
            features (np.ndarray): Feature vectors [n_samples, embed_size]
            nlist (int): Number of cells/clusters (default: 100)
            m (int): Number of subquantizers (default: 8)
            nbits (int): Number of bits per subquantizer (default: 8)
            nprobe (int): Number of cells to probe during search (default: 1)
            use_gpu (bool): Use GPU acceleration if available (default: False)

        Returns:
            faiss.Index: Built IVFPQ index
        """
        embed_size = features.shape[1]
        quantizer = faiss.IndexFlatIP(embed_size)
        index = faiss.IndexIVFPQ(
            quantizer, embed_size, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT
        )
        index.nprobe = nprobe

        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        start_time = time()
        index.train(features)
        index.add(features)
        build_time = time() - start_time

        if self.verbose:
            print(
                f"IVFPQ Index (nlist={nlist}, m={m}, nbits={nbits}) built in {build_time:.3f} seconds"
            )
            print(f"Index size: {index.ntotal} vectors")
            print(f"Number of clusters: {nlist}")
            print(f"Subquantizers: {m}")
            print(f"Bits per subquantizer: {nbits}")
            print(f"Probes per query: {nprobe}")
        else:
            print(
                f"IVFPQ Index (nlist={nlist}, m={m}, nbits={nbits}) built in {build_time:.3f} seconds"
            )

        return index


def test_indexes(
    features: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
    use_gpu: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Tests build and search times for all index types.

    Args:
        features (np.ndarray): Feature vectors to index [n_samples, embed_size]
        queries (np.ndarray): Query vectors [n_queries, embed_size]
        k (int): Number of nearest neighbors to search (default: 10)
        use_gpu (bool): Use GPU acceleration if available (default: False)

    Returns:
        dict: Results with build times, search times, distances, and indices
    """
    builder = FaissIndexBuilder(verbose=verbose)

    # Normalize for cosine similarity (inner product)
    features_norm = builder.normalize_vectors(features)
    queries_norm = builder.normalize_vectors(queries)

    index_types = {
        "FlatIP": builder.build_flat_ip_index,
        "IVF": builder.build_ivf_index,
        "HNSW": builder.build_hnsw_index,
        "PQ": builder.build_pq_index,
        "IVFPQ": builder.build_ivfpq_index,
    }

    results = {}
    for name, build_func in index_types.items():
        if verbose:
            print(f"\nTesting {name} Index")
            print("-" * 20)
        else:
            print(f"\nTesting {name} Index")

        # Build index with normalized features
        index = build_func(features_norm, use_gpu=use_gpu)

        # Test search speed with normalized queries
        start_time = time()
        distances, indices = index.search(queries_norm, k)
        search_time = time() - start_time

        if verbose:
            print(f"Search time for {k} neighbors: {search_time:.4f} seconds")
            print(f"Average distance: {np.mean(distances):.4f}")
            print(f"Result shape: distances={distances.shape}, indices={indices.shape}")
        else:
            print(f"Search time for {k} neighbors: {search_time:.4f} seconds")

        results[name] = {
            "distances": distances,
            "indices": indices,
            "search_time": search_time,
        }

    return results
