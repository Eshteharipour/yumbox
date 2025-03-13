from . import cache, config, data, factory, metrics, nlp, vectors


def random_seed(rank, seed=362):
    import random

    import numpy as np
    import torch

    torch.backends.cudnn.deterministic = True

    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    np.random.seed(seed + rank)

    random.seed(seed + rank)

    rng = np.random.default_rng(seed + rank)
    return rng
