from typing import Literal

from . import cache, config, data, factory, metrics, nlp, vectors

RNG = {}


def random_seed(rank, seed=362, generator: Literal["numpy", "torch", "int"] = "numpy"):
    import random

    import numpy as np
    import torch

    # seed = torch.initial_seed()

    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False  # across multiple executions
    # torch.use_deterministic_algorithms(True)  # impacts performance

    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    np.random.seed(seed + rank)

    random.seed(seed + rank)

    if generator == "numpy":
        rng = np.random.default_rng(seed + rank)
    elif generator == "torch":
        rng = torch.Generator()
        rng.manual_seed(seed + rank)
    else:
        rng = seed + rank

    RNG["rng"] = rng

    return rng


def worker_seed(worker_id):
    import torch

    rank = torch.initial_seed() % 2**32
    random_seed(rank)
