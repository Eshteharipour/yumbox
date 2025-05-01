import functools
import os

import h5py
import numpy as np

from yumbox.config import BFG


def hd5_cache(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_name = func.__name__
        cache_file = os.path.join(cache_dir, f"{func_name}.h5") if cache_dir else None

        keys = kwargs.get("keys", None)
        features = kwargs.get("cache", {})

        if cache_file and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_file}")
            try:
                with h5py.File(cache_file, "r") as f:
                    # If keys are specified, load only those
                    if keys is not None:
                        for key in keys:
                            if key in f:
                                features[key] = np.array(f[key])
                    # Otherwise, load all keys
                    else:
                        for key in f.keys():
                            features[key] = np.array(f[key])
                logger.info(f"Loaded cache for {func_name} from {cache_file}")
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
                features = {}

        result = func(*args, **kwargs, cache=features, cache_file=cache_file)

        if cache_file:
            logger.info(f"Saving cache for {func_name} to {cache_file}")
            try:
                with h5py.File(cache_file, "w") as f:
                    for key, value in result.items():
                        f.create_dataset(key, data=value, compression="gzip")
                logger.info(f"Saved cache for {func_name} to {cache_file}")
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")

        return result

    return wrapper
