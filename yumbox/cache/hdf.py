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

        cache = {}
        keys = kwargs.get("features", {}).keys()  # TODO: refactor features

        if cache_file and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_file}")
            try:
                with h5py.File(cache_file, "r") as f:
                    # If keys are specified, load only those
                    for key in keys:
                        if key in f:
                            cache[key] = np.array(f[key])
                    # load all keys:
                    # for key in f.keys():
                    #     cache[key] = np.array(f[key])
                logger.info(f"Loaded cache for {func_name} from {cache_file}")
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
                cache = {}

        result = func(*args, **kwargs, cache=cache, cache_file=cache_file)

        if cache_file:
            logger.info(f"Saving cache for {func_name} to {cache_file}")
            try:
                with h5py.File(cache_file, "a") as f:  # 'a': preserve previous content
                    for key, value in result.items():
                        if key in f:
                            del f[key]  # overwrite cache
                        f.create_dataset(key, data=value, compression="gzip")
                logger.info(f"Saved cache for {func_name} to {cache_file}")
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")

        return result

    return wrapper
