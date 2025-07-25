import base64
import functools
import os

import h5py
import numpy as np

from yumbox.config import BFG


def _encode_key(key):
    """Encode a key to be HDF5-safe using base64."""
    if isinstance(key, str):
        key = key.encode("utf-8")
    elif not isinstance(key, bytes):
        # Handle other types by converting to string first
        key = str(key).encode("utf-8")
    return base64.b64encode(key).decode("ascii")


def _decode_key(encoded_key):
    """Decode a base64-encoded key back to original bytes."""
    return base64.b64decode(encoded_key.encode("ascii"))


def hd5_cache(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_name = func.__name__
        cache_file = os.path.join(cache_dir, f"{func_name}.h5") if cache_dir else None

        cache = {}
        keys = kwargs.get("keys", [])
        keys = list(keys)

        if cache_file and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_file}")
            try:
                with h5py.File(cache_file, "r") as f:
                    # If keys are specified, load only those
                    if keys:
                        for key in keys:
                            encoded_key = _encode_key(key)
                            if encoded_key in f:
                                cache[key] = np.array(f[encoded_key])
                    else:
                        for key in f.keys():
                            decoded_key = _decode_key(key)
                            cache[decoded_key] = np.array(f[key])
                logger.info(f"Loaded cache for {func_name} from {cache_file}")
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
                cache = {}

        cache = dict(keys=cache.keys(), values=cache.values())
        result = func(*args, **kwargs, cache=cache, cache_file=cache_file)

        if cache_file:
            logger.info(f"Saving cache for {func_name} to {cache_file}")
            try:
                with h5py.File(cache_file, "a") as f:  # 'a': preserve previous content
                    for key, value in result.items():
                        encoded_key = _encode_key(key)

                        if encoded_key in f:
                            del f[encoded_key]

                        f.create_dataset(encoded_key, data=value, compression="gzip")

                logger.info(f"Saved cache for {func_name} to {cache_file}")
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")

        return result

    return wrapper
