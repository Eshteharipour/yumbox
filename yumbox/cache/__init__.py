import functools
import hashlib
import json
import os
import time
from collections.abc import Callable
from time import sleep

import numpy as np
from safetensors.numpy import load_file, save_file

from yumbox.config import BFG

from .fs import *
from .hdf import hd5_cache, hd5_cache_kwargs_list_hash
from .kv import (
    LMDB,
    LMDBMultiIndex,
    VectorLMDB,
    lmdb_cache,
    lmdb_cache_kwargs_list_hash,
)

# TODO: fix safe_save_kw keys= values=


def cache(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_name = func.__name__
        cache_file = os.path.join(cache_dir, f"{func_name}.pkl") if cache_dir else ""

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_file}")
            result = safe_rpickle(cache_file)
            logger.info(f"Loaded cache for {func_name} from {cache_file}")
            return result
            # with open(cache_file, "rb") as fd:
            #     return pickle.load(fd)
        else:
            result = func(*args, **kwargs)
            if cache_dir:
                logger.info(f"Saving cache for {func_name} to {cache_file}")
                safe_wpickle(cache_file, result)
                # with open(cache_file, "wb") as fd:
                #     pickle.dump(result, fd)
                logger.info(f"Saved cache!")
            return result

    return wrapper


def timed_cache(max_age_hours=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_dir = BFG["cache_dir"]
            logger = BFG["logger"]

            func_name = func.__name__
            cache_file = (
                os.path.join(cache_dir, f"{func_name}.pkl") if cache_dir else ""
            )

            # Check if cache file exists and is recent enough
            if cache_dir and os.path.isfile(cache_file):
                # Get file's last modified time
                file_mtime = os.path.getmtime(cache_file)
                current_time = time.time()
                # Convert max_age_hours to seconds
                max_age_seconds = max_age_hours * 3600

                # Check if file was modified within max_age_hours
                if (current_time - file_mtime) <= max_age_seconds:
                    logger.info(f"Loading cache for {func_name} from {cache_file}")
                    result = safe_rpickle(cache_file)
                    logger.info(f"Loaded cache for {func_name} from {cache_file}")
                    return result

            # Either no cache file or cache is too old
            result = func(*args, **kwargs)
            if cache_dir:
                logger.info(f"Saving cache for {func_name} to {cache_file}")
                safe_wpickle(cache_file, result)
                logger.info(f"Saved cache!")
            return result

        return wrapper

    return decorator


def timed_cache_kwargs(max_age_hours=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_dir = BFG["cache_dir"]
            logger = BFG["logger"]

            func_name = func.__name__
            func_kwargs = []
            for k, v in sorted(kwargs.items()):
                func_kwargs.append(f"{k}-{v}")
            func_kwargs = ",".join(func_kwargs)
            cache_file = (
                os.path.join(cache_dir, f"{func_name}_{func_kwargs}.pkl")
                if cache_dir and func_kwargs
                else os.path.join(cache_dir, f"{func_name}.pkl") if cache_dir else ""
            )

            # Check if cache file exists and is recent enough
            if cache_dir and os.path.isfile(cache_file):
                # Get file's last modified time
                file_mtime = os.path.getmtime(cache_file)
                current_time = time.time()
                # Convert max_age_hours to seconds
                max_age_seconds = max_age_hours * 3600

                # Check if file was modified within max_age_hours
                if (current_time - file_mtime) <= max_age_seconds:
                    logger.info(f"Loading cache for {func_name} from {cache_file}")
                    result = safe_rpickle(cache_file)
                    logger.info(f"Loaded cache for {func_name} from {cache_file}")
                    return result

            # Either no cache file or cache is too old
            result = func(*args, **kwargs)
            if cache_dir:
                logger.info(f"Saving cache for {func_name} to {cache_file}")
                safe_wpickle(cache_file, result)
                logger.info(f"Saved cache!")
            return result

        return wrapper

    return decorator


def cache_kwargs(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_name = func.__name__
        func_kwargs = []
        for k, v in sorted(kwargs.items()):
            func_kwargs.append(f"{k}-{v}")
        func_kwargs = ",".join(func_kwargs)
        cache_file = (
            os.path.join(cache_dir, f"{func_name}_{func_kwargs}.pkl")
            if cache_dir and func_kwargs
            else os.path.join(cache_dir, f"{func_name}.pkl") if cache_dir else ""
        )

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_file}")
            result = safe_rpickle(cache_file)
            logger.info(f"Loaded cache for {func_name} from {cache_file}")
            return result
            # with open(cache_file, "rb") as fd:
            #     return pickle.load(fd)
        else:
            result = func(*args, **kwargs)
            if cache_dir:
                logger.info(f"Saving cache for {func_name} to {cache_file}")
                safe_wpickle(cache_file, result)
                # with open(cache_file, "wb") as fd:
                #     pickle.dump(result, fd)
                logger.info(f"Saved cache!")
            return result

    return wrapper


def async_cache(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_name = func.__name__
        cache_file = os.path.join(cache_dir, f"{func_name}.pkl") if cache_dir else ""

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_file}")
            result = safe_rpickle(cache_file)
            logger.info(f"Loaded cache for {func_name} from {cache_file}")
            return result
            # with open(cache_file, "rb") as fd:
            #     return pickle.load(fd)
        else:
            result = await func(*args, **kwargs)
            if cache_dir:
                logger.info(f"Saving cache for {func_name} to {cache_file}")
                safe_wpickle(cache_file, result)
                # with open(cache_file, "wb") as fd:
                #     pickle.dump(result, fd)
                logger.info(f"Saved cache!")
            return result

    return wrapper


def index(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import faiss

        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_name = func.__name__
        cache_file = os.path.join(cache_dir, f"{func_name}.bin") if cache_dir else ""

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_file}")
            result = safe_load(cache_file, faiss.read_index)
            logger.info(f"Loaded cache for {func_name} from {cache_file}")
            return result
            # return faiss.read_index(cache_file)
        else:
            result = func(*args, **kwargs)
            if cache_dir:
                logger.info(f"Saving cache for {func_name} to {cache_file}")
                safe_save(cache_file, result, faiss.write_index)
                # faiss.write_index(result, cache_file)
                logger.info(f"Saved cache!")
            return result

    return wrapper


def np_cache(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_name = func.__name__
        cache_file = os.path.join(cache_dir, f"{func_name}.npz") if cache_dir else ""

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_file}")
            res = safe_load(cache_file, np.load)
            logger.info(f"Loaded cache for {func_name} from {cache_file}")
            # res = np.load(cache_file)
        else:
            res = None
        res = func(*args, **kwargs, cache=res, cache_file=cache_file)
        if cache_dir:
            logger.info(f"Saving cache for {func_name} to {cache_file}")
            safe_save_kw(
                cache_file, np.savez, keys=list(res.keys()), values=list(res.values())
            )
            # np.savez(cache_file, keys=list(res.keys()), values=list(res.values()))
            logger.info(f"Saved cache!")
        return res

    return wrapper


def np_cache_lazy(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_name = func.__name__
        cache_file = os.path.join(cache_dir, f"{func_name}.npz") if cache_dir else ""

        # Initialize features if not provided
        res = {}
        keys = kwargs.get("keys", {})
        keys = set(keys)

        # Load cache (partially or fully based on keys)
        if cache_dir and os.path.isfile(cache_file):
            # Use mmap for lazy loading
            with np.load(cache_file, mmap_mode="r") as cache_data:
                cached_keys = cache_data["keys"]
                cached_values = cache_data["values"]
                # If keys are specified, load only those; otherwise, load all
                if keys:
                    logger.info(f"Loading lazy cache for {func_name} from {cache_file}")
                    # Find indices of requested keys
                    key_indices = [i for i, k in enumerate(cached_keys) if k in keys]
                    if key_indices:
                        selected_keys = [cached_keys[i] for i in key_indices]
                        selected_values = [cached_values[i] for i in key_indices]
                        res.update(dict(zip(selected_keys, selected_values)))
                else:
                    # Load all keys if none specified
                    logger.info(f"Loading cache for {func_name} from {cache_file}")
                    res.update(dict(zip(cached_keys, cached_values)))
            logger.info(f"Loaded cache for {func_name} from {cache_file}")

        res = dict(keys=res.keys(), values=res.values())

        # Call the original function with updated features
        res = func(*args, **kwargs, cache=res, cache_file=cache_file)

        # Save cache (including all keys, even those not used in this run)
        if cache_dir:
            # Merge existing cache with new results
            if os.path.isfile(cache_file):
                logger.info(f"Saving lazy cache for {func_name} to {cache_file}")
                with np.load(cache_file, mmap_mode="r") as cache_data:
                    cached_keys = cache_data["keys"]
                    cached_values = cache_data["values"]
                    prev_cache = dict(zip(cached_keys, cached_values))
                    prev_cache.update(res)
                    # Save using np.savez
                    safe_save_kw(
                        cache_file,
                        np.savez,
                        keys=list(prev_cache.keys()),
                        values=list(prev_cache.values()),
                    )
            else:
                logger.info(f"Saving cache for {func_name} to {cache_file}")
                safe_save_kw(
                    cache_file,
                    np.savez,
                    keys=list(res.keys()),
                    values=list(res.values()),
                )
            logger.info(f"Saved cache for {func_name} to {cache_file}")

        return res

    return wrapper


def np_cache_kwargs(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_name = func.__name__
        func_kwargs = []
        for k, v in sorted(kwargs.items()):
            func_kwargs.append(f"{k}-{v}")
        func_kwargs = ",".join(func_kwargs)
        cache_file = (
            os.path.join(cache_dir, f"{func_name}_{func_kwargs}.npz")
            if cache_dir and func_kwargs
            else os.path.join(cache_dir, f"{func_name}.npz") if cache_dir else ""
        )

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_file}")
            res = safe_load(cache_file, np.load)
            logger.info(f"Loaded cache for {func_name} from {cache_file}")
        else:
            res = None

        res = func(*args, **kwargs, cache=res, cache_file=cache_file)
        if cache_dir:
            logger.info(f"Saving cache for {func_name} to {cache_file}")
            safe_save_kw(
                cache_file, np.savez, keys=list(res.keys()), values=list(res.values())
            )
            logger.info(f"Saved cache!")
        return res

    return wrapper


def cache_kwargs_hash(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_name = func.__name__
        func_kwargs_hash = hashlib.md5(
            json.dumps(kwargs, sort_keys=True, default=str).encode()
        ).hexdigest()
        cache_file = (
            os.path.join(cache_dir, f"{func_name}_{func_kwargs_hash}.pkl")
            if cache_dir
            else ""
        )

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_file}")
            result = safe_rpickle(cache_file)
            logger.info(f"Loaded cache for {func_name} from {cache_file}")
            return result
        else:
            result = func(*args, **kwargs)
            if cache_dir:
                logger.info(f"Saving cache for {func_name} to {cache_file}")
                safe_wpickle(cache_file, result)
                logger.info(f"Saved cache!")
            return result

    return wrapper


def np_cache_kwargs_hash(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_name = func.__name__
        func_kwargs_hash = hashlib.md5(
            json.dumps(kwargs, sort_keys=True, default=str).encode()
        ).hexdigest()
        cache_file = (
            os.path.join(cache_dir, f"{func_name}_{func_kwargs_hash}.npz")
            if cache_dir
            else ""
        )

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_file}")
            res = safe_load(cache_file, np.load)
            logger.info(f"Loaded cache for {func_name} from {cache_file}")
        else:
            res = None

        res = func(*args, **kwargs, cache=res, cache_file=cache_file)
        if cache_dir:
            logger.info(f"Saving cache for {func_name} to {cache_file}")
            safe_save_kw(
                cache_file, np.savez, keys=list(res.keys()), values=list(res.values())
            )
            logger.info(f"Saved cache!")
        return res

    return wrapper


def np_cache_kwargs_list_hash(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_name = func.__name__

        if "cache_kwargs" not in kwargs or not kwargs["cache_kwargs"]:
            logger.warning(
                f"Skipped loaing cache becaise kwargs was empty for {func_name}"
            )
            return func(*args, **kwargs, cache=None, cache_file=None)

        cache_dict = {}
        for c in kwargs["cache_kwargs"]:
            cache_dict[c] = kwargs[c]
        func_kwargs_hash = hashlib.md5(
            json.dumps(cache_dict, sort_keys=True, default=str).encode()
        ).hexdigest()
        cache_file = (
            os.path.join(cache_dir, f"{func_name}_{func_kwargs_hash}.npz")
            if cache_dir
            else ""
        )

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_file}")
            res = safe_load(cache_file, np.load)
            logger.info(f"Loaded cache for {func_name} from {cache_file}")
        else:
            res = None

        res = func(*args, **kwargs, cache=res, cache_file=cache_file)
        if cache_dir:
            logger.info(f"Saving cache for {func_name} to {cache_file}")
            safe_save_kw(
                cache_file, np.savez, keys=list(res.keys()), values=list(res.values())
            )
            logger.info(f"Saved cache!")
        return res

    return wrapper


def np_cache_dict(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_name = func.__name__
        cache_file = os.path.join(cache_dir, f"{func_name}.npz") if cache_dir else ""

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_file}")
            cache = safe_load(cache_file, np.load)
            logger.info(f"Loaded cache for {func_name} from {cache_file}")
            # cache = np.load(cache_file)
        else:
            cache = None
        res = func(*args, **kwargs, cache=cache, cache_file=cache_file)
        cache = res["cache"]
        if cache_dir:
            logger.info(f"Saving cache for {func_name} to {cache_file}")
            safe_save_kw(
                cache_file,
                np.savez,
                keys=list(cache.keys()),
                values=list(cache.values()),
            )
            # np.savez(cache_file, keys=list(cache.keys()), values=list(cache.values()))
            logger.info(f"Saved cache!")
        return res

    return wrapper


def unsafe_np_cache(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_name = func.__name__
        cache_file = os.path.join(cache_dir, f"{func_name}.npz") if cache_dir else ""

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_file}")
            res = safe_load(cache_file, np.load, allow_pickle=True)
            logger.info(f"Loaded cache for {func_name} from {cache_file}")
            # res = np.load(cache_file, allow_pickle=True)
        else:
            res = None
        res = func(*args, **kwargs, cache=res, cache_file=cache_file)
        if cache_dir:
            logger.info(f"Saving cache for {func_name} to {cache_file}")
            safe_save_kw(
                cache_file, np.savez, keys=list(res.keys()), values=list(res.values())
            )
            # np.savez(cache_file, keys=list(res.keys()), values=list(res.values()))
            logger.info(f"Saved cache!")
        return res

    return wrapper


def safe_cache(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_name = func.__name__
        cache_file = (
            os.path.join(cache_dir, f"{func_name}.safetensors") if cache_dir else ""
        )

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_file}")
            res = safe_load(cache_file, load_file)
            logger.info(f"Loaded cache for {func_name} from {cache_file}")
            # res = load_file(cache_file)
        else:
            res = None
        res = func(*args, **kwargs, cache=res, cache_file=cache_file)
        if cache_dir:
            logger.info(f"Saving cache for {func_name} to {cache_file}")
            safe_save(cache_file, res, save_file)
            # save_file(res, cache_file)
            logger.info(f"Saved cache!")
        return res

    return wrapper


def kv_cache(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_name = func.__name__
        cache_file = os.path.join(cache_dir, f"{func_name}.pkl") if cache_dir else ""

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_file}")
            res = safe_rpickle(cache_file)
            logger.info(f"Loaded cache for {func_name} from {cache_file}")
            # with open(cache_file, "rb") as fd:
            #     res = pickle.load(fd)
        else:
            res = None
        res = func(*args, **kwargs, cache=res, cache_file=cache_file)
        if cache_dir:
            logger.info(f"Saving cache for {func_name} to {cache_file}")
            safe_wpickle(cache_file, res)
            # with open(cache_file, "wb") as fd:
            #     pickle.dump(res, fd)
            logger.info(f"Saved cache!")
        return res

    return wrapper


def coalesce(*args):
    # Return the first non-None value
    return next((x for x in args if x is not None))


class Error400(Exception):
    pass


class ValidationError(Exception):
    pass


def retry(
    max_tries=None,
    wait=None,
    validator: Callable | None = None,
    return_exception=True,
    retry_on400=False,
):
    """Args max_tries and wait defined as class attributes have higher precedence."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = BFG["logger"]

            self = args[0] if len(args) else None
            tries = coalesce(max_tries, getattr(self, "max_tries", None), 0)
            delay = coalesce(wait, getattr(self, "wait", None), 0)
            response = {}
            for retry in range(-1, tries):
                try:
                    response = func(*args, **kwargs)
                    if validator:
                        validator(*args, **kwargs, response=response)
                    break
                except Error400 as e:
                    if return_exception:
                        response = {"status": "error", "error": {"message": str(e)}}
                    logger.error(f"400 Exception occured. Error: {e}")
                    logger.error(f"Args: {args}")
                    logger.error(f"KWArgs: {kwargs}")
                    if retry_on400 == False:
                        break
                except ValidationError as e:
                    if return_exception:
                        response = {"status": "error", "error": {"message": str(e)}}
                    if retry + 1 < tries:
                        import traceback

                        logger.warning(
                            f"Validation error occured, retrying {retry+1}/{tries}. Error: {e}"
                        )
                        logger.warning(traceback.format_exc())
                        logger.warning(f"Args: {args}")
                        logger.warning(f"KWArgs: {kwargs}")
                        sleep(delay)
                except Exception as e:
                    if return_exception:
                        response = {"status": "error", "error": {"message": str(e)}}
                    if retry + 1 < tries:
                        import traceback

                        logger.warning(
                            f"Exception occured, retrying {retry+1}/{tries}. Error: {e}"
                        )
                        logger.warning(traceback.format_exc())
                        logger.warning(f"Args: {args}")
                        logger.warning(f"KWArgs: {kwargs}")
                        sleep(delay)

            return response

        return wrapper

    return decorator


def last_offset(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_kwargs = []
        for k, v in sorted(kwargs.items()):
            if k.endswith("_cached"):
                func_kwargs.append(f"{k}-{v}")
        func_kwargs = ",".join(func_kwargs)

        if func_kwargs:
            cache_func_name = func.__name__ + "_" + func_kwargs
        else:
            cache_func_name = func.__name__
        cache_file = (
            os.path.join(cache_dir, f"{cache_func_name}.pkl") if cache_dir else ""
        )
        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {cache_func_name} from {cache_file}")
            output = safe_rpickle(cache_file)
            logger.info(f"Loaded cache for {cache_func_name} from {cache_file}")
            # with open(cache_file, "rb") as fd:
            #     output = pickle.load(fd)
        else:
            output = None

        if func_kwargs:
            offset_func_name = (
                func.__name__ + "_" + func_kwargs + "_" + last_offset.__name__
            )
        else:
            offset_func_name = func.__name__ + "_" + last_offset.__name__
        offset_cache_file = (
            os.path.join(cache_dir, f"{offset_func_name}.txt") if cache_dir else ""
        )

        if cache_dir and os.path.isfile(offset_cache_file):
            logger.info(f"Loading offset for {offset_func_name} from {cache_file}")
            offset = safe_ropen_fd(offset_cache_file, "readline", "r").strip()
            try:
                offset = int(offset)
                logger.info(f"Loaded offset for {offset_func_name} from {cache_file}")
            except ValueError:
                offset = 0
                logger.info(
                    f"Failed to load offset for {offset_func_name} from {cache_dir} with non integer value {offset}"
                )
            # with open(offset_cache_file, "r") as fd:
            #     offset = fd.readline().strip()
            #     try:
            #         offset = int(offset)
            #     except ValueError:
            #         offset = 0
        else:
            offset = 0

        # Un-normalize offset
        try:
            limit = kwargs["limit"]
        except KeyError:
            import inspect

            sig = inspect.signature(func)
            limit = sig.parameters["limit"].default
        offset = offset // limit

        if output:
            output, offset = func(*args, **kwargs, offset=offset, cached_output=output)
        else:
            output, offset = func(*args, **kwargs, offset=offset)

        # Normalize offset
        offset = offset * limit

        if cache_dir:
            logger.info(f"Saving offset for {offset_func_name} to {cache_file}")
            safe_wopen_fd(offset_cache_file, str(offset), "w")
            # with open(offset_cache_file, "w") as fd:
            #     fd.write(str(offset))
            logger.info(f"Saved offset!")

            logger.info(f"Saving cache for {cache_func_name} to {cache_file}")
            safe_wpickle(cache_file, output)
            # with open(cache_file, "wb") as fd:
            #     pickle.dump(output, fd)
            logger.info(f"Saved cache!")

        return output, offset

    return wrapper


def last_str_offset(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_kwargs = []
        for k, v in sorted(kwargs.items()):
            if k.endswith("_cached"):
                func_kwargs.append(f"{k}-{v}")
        func_kwargs = ",".join(func_kwargs)

        cache_func_name = func.__name__
        cache_file = (
            os.path.join(cache_dir, f"{cache_func_name}.pkl") if cache_dir else ""
        )
        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {cache_func_name} from {cache_file}")
            output = safe_rpickle(cache_file)
            logger.info(f"Loaded cache for {cache_func_name} from {cache_file}")
            # with open(cache_file, "rb") as fd:
            #     output = pickle.load(fd)
        else:
            output = None

        if func_kwargs:
            offset_func_name = (
                func.__name__ + "_" + func_kwargs + "_" + last_offset.__name__
            )
        else:
            offset_func_name = func.__name__ + "_" + last_offset.__name__
        offset_cache_file = (
            os.path.join(cache_dir, f"{offset_func_name}.txt") if cache_dir else ""
        )

        if cache_dir and os.path.isfile(offset_cache_file):
            logger.info(f"Loading offset for {offset_func_name} from {cache_file}")
            offset = str(safe_ropen_fd(offset_cache_file, "readline", "r").strip())
            # with open(offset_cache_file, "r") as fd:
            #     offset = str(fd.readline().strip())
            logger.info(f"Loaded offset!")
        else:
            offset = None

        if output:
            output, offset = func(*args, **kwargs, offset=offset, cached_output=output)
        else:
            output, offset = func(*args, **kwargs, offset=offset)

        if cache_dir and offset is not None:
            logger.info(f"Saving offset for {offset_func_name} to {cache_file}")
            safe_wopen_fd(offset_cache_file, str(offset), "w")
            # with open(offset_cache_file, "w") as fd:
            #     fd.write(str(offset))
            logger.info(f"Saved offset!")

        if cache_dir:
            logger.info(f"Saving cache for {cache_func_name} to {cache_file}")
            safe_wpickle(cache_file, output)
            # with open(cache_file, "wb") as fd:
            #     pickle.dump(output, fd)
            logger.info(f"Saved cache!")

        return output, offset

    return wrapper
