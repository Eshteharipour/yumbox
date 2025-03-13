import functools
import os
from collections.abc import Callable, Iterable
from time import sleep
from typing import Optional

import numpy as np
from safetensors.numpy import load_file, save_file

from yumbox.config import BFG

from .fs import *
from .kv import LMDB_API

# TODO: fix safe_save_kw


def cache(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_name = func.__name__
        cache_file = ""
        if cache_dir:
            cache_file = os.path.join(cache_dir, func_name)
            cache_file += ".pkl"

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_dir}")
            return safe_rpickle(cache_file)
            # with open(cache_file, "rb") as fd:
            #     return pickle.load(fd)
        else:
            result = func(*args, **kwargs)
            if cache_dir:
                logger.info(f"Saving cache for {func_name} to {cache_dir}")
                safe_wpickle(cache_file, result)
                # with open(cache_file, "wb") as fd:
                #     pickle.dump(result, fd)
                logger.info(f"Saved cache!")
            return result

    return wrapper


def cache_kwargs(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_name = func.__name__
        func_kwargs = []
        for k, v in kwargs.items():
            func_kwargs.append(f"{k}-{v}")
        func_kwargs = " ".join(func_kwargs)
        cache_file = ""
        if cache_dir:
            cache_file = os.path.join(cache_dir, func_name + "_" + func_kwargs)
            cache_file += ".pkl"

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_dir}")
            return safe_rpickle(cache_file)
            # with open(cache_file, "rb") as fd:
            #     return pickle.load(fd)
        else:
            result = func(*args, **kwargs)
            if cache_dir:
                logger.info(f"Saving cache for {func_name} to {cache_dir}")
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
        cache_file = ""
        if cache_dir:
            cache_file = os.path.join(cache_dir, func_name)
            cache_file += ".pkl"

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_dir}")
            return safe_rpickle(cache_file)
            # with open(cache_file, "rb") as fd:
            #     return pickle.load(fd)
        else:
            result = await func(*args, **kwargs)
            if cache_dir:
                logger.info(f"Saving cache for {func_name} to {cache_dir}")
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
        cache_file = ""
        if cache_dir:
            cache_file = os.path.join(cache_dir, func_name)
            cache_file += ".bin"

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_dir}")
            return safe_load(cache_file, faiss.read_index)
            # return faiss.read_index(cache_file)
        else:
            result = func(*args, **kwargs)
            if cache_dir:
                logger.info(f"Saving cache for {func_name} to {cache_dir}")
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
        cache_file = ""
        if cache_dir:
            cache_file = os.path.join(cache_dir, func_name)
            cache_file += ".npz"

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_dir}")
            res = safe_load(cache_file, np.load)
            # res = np.load(cache_file)
        else:
            res = None
        res = func(*args, **kwargs, cache=res, cache_file=cache_file)
        if cache_dir:
            logger.info(f"Saving cache for {func_name} to {cache_dir}")
            safe_save_kw(
                cache_file, np.savez, keys=list(res.keys()), values=list(res.values())
            )
            # np.savez(cache_file, keys=list(res.keys()), values=list(res.values()))
            logger.info(f"Saved cache!")
        return res

    return wrapper


def unsafe_np_cache(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_name = func.__name__
        cache_file = ""
        if cache_dir:
            cache_file = os.path.join(cache_dir, func_name)
            cache_file += ".npz"

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_dir}")
            res = safe_load(cache_file, np.load, allow_pickle=True)
            # res = np.load(cache_file, allow_pickle=True)
        else:
            res = None
        res = func(*args, **kwargs, cache=res, cache_file=cache_file)
        if cache_dir:
            logger.info(f"Saving cache for {func_name} to {cache_dir}")
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
        cache_file = ""
        if cache_dir:
            cache_file = os.path.join(cache_dir, func_name)
            cache_file += ".safetensors"

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_dir}")
            res = safe_load(cache_file, load_file)
            # res = load_file(cache_file)
        else:
            res = None
        res = func(*args, **kwargs, cache=res, cache_file=cache_file)
        if cache_dir:
            logger.info(f"Saving cache for {func_name} to {cache_dir}")
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
        cache_file = ""
        if cache_dir:
            cache_file = os.path.join(cache_dir, func_name)
            cache_file += ".pkl"

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_dir}")
            res = safe_rpickle(cache_file)
            # with open(cache_file, "rb") as fd:
            #     res = pickle.load(fd)
        else:
            res = None
        res = func(*args, **kwargs, cache=res, cache_file=cache_file)
        if cache_dir:
            logger.info(f"Saving cache for {func_name} to {cache_dir}")
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


def retry(max_tries=None, wait=None, validator: Optional[Callable] = None):
    """Args max_tries and wait defined as class attributes have higher precedence."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = BFG["logger"]

            self = args[0] if len(args) else None
            tries = coalesce(max_tries, getattr(self, "max_tries", None), 5)
            delay = coalesce(wait, getattr(self, "wait", None), 3)
            if tries == 0:
                tries = 1
            exception = None
            for retry in range(0, tries):
                response = {"status": "error", "error": {"message": str(exception)}}
                try:
                    response = func(*args, **kwargs)
                    if validator:
                        validator(*args, **kwargs, response=response)
                    break
                except Error400 as e:
                    exception = e
                    logger.error(f"Exception {e} occured.")
                    logger.error(f"Args: {args}")
                    logger.error(f"KWArgs: {kwargs}")
                    break
                except Exception as e:
                    exception = e
                    if retry + 1 < tries:
                        import traceback

                        logger.warning(
                            f"Exception {e} occured, retrying {retry+1}/{tries}"
                        )
                        logger.warning(traceback.format_exc())
                        logger.warning(f"Args: {args}")
                        logger.warning(f"KWArgs: {kwargs}")
                        sleep(delay)
                    continue

            return response

        return wrapper

    return decorator


def last_offset(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        offset_cache_dir = cache_dir
        logger = BFG["logger"]

        func_kwargs = []
        for k, v in kwargs.items():
            if k.endswith("_cached"):
                func_kwargs.append(f"{k}-{v}")
        func_kwargs = " ".join(func_kwargs)

        cache_func_name = func.__name__
        cache_file = ""
        if cache_dir:
            cache_file = os.path.join(cache_dir, cache_func_name)
            cache_file += ".pkl"
        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {cache_func_name} from {cache_dir}")
            output = safe_rpickle(cache_file)
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
        offset_cache_file = ""
        if offset_cache_dir:
            offset_cache_file = os.path.join(offset_cache_dir, offset_func_name)
            offset_cache_file += ".txt"

        if offset_cache_dir and os.path.isfile(offset_cache_file):
            logger.info(
                f"Loading offset for {offset_func_name} from {offset_cache_dir}"
            )
            offset = safe_ropen_fd(offset_cache_file, "readline", "r").strip()
            try:
                offset = int(offset)
            except ValueError:
                offset = 0
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
            output = func(*args, **kwargs, offset=offset, cached_output=output)
        else:
            output = func(*args, **kwargs, offset=offset)

        # Normalize offset
        offset = offset * limit

        if offset_cache_dir:
            logger.info(f"Saving offset for {offset_func_name} to {offset_cache_dir}")
            safe_wopen_fd(offset_cache_file, str(offset), "w")
            # with open(offset_cache_file, "w") as fd:
            #     fd.write(str(offset))
            logger.info(f"Saved offset!")

        if cache_dir:
            logger.info(f"Saving cache for {cache_func_name} to {cache_dir}")
            safe_wpickle(cache_file, output)
            # with open(cache_file, "wb") as fd:
            #     pickle.dump(output, fd)
            logger.info(f"Saved cache!")

        return output

    return wrapper


def last_str_offset(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        offset_cache_dir = cache_dir
        logger = BFG["logger"]

        func_kwargs = []
        for k, v in kwargs.items():
            if k.endswith("_cached"):
                func_kwargs.append(f"{k}-{v}")
        func_kwargs = " ".join(func_kwargs)

        cache_func_name = func.__name__
        cache_file = ""
        if cache_dir:
            cache_file = os.path.join(cache_dir, cache_func_name)
            cache_file += ".pkl"
        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {cache_func_name} from {cache_dir}")
            output = safe_rpickle(cache_file)
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
        offset_cache_file = ""
        if offset_cache_dir:
            offset_cache_file = os.path.join(offset_cache_dir, offset_func_name)
            offset_cache_file += ".txt"

        if offset_cache_dir and os.path.isfile(offset_cache_file):
            logger.info(
                f"Loading offset for {offset_func_name} from {offset_cache_dir}"
            )
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

        if offset_cache_dir and offset is not None:
            logger.info(f"Saving offset for {offset_func_name} to {offset_cache_dir}")
            safe_wopen_fd(offset_cache_file, str(offset), "w")
            # with open(offset_cache_file, "w") as fd:
            #     fd.write(str(offset))
            logger.info(f"Saved offset!")

        if cache_dir:
            logger.info(f"Saving cache for {cache_func_name} to {cache_dir}")
            safe_wpickle(cache_file, output)
            # with open(cache_file, "wb") as fd:
            #     pickle.dump(output, fd)
            logger.info(f"Saved cache!")

        return output, offset

    return wrapper


def get_feats(keys: Iterable[str], feats: dict[str]) -> dict[str]:
    return np.array([feats[k] for k in keys])
