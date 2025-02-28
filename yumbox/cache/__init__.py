import functools
import os
import pickle
from collections.abc import Callable
from time import sleep
from typing import Optional

import faiss
import numpy as np

from yumbox.config import BFG

# TODO: To prevent corruption on power outage,
# save to temp location then move to target.


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
            with open(cache_file, "rb") as fd:
                return pickle.load(fd)
        else:
            result = func(*args, **kwargs)
            if cache_dir:
                logger.info(f"Saving cache for {func_name} to {cache_dir}")
                with open(cache_file, "wb") as fd:
                    pickle.dump(result, fd)
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
            with open(cache_file, "rb") as fd:
                return pickle.load(fd)
        else:
            result = func(*args, **kwargs)
            if cache_dir:
                logger.info(f"Saving cache for {func_name} to {cache_dir}")
                with open(cache_file, "wb") as fd:
                    pickle.dump(result, fd)
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
            with open(cache_file, "rb") as fd:
                return pickle.load(fd)
        else:
            result = await func(*args, **kwargs)
            if cache_dir:
                logger.info(f"Saving cache for {func_name} to {cache_dir}")
                with open(cache_file, "wb") as fd:
                    pickle.dump(result, fd)
                logger.info(f"Saved cache!")
            return result

    return wrapper


def index(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]

        func_name = func.__name__
        cache_file = ""
        if cache_dir:
            cache_file = os.path.join(cache_dir, func_name)
            cache_file += ".bin"

        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_dir}")
            return faiss.read_index(cache_file)
        else:
            result = func(*args, **kwargs)
            if cache_dir:
                logger.info(f"Saving cache for {func_name} to {cache_dir}")
                faiss.write_index(result, cache_file)
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
            res = np.load(cache_file)
        else:
            res = None
        res = func(*args, **kwargs, cache=res, cache_file=cache_file)
        if cache_dir:
            logger.info(f"Saving cache for {func_name} to {cache_dir}")
            np.savez(cache_file, keys=list(res.keys()), values=list(res.values()))
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
            with open(cache_file, "rb") as fd:
                res = pickle.load(fd)
        else:
            res = None
        res = func(*args, **kwargs, cache=res, cache_file=cache_file)
        if cache_dir:
            logger.info(f"Saving cache for {func_name} to {cache_dir}")
            with open(cache_file, "wb") as fd:
                pickle.dump(res, fd)
            logger.info(f"Saved cache!")
        return res

    return wrapper


def coalesce(*args):
    # Return the first non-None value
    return next((x for x in args if x is not None))


class Error400(Exception):
    pass


def retry(max_tries=5, wait=3, validator: Optional[Callable] = None):
    """Args max_tries and wait defined as class attributes have higher precedence."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = BFG["logger"]

            self = args[0] if len(args) else None
            tries = coalesce(getattr(self, "max_tries", None), max_tries)
            delay = coalesce(getattr(self, "wait", None), wait)
            success = False
            exception = None
            for retry in range(0, tries):
                try:
                    response = func(*args, **kwargs)
                    if validator:
                        validator(*args, **kwargs, response=response)
                    success = True
                    break
                except Error400 as e:
                    exception = e
                    logger.error(f"Exception {e} occured.")
                    break
                except Exception as e:
                    exception = e
                    if retry + 1 < tries:
                        import traceback

                        logger.warning(
                            f"Exception {e} occured, retrying {retry+1}/{tries}"
                        )
                        traceback.print_exc()
                        sleep(delay)
                    continue
            if success == True:
                return response
            else:
                return {"status": "error", "error": {"message": str(exception)}}

        return wrapper

    return decorator


def last_offset(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = BFG["cache_dir"]
        offset_cache_dir = cache_dir
        logger = BFG["logger"]

        func_name = func.__name__
        func_kwargs = []
        for k, v in kwargs.items():
            if k.endswith("_cached"):
                func_kwargs.append(f"{k}-{v}")
        func_kwargs = " ".join(func_kwargs)
        cache_file = ""
        if cache_dir:
            cache_file = os.path.join(cache_dir, func_name + "_" + func_kwargs)
            cache_file += ".pkl"
        if cache_dir and os.path.isfile(cache_file):
            logger.info(f"Loading cache for {func_name} from {cache_dir}")
            with open(cache_file, "rb") as fd:
                output = pickle.load(fd)
        else:
            output = None

        func_name = func_name + "_" + last_offset.__name__
        offset_cache_file = ""
        if offset_cache_dir:
            offset_cache_file = os.path.join(offset_cache_dir, func_name)
            offset_cache_file += ".txt"

        if offset_cache_dir and os.path.isfile(offset_cache_file):
            logger.info(f"Loading offset for {func_name} from {offset_cache_dir}")
            with open(offset_cache_file, "r") as fd:
                offset = fd.readline().strip()
                try:
                    offset = int(offset)
                except ValueError:
                    offset = 0
        else:
            offset = 0

        if output:
            output, offset = func(*args, **kwargs, offset=offset, cached_output=output)
        else:
            output, offset = func(*args, **kwargs, offset=offset)

        if offset_cache_dir:
            logger.info(f"Saving offset for {func_name} to {offset_cache_dir}")
            with open(offset_cache_file, "w") as fd:
                fd.write(str(offset))
            logger.info(f"Saved offset!")

        if cache_dir:
            logger.info(f"Saving cache for {func_name} to {cache_dir}")
            with open(cache_file, "wb") as fd:
                pickle.dump(output, fd)
            logger.info(f"Saved cache!")
        return output, offset

    return wrapper
