import functools
import os
import pickle
import shutil
import tempfile
from collections.abc import Callable
from time import sleep
from typing import Iterable, Optional

import numpy as np
from safetensors.numpy import load_file, save_file

from yumbox.config import BFG


def safe_save(file_path: str, save_func: Callable, *save_func_args, **save_func_kwargs):
    temp_dir = os.path.dirname(file_path)
    file_ext = os.path.splitext(file_path)[1] or None

    with tempfile.NamedTemporaryFile(
        delete=False, dir=temp_dir, suffix=file_ext
    ) as tmp_file:
        save_func(tmp_file.name, *save_func_args, **save_func_kwargs)

    shutil.move(tmp_file.name, file_path)


def safe_load():
    pass


def safe_openw(file_path: str, *save_func_args, **save_func_kwargs):
    temp_dir = os.path.dirname(file_path)
    file_ext = os.path.splitext(file_path)[1] or None

    with tempfile.NamedTemporaryFile(
        delete=False, dir=temp_dir, suffix=file_ext
    ) as tmp_file:
        with open(tmp_file.name) as fd:
            fd.write(*save_func_args, **save_func_kwargs)

    shutil.move(tmp_file.name, file_path)


def safe_openr():
    pass


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
            res = np.load(cache_file, allow_pickle=True)
        else:
            res = None
        res = func(*args, **kwargs, cache=res, cache_file=cache_file)
        if cache_dir:
            logger.info(f"Saving cache for {func_name} to {cache_dir}")
            np.savez(cache_file, keys=list(res.keys()), values=list(res.values()))
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
            res = load_file(cache_file)
        else:
            res = None
        res = func(*args, **kwargs, cache=res, cache_file=cache_file)
        if cache_dir:
            logger.info(f"Saving cache for {func_name} to {cache_dir}")
            save_file(res, cache_file)
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


def retry(max_tries=None, wait=None, validator: Optional[Callable] = None):
    """Args max_tries and wait defined as class attributes have higher precedence."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = BFG["logger"]

            self = args[0] if len(args) else None
            tries = coalesce(max_tries, getattr(self, "max_tries", None), 5)
            delay = coalesce(wait, getattr(self, "wait", None), 3)
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
            with open(cache_file, "rb") as fd:
                output = pickle.load(fd)
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
            with open(offset_cache_file, "r") as fd:
                offset = fd.readline().strip()
                try:
                    offset = int(offset)
                except ValueError:
                    offset = 0
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
            with open(offset_cache_file, "w") as fd:
                fd.write(str(offset))
            logger.info(f"Saved offset!")

        if cache_dir:
            logger.info(f"Saving cache for {cache_func_name} to {cache_dir}")
            with open(cache_file, "wb") as fd:
                pickle.dump(output, fd)
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
            with open(cache_file, "rb") as fd:
                output = pickle.load(fd)
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
            with open(offset_cache_file, "r") as fd:
                offset = str(fd.readline().strip())
        else:
            offset = None

        if output:
            output, offset = func(*args, **kwargs, offset=offset, cached_output=output)
        else:
            output, offset = func(*args, **kwargs, offset=offset)

        if offset_cache_dir and offset is not None:
            logger.info(f"Saving offset for {offset_func_name} to {offset_cache_dir}")
            with open(offset_cache_file, "w") as fd:
                fd.write(str(offset))
            logger.info(f"Saved offset!")

        if cache_dir:
            logger.info(f"Saving cache for {cache_func_name} to {cache_dir}")
            with open(cache_file, "wb") as fd:
                pickle.dump(output, fd)
            logger.info(f"Saved cache!")

        return output, offset

    return wrapper


def get_feats(
    keys: Iterable[str], feats: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return np.array([feats[k] for k in keys])
