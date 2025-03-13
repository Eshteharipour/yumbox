import errno
import functools
import os
import pickle
import shutil
import tempfile
import time
from collections.abc import Callable

from yumbox.config import BFG

__all__ = [
    "retry_on_lock",
    "safe_save_lambda",
    "safe_save_kw",
    "safe_save",
    "safe_wopen",
    "safe_wopen_fd",
    "safe_move",
    "safe_load_lambda",
    "safe_load",
    "safe_ropen",
    "safe_ropen_fd",
    "safe_wpickle",
    "safe_rpickle",
]


def retry_on_lock(max_attempts: int = 5, delay: float = 3.0):
    """Decorator to retry file operations when locked"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(file_path: str, *args, **kwargs):
            logger = BFG["logger"]

            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(file_path, *args, **kwargs)
                except IOError as e:
                    # Check if error is due to file being locked
                    if e.errno in (errno.EACCES, errno.EAGAIN):
                        attempts += 1
                        if attempts == max_attempts:
                            logger.error(
                                f"Failed to access {file_path} after {max_attempts} attempts"
                            )
                            raise
                        logger.warning(
                            f"File {file_path} is locked, retrying in {delay} seconds "
                            f"(attempt {attempts}/{max_attempts})"
                        )
                        time.sleep(delay)
                    else:
                        raise

        return wrapper

    return decorator


def safe_save_lambda(file_path: str, save_func: Callable):
    temp_dir = os.path.dirname(file_path)
    os.makedirs(temp_dir, exist_ok=True)
    file_ext = os.path.splitext(file_path)[1] or None

    with tempfile.NamedTemporaryFile(
        delete=False, dir=temp_dir, suffix=file_ext
    ) as tmp_file:
        save_func(tmp_file.name)

    safe_move(tmp_file.name, file_path)


def safe_save_kw(
    file_path: str,
    save_func: Callable,
    *save_func_args,
    **save_func_kwargs,
):
    temp_dir = os.path.dirname(file_path)
    os.makedirs(temp_dir, exist_ok=True)
    file_ext = os.path.splitext(file_path)[1] or None

    with tempfile.NamedTemporaryFile(
        delete=False, dir=temp_dir, suffix=file_ext
    ) as tmp_file:
        save_func(tmp_file.name, *save_func_args, **save_func_kwargs)

    safe_move(tmp_file.name, file_path)


def safe_save(
    file_path: str,
    obj: object,
    save_func: Callable,
    *save_func_args,
    **save_func_kwargs,
):
    temp_dir = os.path.dirname(file_path)
    os.makedirs(temp_dir, exist_ok=True)
    file_ext = os.path.splitext(file_path)[1] or None

    with tempfile.NamedTemporaryFile(
        delete=False, dir=temp_dir, suffix=file_ext
    ) as tmp_file:
        save_func(tmp_file.name, obj, *save_func_args, **save_func_kwargs)

    safe_move(tmp_file.name, file_path)


def safe_wopen(
    file_path: str,
    obj: object,
    save_func: Callable,
    *save_func_args,
    **save_func_kwargs,
):
    temp_dir = os.path.dirname(file_path)
    os.makedirs(temp_dir, exist_ok=True)
    file_ext = os.path.splitext(file_path)[1] or None

    with tempfile.NamedTemporaryFile(
        delete=False, dir=temp_dir, suffix=file_ext
    ) as tmp_file:
        with open(tmp_file.name, *save_func_args, **save_func_kwargs) as fd:
            save_func(obj, fd)

    safe_move(tmp_file.name, file_path)


def safe_wopen_fd(file_path: str, obj: object, *save_func_args, **save_func_kwargs):
    temp_dir = os.path.dirname(file_path)
    os.makedirs(temp_dir, exist_ok=True)
    file_ext = os.path.splitext(file_path)[1] or None

    with tempfile.NamedTemporaryFile(
        delete=False, dir=temp_dir, suffix=file_ext
    ) as tmp_file:
        with open(tmp_file.name, *save_func_args, **save_func_kwargs) as fd:
            fd.write(obj)

    safe_move(tmp_file.name, file_path)


@retry_on_lock()
def safe_move(src: str, dst: str):
    return shutil.move(src, dst)


@retry_on_lock()
def safe_load_lambda(file_path: str, load_func: Callable):
    logger = BFG["logger"]

    if not os.path.exists(file_path):
        logger.error(f"File {file_path} does not exist")
        raise FileNotFoundError(f"File {file_path} does not exist")

    if not os.access(file_path, os.R_OK):
        logger.error(f"No read permission for {file_path}")
        raise PermissionError(f"No read permission for {file_path}")

    return load_func(file_path)


@retry_on_lock()
def safe_load(file_path: str, load_func: Callable, *load_func_args, **load_func_kwargs):
    logger = BFG["logger"]

    if not os.path.exists(file_path):
        logger.error(f"File {file_path} does not exist")
        raise FileNotFoundError(f"File {file_path} does not exist")

    if not os.access(file_path, os.R_OK):
        logger.error(f"No read permission for {file_path}")
        raise PermissionError(f"No read permission for {file_path}")

    return load_func(file_path, *load_func_args, **load_func_kwargs)


@retry_on_lock()
def safe_ropen(
    file_path: str, read_func: Callable, *load_func_args, **load_func_kwargs
):
    logger = BFG["logger"]

    if not os.path.exists(file_path):
        logger.error(f"File {file_path} does not exist")
        raise FileNotFoundError(f"File {file_path} does not exist")

    if not os.access(file_path, os.R_OK):
        logger.error(f"No read permission for {file_path}")
        raise PermissionError(f"No read permission for {file_path}")

    with open(file_path, *load_func_args, **load_func_kwargs) as fd:
        return read_func(fd)


@retry_on_lock()
def safe_ropen_fd(file_path: str, read_attr: str, *load_func_args, **load_func_kwargs):
    logger = BFG["logger"]

    if not os.path.exists(file_path):
        logger.error(f"File {file_path} does not exist")
        raise FileNotFoundError(f"File {file_path} does not exist")

    if not os.access(file_path, os.R_OK):
        logger.error(f"No read permission for {file_path}")
        raise PermissionError(f"No read permission for {file_path}")

    with open(file_path, *load_func_args, **load_func_kwargs) as fd:
        return getattr(fd, read_attr)()


def safe_wpickle(path: str, obj: object):
    safe_wopen(path, obj, pickle.dump, "wb")
    # with open(path, "wb") as fd:
    #     pickle.dump(obj, fd)


@retry_on_lock()
def safe_rpickle(path: str):
    return safe_ropen(path, pickle.load, "rb")
    # with open(path, "rb") as fd:
    #     return pickle.load(fd)
