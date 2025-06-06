import errno
import functools
import mimetypes
import os
import pickle
import random
import shutil
import tempfile
import time
from collections.abc import Callable

import numpy as np
from PIL import Image

from yumbox.config import BFG
from yumbox.nlp import replace_fromstart

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
    "FSImage",
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


class FSImage:
    def __init__(
        self,
        data_dirs: list[str],
        exts: list[str] | str | None = None,
        depth=0,
    ):
        """Initialize an FSImage object to manage and retrieve image files from specified directories.

        Args:
            data_dirs (list[str]): A list of directory paths to search for image files.
            exts (list[str] | str | None, optional): File extensions or MIME type prefix to filter images by.
                Can be a single string (e.g. 'image' for all image MIME types), a list of strings
                (e.g., ['.jpg', '.png']), or None to include all files. Defaults to None.
            depth (int, optional): Maximum directory depth to search for images. A depth of 0 means no limit,
                1 means only the top-level directory, 2 includes one level of subdirectories, etc. Defaults to 0.
        """

        mimetypes.init()
        exts = [
            ext
            for ext, mime in mimetypes.types_map.items()
            if mime.startswith(f"{exts}/")
        ]
        if exts is None:
            exts = ["." + e if not e.startswith(".") else e for e in exts]

        self.exts = exts
        self.data_dirs = data_dirs
        self.depth = depth

        self.images = []
        for path in self.data_dirs:
            self.images.extend(self.build_files_list(path, self.exts, self.depth))

    def __repr__(self):
        return (
            f"FSImage(data_dirs={self.data_dirs}, exts={self.exts}, depth={self.depth})"
        )

    def build_files_list(self, path: str, exts: list[str], depth: int) -> list[str]:
        walk = os.walk(path)
        all_files = []
        for parent, dirs, files in walk:
            files = [os.path.join(parent, f) for f in files]
            all_files.extend(files)
        all_files = [
            f
            for f in all_files
            if (
                (len(replace_fromstart(f, path).split(os.path.sep)) < depth + 1)
                if depth
                else True
            )
            and ((ext := os.path.splitext(f)[1]) in exts or ext == "" if exts else True)
        ]
        return all_files

    def _get_seed(self, seed: int | None = 362):
        if seed == None:
            seed = 362
        return random.Random(seed)

    def _read_img(self, path):
        return np.array(Image.open(path).convert("RGB"))

    def get_random(self, seed=None):
        state = self._get_seed(seed)
        img = state.choice(self.images)
        return self._read_img(img)

    def get_one_image(self, seed=362):
        return self.get_random(seed)

    def get_ten_images(self, seed=362):
        images = []
        for i in range(0, 10):
            images.append(self.get_random(seed))
        return images
