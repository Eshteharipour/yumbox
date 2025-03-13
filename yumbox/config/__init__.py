import io
import logging
import os
import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from datetime import datetime
from typing import Literal


class CFGClass:
    def __init__(self):
        self.cfg = {"logger": logging.getLogger(), "cache_dir": None}

    def __getitem__(self, index: Literal["logger", "cache_dir"]):
        return self.cfg[index]

    def __setitem__(
        self, index: Literal["logger", "cache_dir"], value: str | logging.Logger
    ):
        if value:
            if index == "cache_dir":
                os.makedirs(value, exist_ok=True)
            self.cfg[index] = value


BFG = CFGClass()


_streams = {"stdout": sys.stdout}


class CustomFormatter(logging.Formatter):
    red__bg_white = "\x1b[91;47m"
    blue = "\x1b[34m"
    bold_cyan = "\x1b[36;1m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )
    FORMATS = {
        logging.DEBUG: blue + format + reset,
        # logging.DEBUG: grey + format + reset,
        logging.INFO: bold_cyan + format + reset,
        # logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: red__bg_white + format + reset,
        # logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(name, level=logging.INFO, path="", stream="stdout"):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
    global _streams
    if stream not in _streams:
        log_folder = os.path.dirname(stream)
        os.makedirs(log_folder, exist_ok=True)
        _streams[stream] = open(stream, "w")
    logger.propagate = False
    logger.setLevel(level)

    sh = logging.StreamHandler(stream=_streams[stream])
    sh.setLevel(level)
    sh.setFormatter(CustomFormatter())
    logger.addHandler(sh)

    # File logging
    if path:
        os.makedirs(path, exist_ok=True)
        date_time = datetime.now().isoformat()
        logfile = os.path.join(path, f"run_{date_time}.log")
        fh = logging.FileHandler(filename=logfile)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)

    return logger


def redir_print(func: Callable, *args, **kwargs) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        func(*args, **kwargs)
    return buffer.getvalue()
