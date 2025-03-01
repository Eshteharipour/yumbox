import io
import logging
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime
from typing import Callable, Literal


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
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File logging
    if path:
        os.makedirs(path, exist_ok=True)
        date_time = datetime.now().isoformat()
        logfile = os.path.join(path, f"run_{date_time}.log")
        fh = logging.FileHandler(filename=logfile)
        fh.setLevel(level)
        logger.addHandler(fh)

    return logger


def redir_print(func: Callable, *args, **kwargs) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        func(*args, **kwargs)
    return buffer.getvalue()
