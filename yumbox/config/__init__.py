import io
import logging
import os
import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from datetime import datetime
from typing import Literal, Union, overload


class CFGClass:
    def __init__(self):
        self.cfg: dict[
            Literal["logger", "cache_dir"], Union[logging.Logger, str, None]
        ] = {"logger": logging.getLogger(), "cache_dir": None}

    @overload
    def __getitem__(self, index: Literal["logger"]) -> logging.Logger: ...
    @overload
    def __getitem__(self, index: Literal["cache_dir"]) -> str | None: ...
    def __getitem__(
        self, index: Literal["logger", "cache_dir"]
    ) -> Union[logging.Logger, str, None]:
        return self.cfg[index]

    @overload
    def __setitem__(self, index: Literal["logger"], value: logging.Logger) -> None: ...
    @overload
    def __setitem__(
        self, index: Literal["cache_dir"], value: Union[str, None]
    ) -> None: ...
    def __setitem__(
        self,
        index: Literal["logger", "cache_dir"],
        value: Union[logging.Logger, str, None],
    ) -> None:
        if value:
            if index == "cache_dir":
                os.makedirs(value, exist_ok=True)
            self.cfg[index] = value


BFG = CFGClass()


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
    # stream arg is deprecated and kept for backwards compatibility

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
    global _streams
    logger.propagate = False
    logger.setLevel(level)

    sh = logging.StreamHandler(stream=sys.stdout)
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
