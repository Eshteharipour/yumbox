import functools
import io
import logging
import os
import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from datetime import datetime
from typing import Literal, Union, overload


class Print:
    @functools.wraps(logging.info)
    def info(self, *args, **kwargs):
        print(*args, **kwargs)

    @functools.wraps(logging.warn)
    def warn(self, *args, **kwargs):
        print(*args, **kwargs)

    @functools.wraps(logging.warning)
    def warning(self, *args, **kwargs):
        print(*args, **kwargs)

    @functools.wraps(logging.error)
    def error(self, *args, **kwargs):
        print(*args, **kwargs)

    @functools.wraps(logging.critical)
    def critical(self, *args, **kwargs):
        print(*args, **kwargs)

    @functools.wraps(logging.exception)
    def exception(self, *args, **kwargs):
        print(*args, **kwargs)

    @functools.wraps(logging.fatal)
    def fatal(self, *args, **kwargs):
        print(*args, **kwargs)

    @functools.wraps(logging.debug)
    def debug(self, *args, **kwargs):
        print(*args, **kwargs)

    @functools.wraps(logging.log)
    def log(self, *args, **kwargs):
        print(*args, **kwargs)

    @functools.wraps(logging.Logger._log)
    def _log(self, *args, **kwargs):
        print(*args, **kwargs)


class CFGClass:
    def __init__(self):
        self.cfg: dict[
            Literal["logger", "cache_dir"], Union[logging.Logger, str, None]
        ] = {"logger": Print(), "cache_dir": None}

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


def setup_logger(
    name,
    level=logging.INFO,
    path="",
    stream="stdout",
    file_level=logging.INFO,
    redirect_prints=False,
    capture_libs: list[str] | None = None,
    capture_all_libs=False,
    suppress_libs: list[str] | None = None,
    suppress_libs_std: list[str] | None = None,
):
    """
    Set up a logger with stream and optional file output, optionally redirecting all prints and library logs.
    Args:
        name: Logger name.
        level: Logging level for stream (console) output (e.g., 'INFO', logging.INFO).
        path: Directory path for log file (if empty, no file logging).
        stream: Stream for console output (default: 'stdout', can be sys.stdout or sys.stderr).
        file_level: Logging level for file output (e.g., 'DEBUG', logging.DEBUG).
        redirect_prints: If True, redirect all print statements to the log file (requires path).
        capture_all_libs: If True, configure root logger to capture logs from all libraries.
        capture_libs: List of specific library names whose logs should be captured (e.g., ['requests', 'urllib3']).
        suppress_libs: List of library names whose logs should be suppressed (e.g., ['rustboard_core']).
        suppress_libs_std: List of text patterns to suppress in stdout/stderr (e.g., ['debug info', 'internal:']).
    Returns:
        Configured logger.
    """
    # stream arg is deprecated and kept for backwards compatibility
    # Set default values for parameters that might be None
    if capture_libs is None:
        capture_libs = []
    if suppress_libs is None:
        suppress_libs = []
    if suppress_libs_std is None:
        suppress_libs_std = []
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    if isinstance(file_level, str):
        file_level = logging.getLevelName(file_level.upper())

    # First, handle suppression of specified libraries
    for lib_name in suppress_libs:
        lib_logger = logging.getLogger(lib_name)
        lib_logger.handlers.clear()  # Remove existing handlers
        lib_logger.setLevel(
            logging.CRITICAL + 10
        )  # Set to a level higher than CRITICAL
        lib_logger.propagate = False  # Prevent propagation to parent loggers

        # For more aggressive suppression, add a null handler
        lib_logger.addHandler(logging.NullHandler())

    # Set up the named logger
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        # Libraries like Kornia add handlers, clear them:
        logger.handlers.clear()
    logger.propagate = False

    # Set logger level to capture all relevant messages
    min_level = min(level, file_level)
    logger.setLevel(min_level)

    # Stream handler (console output)
    stream = sys.stdout if stream == "stdout" else stream
    sh = logging.StreamHandler(stream=stream)
    sh.setLevel(level)
    sh.setFormatter(CustomFormatter())
    logger.addHandler(sh)

    # File logging
    logfile = None
    if path:
        os.makedirs(path, exist_ok=True)
        date_time = datetime.now().isoformat()
        logfile = os.path.join(path, f"run_{date_time}.log")

        # File handler for logs
        fh = logging.FileHandler(filename=logfile)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        fh.setLevel(file_level)
        logger.addHandler(fh)

    # Redirect prints if requested
    if redirect_prints and path:
        # Create custom stdout and stderr that write to both terminal and file
        class TeeOutput:
            def __init__(self, original_stream, log_file_path, suppress_patterns=None):
                self.original_stream = original_stream
                self.log_file = open(log_file_path, "a")  # Append mode
                self.suppress_patterns = suppress_patterns or []

            def write(self, message):
                # Check if the message contains any of the suppression patterns
                should_suppress = any(
                    pattern in message for pattern in self.suppress_patterns
                )

                if not should_suppress:
                    self.original_stream.write(message)
                    self.log_file.write(message)
                    self.log_file.flush()  # Ensure immediate writing to the file

            def flush(self):
                self.original_stream.flush()
                self.log_file.flush()

            def isatty(self):
                return self.original_stream.isatty()

            def close(self):
                self.log_file.close()

        # Save references to original stdout/stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Replace with our tee version that writes to both terminal and file
        sys.stdout = TeeOutput(
            original_stdout, logfile, suppress_patterns=suppress_libs_std
        )
        sys.stderr = TeeOutput(
            original_stderr, logfile, suppress_patterns=suppress_libs_std
        )

    # Capture logs from specific libraries if requested
    if capture_libs and path:
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        for lib_name in capture_libs:
            # Skip if this library is in the suppression list
            if lib_name in suppress_libs:
                continue

            lib_logger = logging.getLogger(lib_name)
            lib_logger.propagate = False  # Prevent double logging
            lib_logger.setLevel(min_level)

            # Add stream handler to library logger
            lib_sh = logging.StreamHandler(stream=stream)
            lib_sh.setLevel(level)
            lib_sh.setFormatter(CustomFormatter())
            lib_logger.addHandler(lib_sh)

            # Add file handler to library logger
            lib_fh = logging.FileHandler(filename=logfile)
            lib_fh.setFormatter(formatter)
            lib_fh.setLevel(file_level)
            lib_logger.addHandler(lib_fh)

            logger.info(f"Capturing logs from library: {lib_name}")

    # Capture logs from all libraries if requested
    if capture_all_libs:
        root_logger = logging.getLogger()  # Root logger
        if root_logger.handlers:  # Clear existing handlers
            root_logger.handlers.clear()

        root_logger.setLevel(min_level)

        # Add stream handler to root logger
        root_sh = logging.StreamHandler(stream=stream)
        root_sh.setLevel(level)
        root_sh.setFormatter(CustomFormatter())
        root_logger.addHandler(root_sh)

        # Add file handler to root logger if path is provided
        if path:
            root_fh = logging.FileHandler(filename=logfile)
            root_fh.setFormatter(formatter)
            root_fh.setLevel(file_level)
            root_logger.addHandler(root_fh)

        # Re-apply suppression for libraries even in capture_all mode
        for lib_name in suppress_libs:
            lib_logger = logging.getLogger(lib_name)
            lib_logger.handlers.clear()
            lib_logger.setLevel(logging.CRITICAL + 10)
            lib_logger.propagate = False
            lib_logger.addHandler(logging.NullHandler())

    return logger


def redir_print(func: Callable, *args, **kwargs) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        func(*args, **kwargs)
    return buffer.getvalue()


def execution_wrapper(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        import traceback
        from time import time

        logger = BFG["logger"]

        begin = time()
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.error(traceback.format_exc())
        finally:
            elapsed = time() - begin
            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)
            logger.info(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    return wrapped


def main_run(main: Callable, *args, **kwargs):
    import traceback
    from time import time

    logger = BFG["logger"]
    begin = time()
    try:
        main(*args, **kwargs)
    except Exception as e:
        logger.error(traceback.format_exc())
    finally:
        elapsed = time() - begin
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        logger.info(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
