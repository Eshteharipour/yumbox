import logging
from typing import Literal


class CFGClass:
    def __init__(self):
        self.cfg = {"logger": logging.getLogger(), "cache_dir": None}

    def __getitem__(self, index: Literal["logger", "cache_dir"]):
        return self.cfg[index]

    def __setitem__(
        self, index: Literal["logger", "cache_dir"], value: str | logging.Logger
    ):
        self.cfg[index] = value


BFG = CFGClass()
