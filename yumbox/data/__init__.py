from typing import Callable, Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ImgDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        path_col: str,
        hash_col: str,
        features: dict[str, np.ndarray],
        transform: Optional[Callable],
    ):
        self.transform = transform

        df_wimages = df[df[path_col].astype(bool)]
        hash2path = {}
        for i, r in df_wimages.iterrows():
            hash2path[r[hash_col]] = r[path_col]

        missing_keys = set(hash2path.keys()).difference(set(features.keys()))
        hash2path = {k: v for k, v in hash2path.items() if k in missing_keys}
        self.bn2path = hash2path
        self.data = list(hash2path.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        hash = self.data[index]
        path = self.bn2path[hash]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return hash, img


class TextDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        name_col: str,
        features: dict[str, np.ndarray],
        preprocessor: Optional[Callable],
        tokenizer: Optional[Callable],
    ):
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer

        df_wnames = df[df[name_col].astype(bool)]
        df_wnames = df_wnames[name_col]

        missing_keys = set(df_wnames.values).difference(set(features.keys()))
        self.data = list(missing_keys)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        name = self.data[index]
        tok = self.preprocessor(name)
        tok = self.tokenizer(tok)
        if not isinstance(tok, str):
            tok = tok.squeeze()
        return name, tok
