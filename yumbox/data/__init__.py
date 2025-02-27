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
        text_col: str,
        id_col: str,
        features: dict[str, np.ndarray],
        preprocessor: Optional[Callable],
        tokenizer: Optional[Callable],
    ):
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer

        id2text = dict(zip(df[id_col], df[text_col]))
        id2text = {k: v for k, v in id2text.items() if k and pd.notna(k)}

        missing_keys = set(id2text.keys()).difference(set(features.keys()))
        self.data = [(k, id2text[k]) for k in missing_keys]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        idx, text = self.data[index]
        tok = self.preprocessor(text)
        tok = self.tokenizer(tok)
        if not isinstance(tok, str):
            tok = tok.squeeze()
        return idx, tok
