from collections.abc import Callable

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

no_op = lambda x: x


class ImgDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        path_col: str,
        hash_col: str,
        features: dict[str, np.ndarray],
        transform: Callable | None = no_op,
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
        preprocessor: Callable | None = no_op,
        tokenizer: Callable | None = no_op,
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


def split_token_ids(ids, chunk_size, overlap):
    start = 0
    while start < len(ids):
        end = min(start + chunk_size, len(ids))
        chunk = ids[start:end]
        yield chunk
        start = end - overlap if end != len(ids) else len(ids)


class TFDocumentDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str,
        id_col: str,
        features: dict[str, np.ndarray],
        max_seq_length: int,
        overlap: int,
        preprocessor: Callable | None = no_op,
        tokenizer: Callable | None = no_op,
    ):
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.overlap = overlap

        assert hasattr(self.tokenizer, "encode"), "BertTokenizerFast expected"
        assert hasattr(self.tokenizer, "decode"), "BertTokenizerFast expected"

        id2text = dict(zip(df[id_col], df[text_col]))
        id2text = {k: v for k, v in id2text.items() if k and pd.notna(k)}

        missing_keys = set(id2text.keys()).difference(set(features.keys()))
        self.data = [(k, id2text[k]) for k in missing_keys]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        idx, text = self.data[index]
        prep = self.preprocessor(text)
        tok = self.tokenizer.encode(prep, truncation=False)
        if len(tok) > self.max_seq_length + self.overlap:
            token_chunks = split_token_ids(
                tok, chunk_size=self.max_seq_length, overlap=self.overlap
            )
            text_chunks = []
            for i, chunk in enumerate(token_chunks):
                chunk_text = self.tokenizer.decode(
                    chunk,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                text_chunks.append(chunk_text)
        else:
            text_chunks = [prep]

        return idx, text_chunks


class ZeroshotDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str,
        id_col: str,
        features: dict[str, np.ndarray],
        templates: list[str],
        preprocessor: Callable | None = no_op,
        tokenizer: Callable | None = no_op,
    ):
        self.templates = templates
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer

        id2text = dict(zip(df[id_col], df[text_col]))
        id2text = {k: v for k, v in id2text.items() if k and pd.notna(k)}

        missing_keys = set(id2text.keys()).difference(set(features.keys()))
        data = [(k, id2text[k]) for k in missing_keys]

        self.data = [d + (t,) for d in data for t in templates]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        idx, cls, temp = self.data[index]
        prompt = self.tokenizer(temp.format(self.preprocessor(cls)))
        prompt = prompt.squeeze()
        return idx, prompt
