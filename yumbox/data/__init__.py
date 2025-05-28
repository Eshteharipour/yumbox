from collections.abc import Callable
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from PIL import Image
from torch.utils.data import Dataset

from yumbox.config import BFG

from .trainer import *

no_op = lambda x: x


class WebImgDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        path_col: str,
        hash_col: str,
        features: dict[str, np.ndarray],
        embed_dim: int,
        transform: Callable | None = no_op,
    ):
        self.transform = no_op if transform is None else transform

        df_wimages = df[df[hash_col].astype(bool) & df[hash_col].notna()]
        hash2path = {r[hash_col]: r[path_col] for _, r in df_wimages.iterrows()}
        missing_keys = set(hash2path.keys()).difference(set(features.keys()))
        self.data = [(k, hash2path[k]) for k in missing_keys]

        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.embed_dim = embed_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        key, url = self.data[index]
        try:
            response = requests.get(url, stream=False, timeout=10, headers=self.headers)
            response.raise_for_status()

            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = self.transform(img)
            return key, img
        except Exception as e:
            print(f"WARNING: download/read failed with exception: {e}")
            return key, torch.empty(0, self.embed_dim, dtype=torch.float32)


class ImgDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        path_col: str,
        hash_col: str,
        features: dict[str, np.ndarray],
        transform: Callable | None = no_op,
    ):
        self.transform = no_op if transform is None else transform

        df_wimages = df[df[hash_col].astype(bool) & df[hash_col].notna()]
        hash2path = {r[hash_col]: r[path_col] for _, r in df_wimages.iterrows()}
        missing_keys = set(hash2path.keys()).difference(set(features.keys()))
        self.data = [(k, hash2path[k]) for k in missing_keys]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        key, path = self.data[index]
        try:
            img = Image.open(path).convert("RGB")
        except OSError as e:
            BFG["logger"].error(f"Error reading corrupted image: {path}")
            BFG["logger"].error(e)
            raise
        except Exception as e:
            BFG["logger"].error(f"Error while reading image: {path}")
            BFG["logger"].error(e)
            raise
        img = self.transform(img)
        return key, img


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
        self.preprocessor = no_op if preprocessor is None else preprocessor
        self.tokenizer = no_op if tokenizer is None else tokenizer

        id2text = dict(zip(df[id_col], df[text_col]))
        missing_keys = set(id2text.keys()).difference(set(features.keys()))
        self.data = [(k, id2text[k]) for k in missing_keys]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        key, text = self.data[index]
        tok = self.preprocessor(text)
        tok = self.tokenizer(tok)
        if not isinstance(tok, str):
            tok = tok.squeeze()
        return key, tok


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
        if preprocessor is None:
            self.preprocessor = no_op
        else:
            self.preprocessor = preprocessor

        if tokenizer is None:
            self.tokenizer = no_op
        else:
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
