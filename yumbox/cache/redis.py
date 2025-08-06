import functools
import gzip
import hashlib
import io
import json
import os
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import numpy as np
import redis
from redis.exceptions import ConnectionError, TimeoutError

from yumbox.config import BFG

P = ParamSpec("P")


class VectorRedis:
    def __init__(
        self,
        db_name: str,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        socket_timeout: int = 30,
    ) -> None:
        """
        Initialize Redis client for numpy arrays with compression.

        :param db_name: Prefix for keys in Redis
        :param host: Redis host
        :param port: Redis port
        :param db: Redis database number
        :param password: Redis password if auth is enabled
        :param socket_timeout: Socket timeout in seconds
        """
        self.db_name = db_name
        self.key_prefix = f"{db_name}:"

        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
            decode_responses=False,  # We handle bytes manually for numpy arrays
        )

        # Test connection
        try:
            self.client.ping()
        except (ConnectionError, TimeoutError) as e:
            raise ConnectionError(f"Cannot connect to Redis at {host}:{port} - {e}")

    def _make_key(self, key: str) -> str:
        """Create full Redis key with prefix."""
        return f"{self.key_prefix}{key}"

    def _serialize_array(self, arr: np.ndarray) -> bytes:
        """Serialize numpy array to compressed bytes."""
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=6) as f:
            np.save(f, arr, allow_pickle=False)
        return buffer.getvalue()

    def _deserialize_array(self, data_bytes: bytes) -> np.ndarray:
        """Deserialize bytes back to numpy array."""
        buffer = io.BytesIO(data_bytes)
        with gzip.GzipFile(fileobj=buffer, mode="rb") as f:
            return np.load(f, allow_pickle=False)

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        return self.client.exists(self._make_key(key)) == 1

    def batch_exists(self, keys: list[str]) -> list[bool]:
        """Check existence of multiple keys in batch."""
        if not keys:
            return []

        redis_keys = [self._make_key(key) for key in keys]
        results = self.client.exists(*redis_keys)
        return [bool(results)] if len(keys) == 1 else [bool(r) for r in results]

    def upsert(self, key: str, array: np.ndarray, ttl: int | None = None) -> bool:
        """
        Upsert a numpy array.

        :param key: Key to store under
        :param array: Numpy array to store
        :param ttl: Time to live in seconds (optional)
        :return: True if successful
        """
        redis_key = self._make_key(key)
        serialized_data = self._serialize_array(array)

        if ttl:
            return self.client.setex(redis_key, ttl, serialized_data)
        else:
            return self.client.set(redis_key, serialized_data)

    def create(self, key: str, array: np.ndarray, ttl: int | None = None) -> bool:
        """
        Create a new entry.

        :param key: Key to store under
        :param array: Numpy array to store
        :param ttl: Time to live in seconds (optional)
        :return: True if created, raises ValueError if already exists
        """
        redis_key = self._make_key(key)
        if self.client.exists(redis_key):
            raise ValueError(f"Key {key} already exists")

        serialized_data = self._serialize_array(array)
        if ttl:
            return self.client.setex(redis_key, ttl, serialized_data)
        else:
            return self.client.set(redis_key, serialized_data)

    def update(self, key: str, array: np.ndarray) -> None:
        """Update the array for an existing key."""
        redis_key = self._make_key(key)
        if not self.client.exists(redis_key):
            raise KeyError(f"Key {key} does not exist")

        serialized_data = self._serialize_array(array)
        self.client.set(redis_key, serialized_data)

    def delete(self, key: str) -> None:
        """Delete a key and its associated array."""
        redis_key = self._make_key(key)
        if not self.client.exists(redis_key):
            raise KeyError(f"Key {key} does not exist")
        self.client.delete(redis_key)

    def get(self, key: str) -> np.ndarray:
        """Get array by key."""
        redis_key = self._make_key(key)
        data_bytes = self.client.get(redis_key)
        if data_bytes is None:
            raise KeyError(f"Key {key} does not exist")
        return self._deserialize_array(data_bytes)

    def get_info(self, key: str) -> dict[str, str | tuple[int, ...] | int]:
        """
        Get array metadata without loading the full array.
        Note: For Redis, we need to load the array to get metadata.
        """
        array = self.get(key)  # Unfortunately, we need to load it
        return {
            "dtype": str(array.dtype),
            "shape": array.shape,
            "size": array.size,
            "nbytes": array.nbytes,
        }

    def list_keys(self) -> list[str]:
        """Get all keys with this database prefix."""
        pattern = f"{self.key_prefix}*"
        keys = []
        for key in self.client.scan_iter(match=pattern):
            # Remove prefix and decode
            key_str = key.decode("utf-8")
            clean_key = key_str[len(self.key_prefix) :]
            keys.append(clean_key)
        return keys

    def get_dataset(self) -> dict[str, np.ndarray]:
        """Export entire database to a dictionary."""
        records = {}
        pattern = f"{self.key_prefix}*"

        for key in self.client.scan_iter(match=pattern):
            key_str = key.decode("utf-8")
            clean_key = key_str[len(self.key_prefix) :]
            data_bytes = self.client.get(key)
            if data_bytes:
                array = self._deserialize_array(data_bytes)
                records[clean_key] = array
        return records

    def close(self) -> None:
        """Close the Redis connection."""
        self.client.close()

    def __enter__(self) -> "VectorRedis":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()

    def bulk_upsert(self, data: dict[str, np.ndarray], ttl: int | None = None) -> int:
        """
        Bulk upsert multiple numpy arrays using Redis pipeline.

        :param data: Dictionary of {key: array} pairs to insert
        :param ttl: Time to live in seconds (optional)
        :return: Number of items successfully inserted
        """
        if not data:
            return 0

        pipe = self.client.pipeline()

        # Add all operations to pipeline
        for key, array in data.items():
            redis_key = self._make_key(key)
            serialized_data = self._serialize_array(array)
            if ttl:
                pipe.setex(redis_key, ttl, serialized_data)
            else:
                pipe.set(redis_key, serialized_data)

        # Execute pipeline
        results = pipe.execute()
        return sum(1 for result in results if result)

    def bulk_upsert_from_lists(
        self, keys: list[str], arrays: list[np.ndarray], ttl: int | None = None
    ) -> int:
        """
        Bulk upsert from separate key and array lists.

        :param keys: List of string keys
        :param arrays: List of numpy arrays (must match keys length)
        :param ttl: Time to live in seconds (optional)
        :return: Number of items successfully inserted
        """
        if len(keys) != len(arrays):
            raise ValueError("Keys and arrays lists must have same length")

        data = dict(zip(keys, arrays))
        return self.bulk_upsert(data, ttl)

    def bulk_get(self, keys: list[str]) -> dict[str, np.ndarray]:
        """
        Bulk get multiple arrays by keys using Redis pipeline.

        :param keys: List of string keys to retrieve
        :return: Dictionary of {key: array} for found keys (missing keys are omitted)
        """
        if not keys:
            return {}

        pipe = self.client.pipeline()
        redis_keys = [self._make_key(key) for key in keys]

        # Add all get operations to pipeline
        for redis_key in redis_keys:
            pipe.get(redis_key)

        # Execute pipeline
        results = pipe.execute()

        # Process results
        found_data = {}
        for key, data_bytes in zip(keys, results):
            if data_bytes is not None:
                found_data[key] = self._deserialize_array(data_bytes)

        return found_data

    def clear_database(self) -> None:
        """Clear all keys with this database prefix."""
        pattern = f"{self.key_prefix}*"
        keys = []
        for key in self.client.scan_iter(match=pattern):
            keys.append(key)

        if keys:
            self.client.delete(*keys)

    @classmethod
    def from_url(cls, db_name: str, redis_url: str) -> "VectorRedis":
        """
        Create VectorRedis instance from Redis URL.

        :param db_name: Prefix for keys in Redis
        :param redis_url: Redis URL in format redis://[password@]host:port/db
        :return: VectorRedis instance
        """
        from urllib.parse import urlparse

        parsed = urlparse(redis_url)

        # Default values
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379
        db = int(parsed.path.lstrip("/")) if parsed.path and parsed.path != "/" else 0
        password = parsed.password

        return cls(db_name=db_name, host=host, port=port, db=db, password=password)


def redis_cache(
    func: Callable[P, dict[str, np.ndarray]],
) -> Callable[P, dict[str, np.ndarray]]:
    """
    Redis-based caching decorator for functions that return dict[str, np.ndarray].

    The decorated function should accept 'cache' and 'cache_file' parameters.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> dict[str, np.ndarray]:
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]
        func_name = func.__name__

        cache: dict[str, np.ndarray] = {}
        keys = set(kwargs.get("keys", []))

        if cache_dir:
            with VectorRedis.from_url(func_name, cache_dir) as db:
                # If keys are specified, load only those
                if keys:
                    cache = db.bulk_get(list(keys))
                    logger.info(f"Loaded {len(cache)} cached items for {func_name}")
                else:
                    # Load all keys
                    cache = db.get_dataset()
                    logger.info(f"Loaded {len(cache)} cached items for {func_name}")

                # Prepare cache for function call
                cache = dict(keys=list(cache.keys()), values=list(cache.values()))

        cache = func(
            *args,
            **kwargs,
            cache=cache,
            cache_file=f"redis://{func_name}",
        )

        if cache_dir:
            with VectorRedis.from_url(func_name, cache_dir) as db:
                saved_count = db.bulk_upsert(cache)
                logger.info(f"Saved {saved_count} items to cache for {func_name}")

        return cache

    return wrapper


def redis_cache_kwargs_list_hash(
    func: Callable[P, dict[str, np.ndarray]],
) -> Callable[P, dict[str, np.ndarray]]:
    """
    Redis-based caching decorator with kwargs hashing for functions that return dict[str, np.ndarray].

    The decorated function should accept 'cache' and 'cache_file' parameters.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> dict[str, np.ndarray]:
        cache_dir = BFG["cache_dir"]
        logger = BFG["logger"]
        func_name = func.__name__

        if "cache_kwargs" not in kwargs or not kwargs["cache_kwargs"]:
            logger.warning(
                f"Skipped loading cache because cache_kwargs was empty for {func_name}"
            )
            return func(*args, **kwargs, cache=None, cache_file=None)

        cache_dict = {c: kwargs[c] for c in kwargs["cache_kwargs"]}
        func_kwargs_hash = hashlib.md5(
            json.dumps(cache_dict, sort_keys=True, default=str).encode()
        ).hexdigest()

        db_name = f"{func_name}_{func_kwargs_hash}"
        cache: dict[str, np.ndarray] = {}
        keys = set(kwargs.get("keys", []))

        if cache_dir:
            with VectorRedis.from_url(db_name, cache_dir) as db:
                # If keys are specified, load only those
                if keys:
                    cache = db.bulk_get(list(keys))
                    logger.info(f"Loaded {len(cache)} cached items for {db_name}")
                else:
                    # Load all keys
                    cache = db.get_dataset()
                    logger.info(f"Loaded {len(cache)} cached items for {db_name}")

                # Prepare cache for function call
                cache = dict(keys=list(cache.keys()), values=list(cache.values()))

        cache = func(*args, **kwargs, cache=cache, cache_file=f"redis://{db_name}")

        if cache_dir:
            with VectorRedis.from_url(db_name, cache_dir) as db:
                saved_count = db.bulk_upsert(cache)
                logger.info(f"Saved {saved_count} items to cache for {func_name}")

        return cache

    return wrapper
