import os

import lmdb
import msgpack
from lmdb import Environment


class LMDBMultiIndex:
    def __init__(self, db_name: str, folder: str, map_size: int = 10485760):
        """
        Initialize LMDB database.
        :param db_name: Name of the database (e.g., "mydata").
        :param folder: Folder where DB files will be stored.
        :param map_size: Max size in bytes (default 10MB, adjust as needed).
        """
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.db_path = os.path.join(folder, db_name)

        self.env: Environment = lmdb.open(self.db_path, map_size=map_size, lock=False)
        self.data_prefix = b"data:"  # Namespace for actual data

    def _data_id_to_bytes(self, data_id: str) -> bytes:
        """Convert data ID to bytes."""
        return self.data_prefix + data_id.encode()

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        with self.env.begin() as txn:
            key_bytes = key.encode()
            return txn.get(key_bytes) is not None

    def batch_exists(self, keys: list[str]) -> list[bool]:
        """Check existence of multiple keys in batch."""
        result = []
        with self.env.begin() as txn:
            for key in keys:
                key_bytes = key.encode()
                result.append(txn.get(key_bytes) is not None)
        return result

    def create(self, key: str, data: dict, data_id: str | None = None) -> str:
        """
        Create a new entry. If data_id is provided, use it; otherwise, generate one.
        :return: The data_id used.
        """
        with self.env.begin(write=True) as txn:
            key_bytes = key.encode()
            if txn.get(key_bytes):
                raise ValueError(f"Key {key} already exists")

            # If no data_id provided, generate a new one
            if data_id is None:
                data_id = os.urandom(16).hex()  # 16-byte random ID
            data_id_bytes = self._data_id_to_bytes(data_id)

            # Store the data if it doesn’t exist
            if not txn.get(data_id_bytes):
                txn.put(data_id_bytes, msgpack.packb(data))

            # Map key to data_id
            txn.put(key_bytes, data_id.encode())
            return data_id

    def update(self, key: str, data: dict):
        """Update the data for an existing key (affects all keys referencing the same data)."""
        with self.env.begin(write=True) as txn:
            key_bytes = key.encode()
            data_id = txn.get(key_bytes)
            if not data_id:
                raise KeyError(f"Key {key} does not exist")

            data_id_bytes = self._data_id_to_bytes(data_id.decode())
            txn.put(data_id_bytes, msgpack.packb(data))

    def delete(self, key: str):
        """Delete a key. Data remains if referenced by other keys."""
        with self.env.begin(write=True) as txn:
            key_bytes = key.encode()
            if not txn.get(key_bytes):
                raise KeyError(f"Key {key} does not exist")
            txn.delete(key_bytes)

    def get(self, key: str) -> dict | None:
        """Get data by key."""
        with self.env.begin() as txn:
            key_bytes = key.encode()
            data_id = txn.get(key_bytes)
            if data_id is None:
                raise KeyError(f"Key {key} does not exist")
            else:
                data_id_bytes = self._data_id_to_bytes(data_id.decode())
                data_bytes = txn.get(data_id_bytes)
                if data_bytes is None:
                    raise KeyError(
                        f"Inner key {data_id.decode()} for key {key} does not exist"
                    )
                else:
                    return msgpack.unpackb(data_bytes, raw=False)
        return None

    def get_dataset(self) -> dict[str, dict]:
        """
        Export database to a dictionary, with options to handle duplicate values.
        Two approaches are provided:
        1. Group keys by data_id (returns dict of data with lists of keys)
        2. Return unique data values with one representative key
        """

        # Option 1: Group by data_id (keys sharing same data)
        def get_grouped_by_data():
            grouped_data = {}  # data_id -> (data, [keys])
            with self.env.begin() as txn:
                cursor = txn.cursor()
                # First pass: collect all key -> data_id mappings
                key_to_data_id = {}
                for key_bytes, value_bytes in cursor:
                    if key_bytes.startswith(self.data_prefix):
                        continue  # Skip data entries
                    key = key_bytes.decode()
                    data_id = value_bytes.decode()
                    key_to_data_id[key] = data_id

                    # Initialize the group for this data_id if it doesn't exist
                    if data_id not in grouped_data:
                        data_id_bytes = self._data_id_to_bytes(data_id)
                        data_bytes = txn.get(data_id_bytes)
                        if data_bytes:
                            data = msgpack.unpackb(data_bytes, raw=False)
                            grouped_data[data_id] = (data, [])

                # Second pass: group keys by data_id
                for key, data_id in key_to_data_id.items():
                    if data_id in grouped_data:
                        grouped_data[data_id][1].append(key)

            # Transform to more usable format
            result = {}
            for data_id, (data, keys) in grouped_data.items():
                result[data_id] = {"data": data, "keys": keys}
            return result

        # Option 2: Return unique data with one representative key
        def get_unique_data():
            records = {}
            seen_data_ids = set()
            with self.env.begin() as txn:
                cursor = txn.cursor()
                for key_bytes, value_bytes in cursor:
                    if key_bytes.startswith(self.data_prefix):
                        continue  # Skip data entries
                    key = key_bytes.decode()
                    data_id = value_bytes.decode()

                    # Only process each data_id once
                    if data_id not in seen_data_ids:
                        seen_data_ids.add(data_id)
                        data_id_bytes = self._data_id_to_bytes(data_id)
                        data_bytes = txn.get(data_id_bytes)
                        if data_bytes:
                            data = msgpack.unpackb(data_bytes, raw=False)
                            records[key] = data
            return records

        # Original implementation (returns all keys with their data)
        def get_all_data():
            records = {}
            with self.env.begin() as txn:
                cursor = txn.cursor()
                for key_bytes, value_bytes in cursor:
                    if key_bytes.startswith(self.data_prefix):
                        continue  # Skip data entries
                    key = key_bytes.decode()
                    data_id = value_bytes.decode()
                    data_id_bytes = self._data_id_to_bytes(data_id)
                    data_bytes = txn.get(data_id_bytes)
                    if data_bytes:
                        data = msgpack.unpackb(data_bytes, raw=False)
                        records[key] = data
            return records

        # Default to the unique data approach
        return get_unique_data()

    def close(self):
        """Close the database."""
        self.env.close()


class LMDB:
    def __init__(self, db_name: str, folder: str, map_size: int = 10485760):
        """
        Initialize LMDB database.
        :param db_name: Name of the database (e.g., "mydata").
        :param folder: Folder where DB files will be stored.
        :param map_size: Max size in bytes (default 10MB, adjust as needed).
        """
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.db_path = os.path.join(folder, db_name)

        self.env: Environment = lmdb.open(self.db_path, map_size=map_size, lock=False)

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        with self.env.begin() as txn:
            key_bytes = key.encode()
            return txn.get(key_bytes) is not None

    def batch_exists(self, keys: list[str]) -> list[bool]:
        """Check existence of multiple keys in batch."""
        result = []
        with self.env.begin() as txn:
            for key in keys:
                key_bytes = key.encode()
                result.append(txn.get(key_bytes) is not None)
        return result

    def create(self, key: str, data: dict) -> bool:
        """
        Create a new entry.
        :return: True if it was written or raises ValueError if already exists.
        """
        with self.env.begin(write=True) as txn:
            key_bytes = key.encode()
            if txn.get(key_bytes):
                raise ValueError(f"Key {key} already exists")

            return txn.put(key_bytes, msgpack.packb(data))

    def update(self, key: str, data: dict):
        """Update the data for an existing key."""
        with self.env.begin(write=True) as txn:
            key_bytes = key.encode()
            if not txn.get(key_bytes):
                raise KeyError(f"Key {key} does not exist")

            txn.put(key_bytes, msgpack.packb(data))

    def delete(self, key: str):
        """Delete a key and its associated data."""
        with self.env.begin(write=True) as txn:
            key_bytes = key.encode()
            if not txn.get(key_bytes):
                raise KeyError(f"Key {key} does not exist")
            txn.delete(key_bytes)

    def get(self, key: str) -> dict | None:
        """Get data by key."""
        with self.env.begin() as txn:
            key_bytes = key.encode()
            data_bytes = txn.get(key_bytes)
            if data_bytes is None:
                raise KeyError(f"Key {key} does not exist")
            else:
                return msgpack.unpackb(data_bytes, raw=False)
        return None

    def get_dataset(self) -> dict[str, dict]:
        """Export entire database to a dictionary."""
        records = {}
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key_bytes, data_bytes in cursor:
                key = key_bytes.decode()
                data = msgpack.unpackb(data_bytes, raw=False)
                records[key] = data
        return records

    def close(self):
        """Close the database."""
        self.env.close()
