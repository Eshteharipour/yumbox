import os

import lmdb
import msgpack


class LMDB_API:
    def __init__(self, db_name: str, folder: str, map_size: int = 10485760):
        """
        Initialize LMDB database.
        :param db_name: Name of the database (e.g., "mydata").
        :param folder: Folder where DB files will be stored.
        :param map_size: Max size in bytes (default 10MB, adjust as needed).
        """
        self.folder = folder
        os.makedirs(folder, exist_ok=True)  # Ensure folder exists
        self.db_path = os.path.join(folder, db_name)
        self.env = lmdb.open(
            self.db_path, map_size=map_size, lock=False
        )  # Single file in folder
        self.data_prefix = b"data:"  # Namespace for actual data

    def _key_to_bytes(self, key: str) -> bytes:
        """Convert str key to bytes."""
        return key.encode()

    def _data_id_to_bytes(self, data_id: str) -> bytes:
        """Convert data ID to bytes."""
        return self.data_prefix + data_id.encode()

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        with self.env.begin() as txn:
            key_bytes = self._key_to_bytes(key)
            return txn.get(key_bytes) is not None

    def batch_exists(self, keys: list[str]) -> dict[str, bool]:
        """Check existence of multiple keys in batch."""
        result = {}
        with self.env.begin() as txn:
            for key in keys:
                key_bytes = self._key_to_bytes(key)
                result[key] = txn.get(key_bytes) is not None
        return result

    def create(self, key: str, data: dict, data_id: str | None = None) -> str:
        """
        Create a new entry. If data_id is provided, use it; otherwise, generate one.
        :return: The data_id used.
        """
        with self.env.begin(write=True) as txn:
            key_bytes = self._key_to_bytes(key)
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
            key_bytes = self._key_to_bytes(key)
            data_id = txn.get(key_bytes)
            if not data_id:
                raise KeyError(f"Key {key} does not exist")

            data_id_bytes = self._data_id_to_bytes(data_id.decode())
            txn.put(data_id_bytes, msgpack.packb(data))

    def delete(self, key: str):
        """Delete a key. Data remains if referenced by other keys."""
        with self.env.begin(write=True) as txn:
            key_bytes = self._key_to_bytes(key)
            if not txn.get(key_bytes):
                raise KeyError(f"Key {key} does not exist")
            txn.delete(key_bytes)

    def get(self, key: str) -> dict | None:
        """Get data by key."""
        with self.env.begin() as txn:
            key_bytes = self._key_to_bytes(key)
            data_id = txn.get(key_bytes)
            if data_id:
                data_id_bytes = self._data_id_to_bytes(data_id.decode())
                data_bytes = txn.get(data_id_bytes)
                if data_bytes:
                    return msgpack.unpackb(data_bytes, raw=False)
        return None

    def get_dataset(self) -> list:
        """Export entire database to a list."""
        records = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key_bytes, value_bytes in cursor:
                if key_bytes.startswith(self.data_prefix):
                    continue  # Skip data entries
                key = key_bytes.decode()
                data_id = value_bytes.decode()
                data_bytes = txn.get(self._data_id_to_bytes(data_id))
                if data_bytes:
                    data = msgpack.unpackb(data_bytes, raw=False)
                    records.append(
                        {"a": key[0], "b": key[1], "c": key[2], "d": key[3], **data}
                    )
        return records

    def close(self):
        """Close the database."""
        self.env.close()
