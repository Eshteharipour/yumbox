import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler, SubsetRandomSampler
import mlflow
import warnings
from typing import Any
from collections.abc import Callable
import random


class FlexibleDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        preprocessors: dict[str, Callable] = None,
        mode: str = "default",
        image_column: str = "path",
        text_column: str = "name",
    ):
        """
        A flexible dataset that supports consistent shuffling and iteration tracking.

        Args:
            dataframe: Input dataframe containing data paths/text/features
            preprocessors: dict of preprocessing functions for different column types
            mode: Mode for preprocessing ('image', 'text', 'multimodal', 'default')
            image_column: Column name for image paths
            text_column: Column name for text data
        """
        self.df = dataframe
        self.preprocessors = preprocessors or {}
        self.mode = mode
        self.image_column = image_column
        self.text_column = text_column
        self.dataset_size = len(dataframe)
        self._original_indices = np.arange(self.dataset_size)
        self._shuffled_indices = self._original_indices.copy()
        self.current_epoch = 0

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Get actual index from shuffled indices if needed
        actual_idx = self._shuffled_indices[idx]
        row = self.df.iloc[actual_idx]

        # Apply appropriate preprocessors based on mode
        if (
            self.mode == "image"
            and "image" in self.preprocessors
            and self.image_column in row
        ):
            return {
                "image": self.preprocessors["image"](row[self.image_column]),
                "label": row.get("label", None),
                "index": actual_idx,
            }
        elif (
            self.mode == "text"
            and "text" in self.preprocessors
            and self.text_column in row
        ):
            return {
                "text": self.preprocessors["text"](row[self.text_column]),
                "label": row.get("label", None),
                "index": actual_idx,
            }
        elif self.mode == "multimodal":
            result = {"index": actual_idx}

            # Process image if available
            if "image" in self.preprocessors and self.image_column in row:
                result["image"] = self.preprocessors["image"](row[self.image_column])

            # Process text if available
            if "text" in self.preprocessors and self.text_column in row:
                result["text"] = self.preprocessors["text"](row[self.text_column])

            # Add label if available
            if "label" in row:
                result["label"] = row["label"]

            return result
        else:  # default mode
            # Just return the row as a dict with index
            return {**row.to_dict(), "index": actual_idx}

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch and update shuffle state."""
        if epoch != self.current_epoch:
            self.current_epoch = epoch
            # Use epoch as random seed to ensure consistent shuffling within epoch
            rng = np.random.RandomState(seed=epoch)
            self._shuffled_indices = self._original_indices.copy()
            rng.shuffle(self._shuffled_indices)

    def get_iteration_indices(self, iteration: int, iteration_size: int) -> np.ndarray:
        """
        Get indices for a specific iteration within the current epoch.

        Args:
            iteration: The iteration number (0-indexed)
            iteration_size: Number of samples in this iteration

        Returns:
            Numpy array of indices for this iteration
        """
        start_idx = iteration * iteration_size
        end_idx = min(start_idx + iteration_size, self.dataset_size)

        if start_idx >= self.dataset_size:
            warnings.warn(
                f"Iteration {iteration} exceeds dataset size. Returning empty array."
            )
            return np.array([], dtype=np.int64)

        return np.arange(start_idx, end_idx, dtype=np.int64)

    def verify_dataset_size(self, expected_size: int) -> bool:
        """Verify that dataset size hasn't changed."""
        if self.dataset_size != expected_size:
            warnings.warn(
                f"Dataset size mismatch! Expected: {expected_size}, Actual: {self.dataset_size}"
            )
            return False
        return True


class ClusterSampler(Sampler):
    """Sample from clusters ensuring each batch contains samples from different clusters."""

    def __init__(
        self,
        dataset: FlexibleDataset,
        cluster_ids: list[int],
        samples_per_cluster: int,
        batch_size: int,
    ):
        """
        Initialize a sampler that creates batches with samples from each cluster.

        Args:
            dataset: The dataset to sample from
            cluster_ids: list of cluster IDs for each sample in the dataset
            samples_per_cluster: Number of samples to take from each cluster
            batch_size: Size of each batch
        """
        self.dataset = dataset
        assert len(cluster_ids) == len(dataset), "Cluster IDs must match dataset length"
        self.cluster_ids = cluster_ids
        self.samples_per_cluster = samples_per_cluster
        self.batch_size = batch_size

        # Group indices by cluster
        self.clusters: dict[int, list[int]] = {}
        for i, cluster_id in enumerate(cluster_ids):
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            self.clusters[cluster_id].append(i)

    def __iter__(self):
        # Set random seed based on epoch for consistent shuffling
        random.seed(self.dataset.current_epoch)

        # Shuffle indices within each cluster
        shuffled_clusters = {}
        for cluster_id, cluster_indices in self.clusters.items():
            # Make a copy to avoid modifying the original
            indices = cluster_indices.copy()
            random.shuffle(indices)
            shuffled_clusters[cluster_id] = indices

        # Create batches with samples from each cluster
        batches = []
        remaining_clusters = list(shuffled_clusters.keys())

        while remaining_clusters:
            batch = []
            # Try to fill a batch from different clusters
            for cluster_id in list(
                remaining_clusters
            ):  # Create a copy for safe iteration
                if not shuffled_clusters[cluster_id]:
                    remaining_clusters.remove(cluster_id)
                    continue

                # Take samples from this cluster
                samples_to_take = min(
                    self.samples_per_cluster, len(shuffled_clusters[cluster_id])
                )
                batch.extend(shuffled_clusters[cluster_id][:samples_to_take])
                shuffled_clusters[cluster_id] = shuffled_clusters[cluster_id][
                    samples_to_take:
                ]

                # If batch is full, add it and start a new one
                if len(batch) >= self.batch_size:
                    batches.append(batch[: self.batch_size])
                    batch = batch[self.batch_size :]

            # Add any remaining samples
            if batch:
                batches.append(batch)

        # Flatten batches
        indices = []
        for batch in batches:
            indices.extend(batch)

        return iter(indices)

    def __len__(self) -> int:
        # This is an approximation
        return len(self.dataset)


class TripletSampler(Sampler):
    """
    Sampler that creates triplets (anchor, positive, negative) for triplet loss.
    Each triplet contains:
    - anchor: a sample
    - positive: a sample with the same label as the anchor
    - negative: a sample with a different label than the anchor
    """

    def __init__(
        self,
        dataset: FlexibleDataset,
        labels: list[int],
        batch_size: int,
        neg_to_pos_ratio: int = 1,
    ):
        """
        Initialize a sampler that creates triplets for triplet loss.

        Args:
            dataset: The dataset to sample from
            labels: list of labels for each sample in the dataset
            batch_size: Size of each batch (must be divisible by (neg_to_pos_ratio + 2))
            neg_to_pos_ratio: Number of negative samples per positive sample
        """
        self.dataset = dataset
        assert len(labels) == len(dataset), "Labels must match dataset length"
        self.labels = labels
        self.batch_size = batch_size
        self.neg_to_pos_ratio = neg_to_pos_ratio

        # Assert batch size is divisible by triplet size
        triplet_size = 2 + neg_to_pos_ratio  # anchor + positive + negatives
        assert (
            batch_size % triplet_size == 0
        ), f"Batch size must be divisible by {triplet_size}"

        # Group indices by label
        self.label_to_indices: dict[int, list[int]] = {}
        for i, label in enumerate(labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(i)

        # Verify we have at least 2 samples per class
        for label, indices in self.label_to_indices.items():
            if len(indices) < 2:
                warnings.warn(
                    f"Label {label} has fewer than 2 samples, which may cause issues with triplet sampling."
                )

        # Verify we have at least 2 classes
        if len(self.label_to_indices) < 2:
            raise ValueError(
                "TripletSampler requires at least 2 different classes/labels."
            )

    def __iter__(self):
        # Set random seed based on epoch for consistent shuffling
        random.seed(self.dataset.current_epoch)

        # Calculate number of triplets
        triplets_per_batch = self.batch_size // (2 + self.neg_to_pos_ratio)
        triplets = []

        # Generate triplets
        labels = list(self.label_to_indices.keys())
        for _ in range(
            triplets_per_batch * 100
        ):  # Generate more than needed, we'll sample from these
            # Randomly select a label with at least 2 samples
            valid_labels = [l for l in labels if len(self.label_to_indices[l]) >= 2]
            if not valid_labels:
                break

            anchor_label = random.choice(valid_labels)

            # Select anchor and positive
            anchor_idx, pos_idx = random.sample(self.label_to_indices[anchor_label], 2)

            # Select negative labels (different from anchor)
            neg_labels = [l for l in labels if l != anchor_label]
            if not neg_labels:
                continue

            # Select negative samples
            neg_indices = []
            for _ in range(self.neg_to_pos_ratio):
                neg_label = random.choice(neg_labels)
                neg_idx = random.choice(self.label_to_indices[neg_label])
                neg_indices.append(neg_idx)

            # Add triplet
            triplet = [anchor_idx, pos_idx] + neg_indices
            triplets.append(triplet)

        # Shuffle and limit triplets
        random.shuffle(triplets)
        triplets = triplets[:triplets_per_batch]

        # Flatten triplets
        indices = []
        for triplet in triplets:
            indices.extend(triplet)

        return iter(indices)

    def __len__(self) -> int:
        # This is an approximation
        return len(self.dataset)


class SiameseSampler(Sampler):
    """
    Sampler that creates pairs (anchor, second) where second can be either
    from the same class (positive) or different class (negative).
    """

    def __init__(
        self,
        dataset: FlexibleDataset,
        labels: list[int],
        batch_size: int,
        pos_neg_ratio: float = 0.5,  # Ratio of positive pairs, e.g., 0.5 means half positive, half negative
    ):
        """
        Initialize a sampler that creates pairs for siamese networks.

        Args:
            dataset: The dataset to sample from
            labels: list of labels for each sample in the dataset
            batch_size: Size of each batch (must be divisible by 2)
            pos_neg_ratio: Ratio of positive pairs to all pairs
        """
        self.dataset = dataset
        assert len(labels) == len(dataset), "Labels must match dataset length"
        self.labels = labels
        self.batch_size = batch_size
        self.pos_neg_ratio = pos_neg_ratio

        # Assert batch size is divisible by 2
        assert batch_size % 2 == 0, "Batch size must be divisible by 2 for pairs"

        # Group indices by label
        self.label_to_indices: dict[int, list[int]] = {}
        for i, label in enumerate(labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(i)

        # Verify we have at least 2 classes
        if len(self.label_to_indices) < 2:
            raise ValueError(
                "SiameseSampler requires at least 2 different classes/labels."
            )

    def __iter__(self):
        # Set random seed based on epoch for consistent shuffling
        random.seed(self.dataset.current_epoch)

        # Calculate number of pairs
        pairs_per_batch = self.batch_size // 2
        pos_pairs = int(pairs_per_batch * self.pos_neg_ratio)
        neg_pairs = pairs_per_batch - pos_pairs

        pairs = []

        # Generate positive pairs
        labels_with_multiple_samples = [
            l
            for l in self.label_to_indices.keys()
            if len(self.label_to_indices[l]) >= 2
        ]

        for _ in range(pos_pairs):
            if not labels_with_multiple_samples:
                break

            # Select a label with at least 2 samples
            label = random.choice(labels_with_multiple_samples)

            # Select two different samples from this label
            idx1, idx2 = random.sample(self.label_to_indices[label], 2)
            pairs.append((idx1, idx2))

        # Generate negative pairs
        all_labels = list(self.label_to_indices.keys())
        for _ in range(neg_pairs):
            if len(all_labels) < 2:
                break

            # Select two different labels
            label1, label2 = random.sample(all_labels, 2)

            # Select one sample from each label
            idx1 = random.choice(self.label_to_indices[label1])
            idx2 = random.choice(self.label_to_indices[label2])
            pairs.append((idx1, idx2))

        # Shuffle pairs
        random.shuffle(pairs)

        # Flatten pairs
        indices = []
        for idx1, idx2 in pairs:
            indices.extend([idx1, idx2])

        return iter(indices)

    def __len__(self) -> int:
        # This is an approximation
        return len(self.dataset)


class ContrastiveSampler(Sampler):
    """
    Sampler that creates batches suitable for contrastive learning.
    Each batch contains a mix of samples so that there are multiple instances of the same class.
    """

    def __init__(
        self,
        dataset: FlexibleDataset,
        labels: list[int],
        batch_size: int,
        num_classes_per_batch: int = 8,  # Number of different classes in each batch
        samples_per_class: int = 4,  # Number of samples per class in each batch
    ):
        """
        Initialize a sampler for contrastive learning.

        Args:
            dataset: The dataset to sample from
            labels: list of labels for each sample in the dataset
            batch_size: Size of each batch
            num_classes_per_batch: Number of different classes in each batch
            samples_per_class: Number of samples per class in each batch
        """
        self.dataset = dataset
        assert len(labels) == len(dataset), "Labels must match dataset length"
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes_per_batch = min(num_classes_per_batch, len(set(labels)))
        self.samples_per_class = samples_per_class

        # Assert batch size is compatible
        assert (
            batch_size >= num_classes_per_batch * samples_per_class
        ), "Batch size must be at least num_classes_per_batch * samples_per_class"

        # Group indices by label
        self.label_to_indices: dict[int, list[int]] = {}
        for i, label in enumerate(labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(i)

        # Filter out labels with too few samples
        self.valid_labels = [
            l
            for l in self.label_to_indices.keys()
            if len(self.label_to_indices[l]) >= samples_per_class
        ]

        if len(self.valid_labels) < num_classes_per_batch:
            warnings.warn(
                f"Only {len(self.valid_labels)} classes have {samples_per_class}+ samples. "
                f"Reducing num_classes_per_batch from {num_classes_per_batch} to {len(self.valid_labels)}."
            )
            self.num_classes_per_batch = len(self.valid_labels)

    def __iter__(self):
        # Set random seed based on epoch for consistent shuffling
        random.seed(self.dataset.current_epoch)

        # Shuffle the order of labels
        random.shuffle(self.valid_labels)

        # Calculate number of batches
        labels_per_batch = min(self.num_classes_per_batch, len(self.valid_labels))
        batches = []

        # Create a cycle of valid labels
        label_cycle = self.valid_labels.copy()

        while True:
            if len(label_cycle) < labels_per_batch:
                # Reshuffle and extend cycle if needed
                random.shuffle(self.valid_labels)
                label_cycle.extend(self.valid_labels)

            # Get labels for this batch
            batch_labels = label_cycle[:labels_per_batch]
            label_cycle = label_cycle[labels_per_batch:]

            # Create batch
            batch = []
            for label in batch_labels:
                # Shuffle indices for this label
                indices = self.label_to_indices[label].copy()
                random.shuffle(indices)

                # Take required number of samples
                samples_to_take = min(self.samples_per_class, len(indices))
                batch.extend(indices[:samples_to_take])

            batches.append(batch)

            # Stop when we have enough batches
            if len(batches) * self.batch_size >= len(self.dataset):
                break

        # Flatten batches
        indices = []
        for batch in batches:
            indices.extend(batch)

        return iter(indices[: len(self.dataset)])

    def __len__(self) -> int:
        return len(self.dataset)


def get_dataloader(
    dataset: FlexibleDataset,
    epoch: int,
    iteration: int,
    iteration_size: int,
    batch_size: int,
    sampler: Sampler|None = None,
    **dataloader_kwargs,
) -> DataLoader:
    """
    Train for one iteration within an epoch.

    Args:
        model: The model to train
        dataset: FlexibleDataset instance
        epoch: Current epoch number
        iteration: Current iteration within the epoch
        iteration_size: Number of batches in this iteration
        batch_size: Batch size for training
        sampler: Optional sampler to use for this iteration
        dataloader_kwargs: Additional arguments for DataLoader

    Returns:
        The trained model
    """
    # Set epoch to ensure consistent shuffling
    dataset.set_epoch(epoch)

    # Log training metadata to MLflow
    mlflow.log_param("epoch", epoch)
    mlflow.log_param("iteration", iteration)
    mlflow.log_param("iteration_size", iteration_size)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("total_samples", iteration_size * batch_size)
    mlflow.log_param("dataset_size", dataset.dataset_size)

    # Verify dataset size hasn't changed
    expected_size = (
        mlflow.get_param("dataset_size")
        if mlflow.get_param("dataset_size")
        else dataset.dataset_size
    )
    dataset.verify_dataset_size(expected_size)

    # If no sampler provided, get indices for this iteration
    if sampler is None:
        iter_indices = dataset.get_iteration_indices(iteration, iteration_size)
        subset_sampler = SubsetRandomSampler(iter_indices)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=subset_sampler, **dataloader_kwargs
        )
    else:
        # Use the provided sampler
        dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, **dataloader_kwargs
        )

    return dataloader
