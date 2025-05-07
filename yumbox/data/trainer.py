import random
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class FlexibleDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        preprocessors: Dict[str, Callable] = None,
        mode: str = "default",
        custom_sampler: Optional[Callable] = None,
    ):
        """
        A flexible dataset that supports consistent shuffling and iteration tracking.

        Args:
            dataframe: Input dataframe containing data paths/text/features
            preprocessors: Dict of preprocessing functions for different column types
            mode: Mode for preprocessing ('image', 'text', 'multimodal', 'default')
            custom_sampler: Optional function for custom sampling logic
        """
        self.df = dataframe
        self.preprocessors = preprocessors or {}
        self.mode = mode
        self.custom_sampler = custom_sampler
        self.dataset_size = len(dataframe)
        self._original_indices = np.arange(self.dataset_size)
        self._shuffled_indices = self._original_indices.copy()
        self.current_epoch = 0

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Get actual index from shuffled indices if needed
        actual_idx = self._shuffled_indices[idx]
        row = self.df.iloc[actual_idx]

        # Apply appropriate preprocessors based on mode
        if self.mode == "image":
            if "image" in self.preprocessors:
                return self.preprocessors["image"](row)
        elif self.mode == "text":
            if "text" in self.preprocessors:
                return self.preprocessors["text"](row)
        elif self.mode == "multimodal":
            result = {}
            for key, preprocessor in self.preprocessors.items():
                if key in row:
                    result[key] = preprocessor(row[key])
                else:
                    result[key] = preprocessor(row)
            return result
        else:  # default mode
            # Apply any preprocessing if available, otherwise return row as is
            return (
                {
                    key: prep(row) if key in row else prep(row)
                    for key, prep in self.preprocessors.items()
                }
                if self.preprocessors
                else row
            )

        return row  # Fallback to return raw row if no processing applied

    def set_epoch(self, epoch: int):
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
            return np.array([])

        return np.arange(start_idx, end_idx)

    def verify_dataset_size(self, expected_size: int) -> bool:
        """Verify that dataset size hasn't changed."""
        if self.dataset_size != expected_size:
            warnings.warn(
                f"Dataset size mismatch! Expected: {expected_size}, Actual: {self.dataset_size}"
            )
            return False
        return True


class IterationSampler(Sampler):
    """Sampler that returns indices for a specific iteration within an epoch."""

    def __init__(
        self,
        dataset: FlexibleDataset,
        iteration: int,
        iteration_size: int,
        batch_size: int,
    ):
        self.dataset = dataset
        self.iteration = iteration
        self.iteration_size = iteration_size
        self.batch_size = batch_size

    def __iter__(self):
        # Get indices for this iteration
        indices = self.dataset.get_iteration_indices(
            self.iteration, self.iteration_size
        )
        return iter(indices)

    def __len__(self):
        return min(
            self.iteration_size,
            len(self.dataset) - self.iteration * self.iteration_size,
        )


class ClusterSampler(Sampler):
    """Example of a custom sampler that samples from clusters."""

    def __init__(
        self,
        dataset: FlexibleDataset,
        cluster_ids: List[int],
        samples_per_cluster: int,
        iteration: int,
        iteration_size: int,
    ):
        self.dataset = dataset
        self.cluster_ids = cluster_ids
        self.samples_per_cluster = samples_per_cluster
        self.iteration = iteration
        self.iteration_size = iteration_size

        # Group indices by cluster
        self.clusters = {}
        for i, cluster_id in enumerate(cluster_ids):
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            self.clusters[cluster_id].append(i)

    def __iter__(self):
        # Set random seed based on epoch for consistent shuffling
        random.seed(self.dataset.current_epoch)

        # Calculate how many samples from each cluster for this iteration
        total_clusters = len(self.clusters)
        samples_per_cluster = max(1, self.iteration_size // total_clusters)

        indices = []
        for cluster_id, cluster_indices in self.clusters.items():
            # Shuffle indices within each cluster
            shuffled = random.sample(
                cluster_indices, min(samples_per_cluster, len(cluster_indices))
            )
            indices.extend(shuffled)

        # Shuffle final list of indices
        random.shuffle(indices)

        # Take only what we need for this iteration
        start_idx = self.iteration * self.iteration_size
        end_idx = min(start_idx + self.iteration_size, len(indices))
        return iter(indices[start_idx:end_idx])

    def __len__(self):
        return min(
            self.iteration_size,
            len(self.dataset) - self.iteration * self.iteration_size,
        )


def train_iteration(
    model,
    dataset: FlexibleDataset,
    epoch: int,
    iteration: int,
    iteration_size: int,
    batch_size: int,
    use_custom_sampler: bool = False,
    custom_sampler_args: Dict = None,
    **dataloader_kwargs,
):
    """
    Train for one iteration within an epoch.

    Args:
        model: The model to train
        dataset: FlexibleDataset instance
        epoch: Current epoch number
        iteration: Current iteration within the epoch
        iteration_size: Number of samples in this iteration
        batch_size: Batch size for training
        use_custom_sampler: Whether to use custom sampling logic
        custom_sampler_args: Arguments for custom sampler
        dataloader_kwargs: Additional arguments for DataLoader
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
    dataset.verify_dataset_size(mlflow.get_param("dataset_size"))

    # Create appropriate sampler
    if use_custom_sampler and custom_sampler_args:
        sampler = ClusterSampler(
            dataset=dataset,
            iteration=iteration,
            iteration_size=iteration_size,
            **custom_sampler_args,
        )
    else:
        sampler = IterationSampler(
            dataset=dataset,
            iteration=iteration,
            iteration_size=iteration_size,
            batch_size=batch_size,
        )

    # Create DataLoader with the sampler
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, **dataloader_kwargs
    )

    # Train model for this iteration
    for batch in dataloader:
        # Your training logic here
        pass

    return model


# Example usage
if __name__ == "__main__":
    # Sample data
    df = pd.DataFrame(
        {
            "image_path": [f"path/to/image_{i}.jpg" for i in range(1000)],
            "text": [f"Sample text {i}" for i in range(1000)],
            "label": np.random.randint(0, 5, 1000),
        }
    )

    # Example preprocessors
    def image_preprocessor(row):
        # In real code, this would load and transform the image
        return {"image": f"Processed {row['image_path']}", "label": row["label"]}

    def text_preprocessor(row):
        # In real code, this would tokenize the text
        return {"text": f"Tokenized {row['text']}", "label": row["label"]}

    preprocessors = {"image": image_preprocessor, "text": text_preprocessor}

    # Create dataset
    dataset = FlexibleDataset(
        dataframe=df, preprocessors=preprocessors, mode="multimodal"
    )

    # Example training loop
    with mlflow.start_run():
        mlflow.log_param("dataset_size", len(dataset))

        for epoch in range(3):
            iteration_size = 200  # Can be changed between runs
            iterations_per_epoch = len(dataset) // iteration_size

            for iteration in range(iterations_per_epoch):
                train_iteration(
                    model=None,  # Replace with your model
                    dataset=dataset,
                    epoch=epoch,
                    iteration=iteration,
                    iteration_size=iteration_size,
                    batch_size=32,
                    num_workers=4,
                    pin_memory=True,
                )
