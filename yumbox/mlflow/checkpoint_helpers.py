import os
from collections import defaultdict
from pathlib import Path
from typing import Literal

import mlflow
from mlflow import MlflowClient

from yumbox.cache import BFG


def get_all_checkpoints(checkpoints_dir: str) -> set[str]:
    """
    Recursively find all checkpoint files in the given directory.
    Returns absolute paths of checkpoint files.
    """
    logger = BFG["logger"]
    checkpoints = set()

    checkpoint_extensions = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}
    checkpoints_path = Path(checkpoints_dir)

    if not checkpoints_path.exists():
        logger.error(f"Checkpoints directory does not exist: {checkpoints_dir}")
        return checkpoints

    for root, dirs, files in os.walk(checkpoints_path):
        for file in files:
            if any(file.endswith(ext) for ext in checkpoint_extensions):
                full_path = os.path.abspath(os.path.join(root, file))
                checkpoints.add(full_path)

    logger.info(f"Found {len(checkpoints)} checkpoint files in {checkpoints_dir}")
    return checkpoints


def get_experiment_checkpoints(
    storage_path: str,
    metric_direction_map: dict[str, Literal["min", "max"]],
    ignore_metrics: set[str],
) -> tuple[set[str], dict[str, str]]:
    """
    Analyze MLflow experiments to find checkpoints to keep.

    Args:
        storage_path: Path to MLflow storage
        metric_direction_map: Dict mapping metric name patterns to 'min' or 'max'
        ignore_metrics: A set of metric names to ignore during analysis.

    Returns:
        Tuple of (checkpoints_to_keep, reasons_dict)
        - checkpoints_to_keep: Set of absolute paths to keep
        - reasons_dict: Dict mapping checkpoint path to reason for keeping
    """
    logger = BFG["logger"]

    # Set MLflow tracking URI
    from yumbox.mlflow import set_tracking_uri

    set_tracking_uri(storage_path)
    client = MlflowClient()

    checkpoints_to_keep = set()
    reasons = {}

    # Get all active experiments
    experiments = client.search_experiments(
        view_type=mlflow.entities.ViewType.ACTIVE_ONLY
    )

    if not experiments:
        logger.warning("No active experiments found")
        return checkpoints_to_keep, reasons

    missing_direction_keys = set()

    for exp in experiments:
        logger.info(f"Processing experiment: {exp.name}")

        # Get all successful runs for this experiment
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="status = 'FINISHED'",
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
            order_by=["start_time DESC"],
        )

        # Filter out child runs, keep only parent runs
        parent_runs = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]

        if not parent_runs:
            logger.warning(f"No successful parent runs found for experiment {exp.name}")
            continue

        # Always keep the last (most recent) run's checkpoint
        last_run = parent_runs[0]  # Most recent due to DESC order
        last_run_checkpoint = last_run.data.params.get("model_path")

        if last_run_checkpoint and os.path.exists(last_run_checkpoint):
            abs_path = os.path.abspath(last_run_checkpoint)
            checkpoints_to_keep.add(abs_path)
            reasons[abs_path] = f"last_run_{exp.name}"
            logger.info(f"Keeping last run checkpoint: {abs_path}")

        # Analyze metrics to find best performing runs
        experiment_metrics: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
        run_checkpoint_map = {}

        # Collect all metrics and checkpoints from all runs
        for run in parent_runs:
            checkpoint_path = run.data.params.get("model_path")
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                continue

            abs_checkpoint_path = os.path.abspath(checkpoint_path)
            run_checkpoint_map[run.info.run_id] = abs_checkpoint_path

            # Collect metrics for this run
            for metric_name, metric_value in run.data.metrics.items():
                experiment_metrics[metric_name].append(
                    {
                        "run_id": run.info.run_id,
                        "value": metric_value,
                        "checkpoint": abs_checkpoint_path,
                    }
                )

        # Find best checkpoints for each metric
        for metric_name, metric_data in experiment_metrics.items():
            # if metric_name.lower() == "epoch":
            #     continue

            import re

            ignore_current_metric = False
            for pattern in ignore_metrics:

                if pattern.lower() == metric_name.lower():
                    logger.info(f"Ignoring metric '{metric_name}' as requested.")
                    ignore_current_metric = True
                    break

                if re.search(
                    r"(?<![^ _-])" + re.escape(pattern.lower()) + r"(?![^ _-])",
                    metric_name.lower(),
                ):
                    logger.info(f"Ignoring metric '{metric_name}' as requested.")
                    ignore_current_metric = True
                    break

            if ignore_current_metric == True:
                continue

            if len(metric_data) <= 1:
                continue  # Skip if only one data point

            # Determine direction for this metric
            direction = None
            for pattern, dir_val in metric_direction_map.items():
                if metric_name.lower() == "epoch":
                    direction = "max"
                    break
                if pattern.lower() == metric_name.lower():
                    direction = dir_val
                    break
                if re.search(
                    r"(?<![^ _-])" + re.escape(pattern.lower()) + r"(?![^ _-])",
                    metric_name.lower(),
                ):
                    direction = dir_val
                    break

            if direction is None:
                missing_direction_keys.add(metric_name)
                continue

            # Find best run for this metric
            if direction == "max":
                best_run_data = max(metric_data, key=lambda x: x["value"])
            else:  # direction == "min"
                best_run_data = min(metric_data, key=lambda x: x["value"])

            best_checkpoint = best_run_data["checkpoint"]
            checkpoints_to_keep.add(best_checkpoint)

            # Update reason (might overwrite, but that's fine)
            reason_key = f"best_{metric_name}_{exp.name}"
            if best_checkpoint in reasons:
                reasons[best_checkpoint] += f", {reason_key}"
            else:
                reasons[best_checkpoint] = reason_key

            logger.info(
                f"Keeping best {metric_name} checkpoint: {best_checkpoint} (value: {best_run_data['value']:.4f})"
            )

    # Report missing direction mappings
    if missing_direction_keys:
        raise ValueError(
            f"Missing direction mapping for metrics: {sorted(missing_direction_keys)}. "
            f"Please add these keys to your metric_direction_map with 'min' or 'max' values."
        )

    logger.info(f"Total checkpoints to keep: {len(checkpoints_to_keep)}")
    return checkpoints_to_keep, reasons


def analyze_checkpoint_status(
    checkpoints_dir: str,
    storage_path: str,
    metric_direction_map: dict[str, Literal["min", "max"]],
    ignore_metrics: set[str],
) -> tuple[set[str], set[str], set[str], dict[str, str]]:
    """
    Analyze checkpoint status and provide recommendations.

    Args:
        checkpoints_dir: Directory containing checkpoint files
        storage_path: Path to MLflow storage
        metric_direction_map: Dict mapping metric patterns to min/max
        ignore_metrics: A set of metric names to ignore during analysis.

    Returns:
        Tuple of (keep_set, remove_set, deleted_set, reasons_dict)
    """
    logger = BFG["logger"]

    # Get all checkpoints in directory
    all_checkpoints = get_all_checkpoints(checkpoints_dir)

    # Get checkpoints that should be kept based on MLflow analysis
    keep_checkpoints, reasons = get_experiment_checkpoints(
        storage_path, metric_direction_map, ignore_metrics
    )

    # Find deleted checkpoints (referenced in MLflow but not on disk)
    deleted_checkpoints = keep_checkpoints - all_checkpoints

    # Find checkpoints to keep (exist on disk and should be kept)
    keep_set = keep_checkpoints & all_checkpoints

    # Find checkpoints to remove (exist on disk but not in keep list)
    # TODO: only removed checkpoints referenced in mlflow,
    # TODO: report a separate set for non-referenced checkpoints that exist in directory
    remove_set = all_checkpoints - keep_checkpoints

    logger.info(f"Analysis complete:")
    logger.info(f"  - Checkpoints to keep: {len(keep_set)}")
    logger.info(f"  - Checkpoints to remove: {len(remove_set)}")
    logger.info(f"  - Already deleted checkpoints: {len(deleted_checkpoints)}")

    return keep_set, remove_set, deleted_checkpoints, reasons


def format_checkpoint_report(
    keep_set: set[str],
    remove_set: set[str],
    deleted_set: set[str],
    reasons: dict[str, str],
    checkpoints_dir: str,
) -> str:
    """Format a detailed report of checkpoint analysis."""

    def relative_path(path: str) -> str:
        """Convert absolute path to relative from checkpoints_dir if possible."""
        try:
            return os.path.relpath(path, checkpoints_dir)
        except ValueError:
            return path

    report = []
    report.append("=" * 80)
    report.append("CHECKPOINT ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    # Summary
    report.append("SUMMARY:")
    report.append(f"  Total checkpoints to keep: {len(keep_set)}")
    report.append(f"  Total checkpoints to remove: {len(remove_set)}")
    report.append(f"  Already deleted checkpoints: {len(deleted_set)}")
    report.append("")

    # Checkpoints to keep
    if keep_set:
        report.append("CHECKPOINTS TO KEEP:")
        report.append("-" * 40)
        for checkpoint in sorted(keep_set):
            reason = reasons.get(checkpoint, "unknown")
            report.append(f"  KEEP: {relative_path(checkpoint)}")
            report.append(f"        Reason: {reason}")
            report.append("")

    # Checkpoints to remove
    if remove_set:
        report.append("CHECKPOINTS TO REMOVE:")
        report.append("-" * 40)
        for checkpoint in sorted(remove_set):
            report.append(f"  REMOVE: {relative_path(checkpoint)}")
        report.append("")

    # Deleted checkpoints
    if deleted_set:
        report.append("ALREADY DELETED CHECKPOINTS (referenced in MLflow but missing):")
        report.append("-" * 40)
        for checkpoint in sorted(deleted_set):
            reason = reasons.get(checkpoint, "unknown")
            report.append(f"  DELETED: {relative_path(checkpoint)}")
            report.append(f"           Reason: {reason}")
        report.append("")

    # Disk usage estimation
    if remove_set:
        total_size = 0
        for checkpoint in remove_set:
            if os.path.exists(checkpoint):
                total_size += os.path.getsize(checkpoint)

        size_gb = total_size / (1024**3)
        report.append(f"ESTIMATED SPACE TO FREE: {size_gb:.2f} GB")
        report.append("")

    report.append("=" * 80)

    return "\n".join(report)


def execute_checkpoint_removal(remove_set: set[str], dry_run: bool = True) -> None:
    """
    Execute checkpoint removal.

    Args:
        remove_set: Set of checkpoint paths to remove
        dry_run: If True, only print what would be removed without actually removing
    """
    logger = BFG["logger"]

    if not remove_set:
        logger.info("No checkpoints to remove.")
        return

    if dry_run:
        logger.info("DRY RUN: The following checkpoints would be removed:")
        for checkpoint in sorted(remove_set):
            logger.info(f"  Would remove: {checkpoint}")
    else:
        logger.info("Removing checkpoints:")
        removed_count = 0
        for checkpoint in sorted(remove_set):
            try:
                if os.path.exists(checkpoint):
                    os.remove(checkpoint)
                    logger.info(f"  Removed: {checkpoint}")
                    removed_count += 1
                else:
                    logger.warning(f"  File not found: {checkpoint}")
            except Exception as e:
                logger.error(f"  Failed to remove {checkpoint}: {e}")

        logger.info(f"Successfully removed {removed_count} checkpoint files.")
