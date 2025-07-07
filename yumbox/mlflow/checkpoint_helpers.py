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
) -> tuple[set[str], dict[str, str], set[str]]:
    """
    Analyze MLflow experiments to find checkpoints to keep.

    Args:
        storage_path: Path to MLflow storage
        metric_direction_map: Dict mapping metric name patterns to 'min' or 'max'
        ignore_metrics: A set of metric names to ignore during analysis.

    Returns:
        Tuple of (checkpoints_to_keep, reasons_dict, all_referenced_checkpoints)
        - checkpoints_to_keep: Set of absolute paths to keep
        - reasons_dict: Dict mapping checkpoint path to reason for keeping
        - all_referenced_checkpoints: Set of all checkpoints referenced in MLflow
    """
    logger = BFG["logger"]

    # Set MLflow tracking URI
    from yumbox.mlflow import set_tracking_uri

    set_tracking_uri(storage_path)
    client = MlflowClient()

    checkpoints_to_keep = set()
    all_referenced_checkpoints = set()
    reasons = {}

    # Get all active experiments
    experiments = client.search_experiments(
        view_type=mlflow.entities.ViewType.ACTIVE_ONLY
    )

    if not experiments:
        logger.warning("No active experiments found")
        return checkpoints_to_keep, reasons, all_referenced_checkpoints

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
            all_referenced_checkpoints.add(abs_path)
            reasons[abs_path] = f"last_run_{exp.name}"
            logger.info(f"Keeping last run checkpoint: {abs_path}")

        # Analyze metrics to find best performing runs
        experiment_metrics: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
        run_checkpoint_map = {}

        # Collect all metrics and checkpoints from all runs
        for run in parent_runs:
            checkpoint_path = run.data.params.get("model_path")
            if not checkpoint_path:
                continue

            abs_checkpoint_path = os.path.abspath(checkpoint_path)
            all_referenced_checkpoints.add(abs_checkpoint_path)

            if not os.path.exists(checkpoint_path):
                continue

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
    return checkpoints_to_keep, reasons, all_referenced_checkpoints


def analyze_checkpoint_status(
    checkpoints_dir: str,
    storage_path: str,
    metric_direction_map: dict[str, Literal["min", "max"]],
    ignore_metrics: set[str],
) -> tuple[set[str], set[str], set[str], set[str], dict[str, str]]:
    """
    Analyze checkpoint status and provide recommendations.

    Args:
        checkpoints_dir: Directory containing checkpoint files
        storage_path: Path to MLflow storage
        metric_direction_map: Dict mapping metric patterns to min/max
        ignore_metrics: A set of metric names to ignore during analysis.

    Returns:
        Tuple of (keep_set, remove_set, deleted_set, non_referenced_set, reasons_dict)
        - keep_set: Checkpoints that exist on disk and should be kept
        - remove_set: MLflow-referenced checkpoints that exist on disk but should be removed
        - deleted_set: MLflow-referenced checkpoints that are missing from disk
        - non_referenced_set: Checkpoints that exist on disk but are not referenced in MLflow
        - reasons_dict: Dict mapping checkpoint path to reason for keeping
    """
    logger = BFG["logger"]

    # Get all checkpoints in directory
    all_checkpoints = get_all_checkpoints(checkpoints_dir)

    # Get checkpoints that should be kept based on MLflow analysis
    keep_checkpoints, reasons, all_referenced_checkpoints = get_experiment_checkpoints(
        storage_path, metric_direction_map, ignore_metrics
    )

    # Find deleted checkpoints (referenced in MLflow but not on disk)
    deleted_checkpoints = all_referenced_checkpoints - all_checkpoints

    # Find checkpoints to keep (exist on disk and should be kept)
    keep_set = keep_checkpoints & all_checkpoints

    # Find MLflow-referenced checkpoints that exist on disk
    referenced_on_disk = all_referenced_checkpoints & all_checkpoints

    # Find checkpoints to remove (MLflow-referenced, exist on disk, but not in keep list)
    remove_set = referenced_on_disk - keep_checkpoints

    # Find non-referenced checkpoints (exist on disk but not referenced in MLflow)
    non_referenced_set = all_checkpoints - all_referenced_checkpoints

    logger.info(f"Analysis complete:")
    logger.info(f"  - Checkpoints to keep: {len(keep_set)}")
    logger.info(f"  - MLflow-referenced checkpoints to remove: {len(remove_set)}")
    logger.info(f"  - Already deleted checkpoints: {len(deleted_checkpoints)}")
    logger.info(f"  - Non-referenced checkpoints: {len(non_referenced_set)}")

    return keep_set, remove_set, deleted_checkpoints, non_referenced_set, reasons


def format_checkpoint_report(
    keep_set: set[str],
    remove_set: set[str],
    deleted_set: set[str],
    non_referenced_set: set[str],
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
    report.append(f"  MLflow-referenced checkpoints to remove: {len(remove_set)}")
    report.append(f"  Already deleted checkpoints: {len(deleted_set)}")
    report.append(f"  Non-referenced checkpoints: {len(non_referenced_set)}")
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

    # MLflow-referenced checkpoints to remove
    if remove_set:
        report.append("MLFLOW-REFERENCED CHECKPOINTS TO REMOVE:")
        report.append("-" * 40)
        for checkpoint in sorted(remove_set):
            report.append(f"  REMOVE: {relative_path(checkpoint)}")
        report.append("")

    # Non-referenced checkpoints
    if non_referenced_set:
        report.append("NON-REFERENCED CHECKPOINTS (not in MLflow experiments):")
        report.append("-" * 40)
        for checkpoint in sorted(non_referenced_set):
            report.append(f"  NON-REF: {relative_path(checkpoint)}")
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
    total_remove_size = 0
    total_non_ref_size = 0

    if remove_set:
        for checkpoint in remove_set:
            if os.path.exists(checkpoint):
                total_remove_size += os.path.getsize(checkpoint)

    if non_referenced_set:
        for checkpoint in non_referenced_set:
            if os.path.exists(checkpoint):
                total_non_ref_size += os.path.getsize(checkpoint)

    if total_remove_size > 0 or total_non_ref_size > 0:
        report.append("DISK USAGE ANALYSIS:")
        report.append("-" * 40)
        if total_remove_size > 0:
            size_gb = total_remove_size / (1024**3)
            report.append(f"  Space from MLflow-referenced removals: {size_gb:.2f} GB")
        if total_non_ref_size > 0:
            size_gb = total_non_ref_size / (1024**3)
            report.append(f"  Space from non-referenced checkpoints: {size_gb:.2f} GB")
        if total_remove_size > 0 and total_non_ref_size > 0:
            total_size_gb = (total_remove_size + total_non_ref_size) / (1024**3)
            report.append(f"  Total potential space to free: {total_size_gb:.2f} GB")
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
