import os
import re
import subprocess
import tempfile
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Literal, Optional, Union

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient, entities
from omegaconf import DictConfig, OmegaConf

from yumbox.cache import BFG

from .tools import flatten_json_columns

DATE_TIME_FORMAT = "%Y-%m-%dT%H-%M-%S%z"


def now_formatted():
    return datetime.now().strftime(DATE_TIME_FORMAT)


def log_params(cfg: DictConfig, prefix: str | None = None):
    """Recursively log parameters from a nested OmegaConf configuration"""
    for k in cfg.keys():
        # Create the parameter key with proper prefix
        param_key = k if prefix is None else f"{prefix}.{k}"

        # Get the value
        v = cfg[k]

        # If value is a nested dict-like object, recurse
        if hasattr(v, "items") and callable(v.items):
            log_params(v, param_key)
        # Otherwise, log the parameter
        else:
            mlflow.log_param(param_key, v)


def log_scores_dict(
    scores_dict: dict, name: str | None = None, step: int | None = None
):
    dict_w_name = {}
    for k, v in scores_dict.items():
        if name:
            k = name + "_" + k
        mlflow.log_metric(k, v, step=step)
        dict_w_name[k] = v
    return dict_w_name


def log_config(
    cfg: DictConfig, logger: None | Logger, as_artifact=True, as_params=True
):
    if as_artifact:
        temp_dir = tempfile.mkdtemp()
        config_path = os.path.join(temp_dir, "config.yaml")

        # OmegaConf.to_yaml(CFG)
        OmegaConf.save(cfg, config_path)
        mlflow.log_artifact(config_path)

        os.remove(config_path)
        os.rmdir(temp_dir)

    if as_params:
        cfg_dict = OmegaConf.to_container(cfg)
        mlflow.log_params(cfg_dict)

    if logger:
        logger.info(cfg)


def check_incomplete_mlflow_runs(mlflow_dir="mlruns"):
    """Check for incomplete MLflow runs across all experiments and log warnings.
    Args:
        mlflow_dir (str): Base directory for MLflow runs (default: 'mlruns')
    """
    logger = BFG["logger"]

    # Check MLflow runs across all experiments
    client = MlflowClient()
    if os.path.exists(mlflow_dir):
        for experiment_id in os.listdir(mlflow_dir):
            experiment_path = os.path.join(mlflow_dir, experiment_id)
            if os.path.isdir(experiment_path) and experiment_id.isdigit():
                try:
                    # Query the run status directly
                    runs = client.search_runs(experiment_ids=[experiment_id])
                    for run in runs:
                        if run.info.status != "FINISHED":
                            logger.warning(
                                f"Incomplete MLflow run found: Run ID {run.info.run_id} "
                                f"in Experiment ID {experiment_id} (status: {run.info.status})"
                            )
                except Exception as e:
                    logger.error(f"Error checking experiment {experiment_id}: {str(e)}")
    else:
        logger.info(f"MLflow directory not found: {mlflow_dir}")


def natural_sorter(items: list[str]):
    # from natsort import natsorted
    # return natsorted(items)

    import re

    def natural_sort_key(s: str):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", s)
        ]

    items.sort(key=natural_sort_key)
    return items


def get_configs(configs_dir="configs", ext=".yaml"):
    config_files = [f for f in os.listdir(configs_dir) if f.endswith(ext)]
    return [os.path.join(configs_dir, config_file) for config_file in config_files]


def get_committed_configs(configs_dir="configs", ext=".yaml"):
    logger = BFG["logger"]

    try:
        result = subprocess.run(
            ["git", "ls-files", configs_dir], capture_output=True, text=True, check=True
        )
        all_files = result.stdout.splitlines()
        yaml_files = [f for f in all_files if f.endswith(ext)]
        return yaml_files
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: Could not get Git-tracked files.")
        logger.error(f"Git error: {e.stderr}")
    except FileNotFoundError:
        logger.error("Error: Git is not installed or not found in PATH.")
    return []


def run_all_configs(
    configs_dir="configs",
    configs_list: list[str] | None = None,
    ext=".yaml",
    mode: Literal["committed", "all", "list"] = "committed",
    executable="python",
    script="main.py",
    config_arg="-c",
    extra_args=None,
    config_mode: Literal["name", "path"] = "path",
    disable_tqm=False,
):
    logger = BFG["logger"]

    if disable_tqm:
        tqdm_default = os.getenv("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"

    config_files = []
    if mode == "committed":
        config_files = get_committed_configs(configs_dir, ext)
    elif mode == "all":
        config_files = get_configs(configs_dir, ext)
    elif mode == "list":
        config_files = configs_list

    if not config_files:
        logger.info(f"No {ext} files found. Exiting.")
        return

    config_files.sort()
    # config_files=natural_sorter(config_files)
    if config_mode == "name":
        config_files = [os.path.basename(f) for f in config_files]

    logger.info(f"Found {len(config_files)} {ext} files:")
    for f in config_files:
        logger.info(f" - {f}")
    logger.info("-" * 50)

    for config_file in config_files:
        logger.info(f"Starting: {config_file}")

        cmd = [executable, script, config_arg, config_file]
        if extra_args:
            cmd.extend(extra_args)

        logger.info(f"Running command: {cmd}")

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            while process.poll() is None:
                import select

                reads = [process.stdout, process.stderr]
                readable, _, _ = select.select(reads, [], [], 0.1)  # 0.1s timeout

                for readable_pipe in readable:
                    if readable_pipe == process.stdout:
                        output = readable_pipe.readline()
                        if output:
                            logger.info(output.strip())
                    elif readable_pipe == process.stderr:
                        error = readable_pipe.readline()
                        if error:
                            logger.info(f"Stderr: {error.strip()}")

            # Get any remaining output
            stdout, stderr = process.communicate()
            if stdout:
                for line in stdout.splitlines():
                    logger.info(line.strip())
            if stderr:
                logger.error(f"Error: {stderr}")

            if process.returncode != 0:
                logger.error(f"✗ Failed: {config_file}")
            else:
                logger.info(f"✓ Completed: {config_file}")

        except Exception as e:
            logger.error(f"✗ Failed: {config_file}")
            logger.error(f"Error: {str(e)}")

        logger.info("-" * 50)

    if disable_tqm and tqdm_default is not None:
        os.environ["TQDM_DISABLE"] = tqdm_default


def get_mlflow_runs(
    experiment_name: str,
    status: Literal["success", "failed"] | None = "success",
    level: Literal["parent", "child"] | None = "parent",
    filter: str | None = None,
) -> list[entities.Run]:
    """Get runs based on experiment name, status, hierarchy level, and optional filter.

    Args:
        experiment_name (str): Name of the MLflow experiment
        status (str): Run status - "success" for FINISHED runs, "failed" for FAILED runs,
                     None for any status
        level (str): Run hierarchy level - "parent" for parent runs only, "child" for child runs only,
                    None for all runs regardless of hierarchy
        filter (str): Optional MLflow filter string for additional filtering on params, metrics, or tags.
                     Examples: "params.dataset = 'valid'", "metrics.accuracy > 0.8", "tags.model_type = 'bert'"

    Returns:
        List[entities.Run]: List of runs matching the criteria, sorted by start_time DESC
    """
    logger = BFG["logger"]

    # try:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        logger.warning(f"Experiment '{experiment_name}' not found")
        return []

    # Build filter string based on status
    filter_conditions = []
    if status and status.lower() == "success":
        filter_conditions.append("status = 'FINISHED'")
    elif status and status.lower() == "failed":
        filter_conditions.append("status = 'FAILED'")

    # Add custom filter if provided
    if filter:
        filter_conditions.append(filter)

    filter_string = " AND ".join(filter_conditions) if filter_conditions else ""

    # Search runs with combined filter
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        run_view_type=entities.ViewType.ACTIVE_ONLY,
        order_by=["start_time DESC"],  # new to old
    )

    # Apply hierarchy level filtering
    if level and level.lower() == "parent":
        # Filter for parent runs only (no parentRunId tag)
        runs = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]
    elif level and level.lower() == "child":
        # Filter for child runs only (has parentRunId tag)
        runs = [run for run in runs if "mlflow.parentRunId" in run.data.tags]

    if not runs:
        logger.info(
            f"No runs found for experiment '{experiment_name}' with status='{status}', level='{level}', and filter='{filter}'"
        )
        return []

    logger.info(
        f"Found {len(runs)} run(s) for experiment '{experiment_name}' with status='{status}', level='{level}', and filter='{filter}'"
    )
    return runs

    # except Exception as e:
    #     logger.error(f"Error retrieving runs for '{experiment_name}': {str(e)}")
    #     return []


# Helper functions for backward compatibility and convenience
def get_last_successful_run(experiment_name: str) -> Optional[entities.Run]:
    """Find the most recent successful parent run."""
    logger = BFG["logger"]
    runs = get_mlflow_runs(experiment_name, status="success", level="parent")
    if runs:
        logger.info(
            f"Found successful run {runs[0].info.run_id} for experiment '{experiment_name}'"
        )
        return runs[0]
    else:
        logger.warning(f"No successful runs found for experiment '{experiment_name}'")


def get_last_run_failed(experiment_name: str) -> Optional[entities.Run]:
    """Find the most recent run and return it if it's failed.

    Args:
        experiment_name (str): Name of the MLflow experiment

    Returns:
        Optional[entities.Run]: Most recent run if it' has failed or None
    """
    logger = BFG["logger"]

    # try:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        logger.warning(f"Experiment '{experiment_name}' not found")
        return None

    # Filter for only completed runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        run_view_type=entities.ViewType.ACTIVE_ONLY,
        order_by=["start_time DESC"],
    )
    runs = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]

    if not runs:
        logger.info(f"No runs found for experiment '{experiment_name}'")
        return None

    if runs[0].info.status == "FAILED":
        logger.info(
            f"Found failed run {runs[0].info.run_id} for experiment '{experiment_name}'"
        )
        return runs[0]
    # except Exception as e:
    #     logger.error(
    #         f"Error retrieving last run for '{experiment_name}': {str(e)}"
    #     )
    #     return None


def set_tracking_uri(path: str):
    # If path is absolute
    if path.startswith("/"):
        mlflow.set_tracking_uri(f"file:{path}")
    # Otherwise get parent dir of entrypoint script
    else:
        # Resolves to interpreter path on console:
        # main_file = Path(sys.argv[0]).parent.resolve()

        main_file = Path(os.getcwd()).resolve()
        mlflow_path = os.path.join(main_file, path)
        os.makedirs(mlflow_path, exist_ok=True)
        mlflow.set_tracking_uri(f"file:{mlflow_path}")

    return mlflow.get_tracking_uri()


def plot_metric_across_runs(
    metric_key, experiment_name=None, run_ids=None, artifact_file=None
):
    """
    Plots the specified metric across multiple MLflow runs and logs the plot as a figure in MLflow.

    Parameters:
    - metric_key (str): The key of the metric to plot (e.g., 'train_loss', 'val_accuracy').
    - experiment_name (str, optional): The name of the experiment to fetch finished runs from.
    - run_ids (list of str, optional): List of specific run IDs to fetch (only finished runs).
    - artifact_file (str, optional): The file name to save the plot as in the artifact store.
                                     Defaults to "{metric_key}_plot.png".

    Raises:
    - ValueError: If neither experiment_name nor run_ids is provided, or if both are provided,
                  or if the experiment_name is not found.

    Notes:
    - The function must be called within an active MLflow run to log the figure.
    - Only runs with status 'FINISHED' are included.
    - If a run lacks the specified metric, it is skipped with a warning message.
    """

    import matplotlib.pyplot as plt

    logger = BFG["logger"]

    # Ensure only one of experiment_name or run_ids is provided
    if experiment_name is not None and run_ids is not None:
        logger.error("Specify either experiment_name or run_ids, not both")
        return
    if experiment_name is None and run_ids is None:
        logger.error("Either experiment_name or run_ids must be provided")
        return

    client = MlflowClient()

    # Handle case where experiment_name is provided
    if experiment_name is not None:
        # Get experiment by name
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"Experiment '{experiment_name}' not found")
            return
        experiment_id = experiment.experiment_id

        # Fetch only finished runs from the experiment
        df = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string="status = 'FINISHED'",
            run_view_type=entities.ViewType.ACTIVE_ONLY,
            order_by=["start_time ASC"],
        )
        if "tags.mlflow.parentRunId" in df.columns:
            df = df[~df["tags.mlflow.parentRunId"].notna()]
    # Handle case where run_ids are provided
    else:
        # Construct filter string to fetch specific finished runs
        filter_string = (
            "status = 'FINISHED' AND ("
            + " OR ".join([f"attribute.run_id = '{run_id}'" for run_id in run_ids])
            + ")"
        )
        df = mlflow.search_runs(
            filter_string=filter_string,
            run_view_type=entities.ViewType.ACTIVE_ONLY,
            order_by=["start_time ASC"],
        )

    if len(df) == 0:
        logger.warning(f"No runs found.")
        return

    # Create plot
    fig, ax = plt.subplots()

    # Iterate over runs to fetch and plot metric data
    for _, row in df.iterrows():
        run_id = row["run_id"]
        # Use run name if available, otherwise fall back to run_id
        run_name = row.get("tags.mlflow.runName", run_id)
        try:
            metrics = client.get_metric_history(run_id, metric_key)
            if metrics:
                # Sort metrics by step to ensure correct order
                metrics = sorted(metrics, key=lambda m: m.step)
                steps = [m.step for m in metrics]
                values = [m.value for m in metrics]
                ax.plot(steps, values, label=run_name)
        except Exception as e:
            logger.warning(f"Error fetching metric for run {run_id}: {e}")

    # Configure and log the plot if there is data to display
    if ax.get_lines():
        ax.set_title(f"{metric_key} across runs")
        ax.set_xlabel("Step")
        ax.set_ylabel(metric_key)
        ax.legend(fontsize=14)
        if artifact_file is None:
            artifact_file = f"{metric_key}_plot.png"
        mlflow.log_figure(fig, artifact_file)
    else:
        logger.warning(
            f"No runs have the metric {metric_key} or no finished runs found"
        )

    # Clean up
    plt.close(fig)
    cleanup_plots()


def process_experiment_metrics(
    storage_path: Union[str, Path],
    select_metrics: list[str],
    sort_metric: str,
    legend_names: list[str],
    y_metric: str,
    metrics_to_mean: list[str] = [],
    mean_metric_name: str = "mean",
    aggregate_all_runs: bool = False,
    run_mode: Literal["parent", "children", "both"] = "both",
    filter: Optional[str] = None,
    plot_experiments: Optional[list[str]] = None,
    plot_mode: Literal["step", "epoch"] = "epoch",
    output_file: str = None,
    plot_title: str = None,
    figsize: tuple = (12, 8),
    dpi: int = 300,
    subsample_interval: int = 100,
    marker_size: int = 6,
) -> pd.DataFrame:
    """
    Process MLflow experiments to calculate mean metrics and sort results, with optional plotting.

    Args:
        storage_path: Path to MLflow storage folder
        select_metrics: List of metric names to collect
        metrics_to_mean: List of metric names to calculate mean (subset of select_metrics)
        mean_metric_name: Name for the new mean metric
        sort_metric: Metric to sort results by
        aggregate_all_runs: If True, get all successful runs; if False, get last run
        run_mode: Filter runs by 'parent', 'children', or 'both'
        filter: Optional MLflow filter string (e.g., "params.dataset = 'lip'")

        # New plotting parameters
        plot_experiments: List of experiment names to include in plot (if None, uses all found experiments)
        plot_mode: X-axis mode - "step" or "epoch"
        save_plot: If True, saves plot as artifact
        plot_title: Custom title for the plot
        figsize: Figure size in inches (width, height)
        dpi: DPI for high-quality output
        subsample_interval: Subsample 1 point every N steps to reduce density
        marker_size: Size of markers on the plot

    Returns:
        pandas.DataFrame: Processed metrics with mean and sorted results
    """
    if run_mode not in ["parent", "children", "both"]:
        raise ValueError(
            f"Invalid run_mode: {run_mode}. Must be 'parent', 'children', or 'both'."
        )

    # Set MLflow tracking URI
    set_tracking_uri(storage_path)
    client = MlflowClient()

    # Get all active experiments
    experiments = client.search_experiments(view_type=entities.ViewType.ACTIVE_ONLY)
    if not experiments:
        print("No active experiments found")
        return pd.DataFrame()

    results = []
    experiment_names = []  # Track experiment names for plotting

    for exp in experiments:
        # Get runs based on mode and filter
        filter_string = "status = 'FINISHED'"
        if filter:
            filter_string = f"{filter_string} and {filter}"

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=filter_string,
            run_view_type=entities.ViewType.ACTIVE_ONLY,
            order_by=["start_time DESC"],
        )

        if run_mode == "children":
            runs = [
                run
                for run in runs
                if "mlflow.parentRunId" in run.data.tags
                and run.data.tags["mlflow.parentRunId"]
            ]
        elif run_mode == "parent":
            runs = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]

        if aggregate_all_runs is False:
            runs = runs[:1]

        if not runs:
            print(f"No runs found for experiment {exp.name} with run_mode {run_mode}")
            continue

        experiment_names.append(exp.name)  # Track experiment name

        for run in runs:
            run_metrics = {}
            # Collect specified metrics
            for metric in select_metrics:
                run_metrics[metric] = run.data.metrics.get(metric, np.nan)
                continue
                try:
                    metric_history = client.get_metric_history(run.info.run_id, metric)
                    if metric_history:
                        run_metrics[metric] = metric_history[-1].value
                except:
                    run_metrics[metric] = np.nan
                    print(
                        f"WARNING: metric {metric} missing "
                        f"for run with id {run.info.run_id} and name {run.info.run_name} "
                        f"of experiment with id {exp.experiment_id} name {exp.name}"
                    )

            # Calculate mean metric if all specified metrics exist
            if all(metric in run_metrics for metric in metrics_to_mean):
                mean_value = np.mean([run_metrics[m] for m in metrics_to_mean])
                run_metrics[mean_metric_name] = mean_value
            else:
                missing_metrics = [m for m in metrics_to_mean if m not in run_metrics]
                print(
                    f"WARNING: metric(s) {missing_metrics} missing in mean metrics "
                    f"for run with id {run.info.run_id} and name {run.info.run_name} "
                    f"of experiment with id {exp.experiment_id} name {exp.name}"
                )

            # Add experiment and run info
            run_metrics["experiment_name"] = exp.name
            run_metrics["run_id"] = run.info.run_id
            results.append(run_metrics)

    # Create DataFrame and sort
    if not results:
        print("No valid runs found with specified metrics")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    if sort_metric and sort_metric in df.columns:
        df = df.sort_values(by=sort_metric, ascending=False)
    else:
        print(f"WARNING: sort metric {sort_metric} not valid.")

    # Generate plots if requested
    plot_experiments_list = plot_experiments if plot_experiments else experiment_names

    # Create multi-metric plot
    if output_file:
        plot_multiple_metrics_across_experiments(
            experiment_names=plot_experiments_list,
            metric_keys=select_metrics,
            mode=plot_mode,
            artifact_file=output_file,
            title=plot_title,
            figsize=figsize,
            dpi=dpi,
            subsample_interval=subsample_interval,
            marker_size=marker_size,
            legend_names=legend_names,
            y_metric=y_metric,
        )

    return df


def plot_multiple_metrics_across_experiments(
    experiment_names: list[str],
    metric_keys: list[str],
    y_metric: str,
    legend_names: list[str] = None,
    mode: Literal["step", "epoch"] = "epoch",
    artifact_file: str = None,
    title: str = None,
    figsize: tuple = (12, 8),
    dpi: int = 300,
    subsample_interval: int = 100,
    marker_size: int = 6,
) -> None:
    """
    Plots multiple metrics across multiple MLflow experiments on the same plot.

    Parameters:
    - experiment_names (list[str]): List of experiment names to compare
    - metric_keys (list[str]): List of metrics to plot (e.g., ['f1-binary', 'f1-weighted'])
    - mode (str): X-axis mode - "step" or "epoch"
    - legend_names (list[str], optional): Custom names for legend
    - artifact_file (str, optional): File name to save the plot
    - title (str, optional): Custom title for the plot
    - figsize (tuple): Figure size in inches (width, height)
    - dpi (int): DPI for high-quality output
    - subsample_interval (int): Subsample 1 point every N steps to reduce density
    - marker_size (int): Size of markers on the plot
    """

    # Validate legend_names if provided
    if legend_names is not None:
        if len(legend_names) != len(experiment_names) * len(metric_keys):
            raise ValueError(
                f"legend_names length ({len(legend_names)}) must match experiment_names length ({len(experiment_names)})"
            )
    else:
        legend_names = experiment_names

    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    client = MlflowClient()

    # Print-friendly line styles and colors
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    legend_elements = []
    plot_data_found = False

    # Create style combinations for each metric-experiment pair
    style_combinations = []
    for i, metric in enumerate(metric_keys):
        legend_name = legend_names[i]  # Get corresponding legend name
        for j, exp_name in enumerate(experiment_names):
            style_combinations.append(
                {
                    "metric": metric,
                    "experiment": exp_name,
                    "color": colors[i % len(colors)],
                    "linestyle": line_styles[j % len(line_styles)],
                    "marker": markers[(i * len(experiment_names) + j) % len(markers)],
                    "legend_name": legend_name,
                }
            )

    for combo in style_combinations:
        metric_key = combo["metric"]
        exp_name = combo["experiment"]
        legend_name = combo["legend_name"]

        # Get experiment by name
        try:
            experiment = client.get_experiment_by_name(exp_name)
            if not experiment:
                print(f"Experiment '{exp_name}' not found")
                continue
        except Exception as e:
            print(f"Error getting experiment '{exp_name}': {e}")
            continue

        # Get all successful runs
        try:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="status = 'FINISHED'",
                run_view_type=entities.ViewType.ACTIVE_ONLY,
                order_by=["start_time DESC"],
            )

            # Filter for parent runs only
            runs = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]

            if not runs:
                print(f"No successful runs found for experiment '{exp_name}'")
                continue

            # Reverse to get oldest to newest
            runs = runs[::-1]
        except Exception as e:
            print(f"Error getting runs for experiment '{exp_name}': {e}")
            continue

        # Collect data for this metric-experiment combination
        all_x_values = []
        all_y_values = []
        runs_plotted = 0

        for run in runs:
            try:
                # Get metric history
                metrics = client.get_metric_history(run.info.run_id, metric_key)
                if not metrics:
                    continue

                # Sort metrics by step
                metrics = sorted(metrics, key=lambda m: m.step)

                # Extract x and y values
                x_values = [m.step for m in metrics]
                y_values = [m.value for m in metrics]

                # Subsample if needed
                if len(x_values) > subsample_interval:
                    indices = [0]  # Keep first point
                    for idx in range(
                        subsample_interval, len(x_values) - 1, subsample_interval
                    ):
                        indices.append(idx)
                    if len(x_values) - 1 not in indices:
                        indices.append(len(x_values) - 1)  # Keep last point

                    x_values = [x_values[idx] for idx in indices]
                    y_values = [y_values[idx] for idx in indices]

                all_x_values.extend(x_values)
                all_y_values.extend(y_values)
                runs_plotted += 1

            except Exception as e:
                print(f"Error processing run {run.info.run_id}: {e}")
                continue

        # Plot if we have data
        if all_x_values and all_y_values:
            # Convert to epochs if requested
            if mode.lower() == "epoch":
                # Simple conversion: assume steps are roughly evenly distributed across epochs
                max_step = max(all_x_values)
                all_x_values = [x / max_step * runs_plotted for x in all_x_values]
                x_label = "Epoch"
            else:
                x_label = "Step"

            # Sort by x-values
            combined_data = list(zip(all_x_values, all_y_values))
            combined_data.sort(key=lambda x: x[0])
            sorted_x, sorted_y = zip(*combined_data)

            # Plot the line
            ax.plot(
                sorted_x,
                sorted_y,
                linestyle=combo["linestyle"],
                marker=combo["marker"],
                color=combo["color"],
                markersize=marker_size,
                linewidth=2.0,
                alpha=0.8,
                label=f"{legend_name.title()} ({runs_plotted} runs)",
                markerfacecolor="white",
                markeredgecolor=combo["color"],
                markeredgewidth=6.0,
            )

            # Add to legend
            legend_elements.append(
                mlines.Line2D(
                    [],
                    [],
                    color=combo["color"],
                    linestyle=combo["linestyle"],
                    marker=combo["marker"],
                    markersize=marker_size,
                    markerfacecolor="white",
                    markeredgecolor=combo["color"],
                    linewidth=2.0,
                    markeredgewidth=6.0,  # marker legend size
                    # label=f"{metric_key} ({exp_name})",
                    label=f"{legend_name.title()} ({runs_plotted} runs)",
                )
            )

            plot_data_found = True

    if not plot_data_found:
        print("No data found for any metric-experiment combination")
        plt.close(fig)
        return

    # Configure plot
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_metric.replace("_", " ").title(), fontsize=12, fontweight="bold")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    else:
        metric_names = ", ".join(metric_keys)
        ax.set_title(
            f"Metrics Comparison: {metric_names}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

    # Enhanced legend
    ax.legend(
        handles=legend_elements,
        loc="best",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=14,
        framealpha=0.9,
    )

    # Grid for better readability
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Improve layout
    plt.tight_layout()

    # Save or show plot
    if artifact_file:
        plt.savefig(artifact_file, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved as: {artifact_file}")
    else:
        plt.show()

    plt.close(fig)


def visualize_metrics(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    color_metric: Optional[str] = None,
    title: str = "Experiment Metrics Visualization",
    theme: str = "plotly_dark",
    output_file: Optional[str] = None,
) -> None:
    """
    Visualize metrics using Plotly scatter plot.

    Args:
        df: DataFrame from process_experiment_metrics
        x_metric: Metric for x-axis
        y_metric: Metric for y-axis
        color_metric: Metric for color scale (optional)
        title: Plot title
        theme: Plotly theme (e.g., 'plotly', 'plotly_dark', 'plotly_white')
        output_file: Path to save the plot (supports PNG, JPG, SVG, PDF, HTML formats).
                    If None, displays plot interactively.
    """
    import plotly.express as px
    import plotly.io as pio

    if df.empty:
        print("No data to visualize")
        return

    if x_metric not in df.columns or y_metric not in df.columns:
        print(
            f"Invalid metrics: x_metric={x_metric}, y_metric={y_metric} not in DataFrame"
        )
        return

    # Set Plotly theme
    pio.templates.default = theme

    fig = px.scatter(
        df,
        x=x_metric,
        y=y_metric,
        color=color_metric,
        hover_data=df.columns,
        title=title,
        labels={
            x_metric: x_metric.replace("_", " ").title(),
            y_metric: y_metric.replace("_", " ").title(),
        },
    )

    fig.update_traces(marker=dict(size=12))
    fig.update_layout(showlegend=True)

    # Save to file or show interactively
    if output_file:
        # Determine file format from extension
        file_ext = output_file.lower().split(".")[-1]

        if file_ext == "html":
            # Save as interactive HTML
            fig.write_html(output_file)
            print(f"Interactive plot saved to {output_file}")
        elif file_ext in ["png", "jpg", "jpeg", "svg", "pdf"]:
            # Save as static image (requires kaleido package)
            try:
                fig.write_image(output_file)
                print(f"Static plot saved to {output_file}")
            except Exception as e:
                print(f"Error saving static image: {e}")
                print(
                    "Make sure you have the 'kaleido' package installed for static image export:"
                )
                print("pip install kaleido")
                # Fallback to HTML
                html_file = output_file.rsplit(".", 1)[0] + ".html"
                fig.write_html(html_file)
                print(f"Saved as HTML instead: {html_file}")
        else:
            print(f"Unsupported file format: {file_ext}")
            print("Supported formats: PNG, JPG, SVG, PDF, HTML")
            print("Displaying plot interactively instead.")
            fig.show()
    else:
        # Display interactively
        fig.show()


def cleanup_plots():
    """Clean up all matplotlib figures safely during training"""
    import gc

    import matplotlib.pyplot as plt

    plt.clf()
    plt.cla()

    # Alternative: remove from figure manager
    from matplotlib._pylab_helpers import Gcf

    Gcf.destroy_all()

    # Get all figure numbers and delete them
    # fig_nums = plt.get_fignums()
    # for num in fig_nums:
    #     fig = plt.figure(num)
    #     fig.clear()
    #     del fig

    gc.collect()


def plot_metric_across_experiments(
    experiment_names: list[str],
    metric_key: str,
    y_metric: Optional[str] = None,
    mode: Literal["step", "epoch"] = "epoch",
    legend_names: list[str] = None,
    artifact_file: str = None,
    title: str = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
    subsample_interval: int = 100,  # NEW: configurable subsampling interval
    marker_size: int = 6,  # NEW: configurable marker size
) -> None:
    """
    Plots the specified metric across multiple MLflow experiments with print-friendly styling.

    Parameters:
    - experiment_names (list[str]): List of experiment names to compare
    - metric_key (str): The key of the metric to plot (e.g., 'train_loss', 'val_accuracy')
    - mode (str): X-axis mode - "step" or "epoch" (default: "step")
    - legend_names (list[str], optional): Custom names for legend in same order as experiment_names.
                                        If None, uses experiment_names directly.
    - artifact_file (str, optional): The file name to save the plot.
                                   Defaults to "{metric_key}_across_experiments.png"
    - title (str, optional): Custom title for the plot
    - figsize (tuple): Figure size in inches (width, height)
    - dpi (int): DPI for high-quality output suitable for printing
    - subsample_interval (int): Subsample 1 point every N steps to reduce density (default: 100)
    - marker_size (int): Size of markers on the plot (default: 6)

    Notes:
    - The function must be called within an active MLflow run to log the figure
    - Only runs with status 'FINISHED' are included
    - Uses print-friendly line styles (different patterns, markers) for B&W printing
    - Orders runs from oldest to newest within each experiment
    - If legend_names is provided, it must have the same length as experiment_names
    - Subsamples data points to reduce visual density while maintaining trend visibility
    """
    from itertools import cycle

    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    logger = BFG["logger"]
    client = MlflowClient()

    if y_metric is None:
        y_metric = metric_key

    # Validate legend_names if provided
    if legend_names is not None:
        if len(legend_names) != len(experiment_names):
            raise ValueError(
                f"legend_names length ({len(legend_names)}) must match experiment_names length ({len(experiment_names)})"
            )
    else:
        legend_names = experiment_names

    # Print-friendly line styles: different patterns and markers
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Create cycling iterators for styles
    style_cycle = cycle(line_styles)
    marker_cycle = cycle(markers)
    color_cycle = cycle(colors)

    # Create plot with high DPI for print quality
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    legend_elements = []
    experiments_found = 0

    for i, exp_name in enumerate(experiment_names):
        legend_name = legend_names[i]  # Get corresponding legend name
        # Get experiment by name
        experiment = client.get_experiment_by_name(exp_name)
        if not experiment:
            logger.warning(f"Experiment '{exp_name}' not found")
            continue

        # Get all successful runs, ordered from oldest to newest
        runs = get_mlflow_runs(exp_name, status="success", level="parent")
        if not runs:
            logger.warning(f"No successful runs found for experiment '{exp_name}'")
            continue

        # Reverse to get oldest to newest (get_mlflow_runs returns newest first)
        runs = runs[::-1]

        # Get current style elements
        current_style = next(style_cycle)
        current_marker = next(marker_cycle)
        current_color = next(color_cycle)

        runs_plotted = 0

        if mode.lower() == "epoch":
            # For epoch mode, collect all points from all runs and plot together
            all_x_values = []
            all_y_values = []

            for run_idx, run in enumerate(runs):
                try:
                    # Get metric history
                    metrics = client.get_metric_history(run.info.run_id, metric_key)
                    if not metrics:
                        logger.warning(
                            f"No metric '{metric_key}' found for run {run.info.run_id}"
                        )
                        continue

                    # Sort metrics by step/timestamp to ensure correct order
                    metrics = sorted(metrics, key=lambda m: m.step)

                    # Extract x and y values
                    x_values = [m.step for m in metrics]
                    y_values = [m.value for m in metrics]

                    # NEW: Subsample data points to reduce density
                    if len(x_values) > subsample_interval:
                        # Keep first and last points, then subsample in between
                        indices = [0]  # Always keep first point

                        # Add subsampled points
                        for idx in range(
                            subsample_interval, len(x_values) - 1, subsample_interval
                        ):
                            indices.append(idx)

                        # Always keep last point
                        if len(x_values) - 1 not in indices:
                            indices.append(len(x_values) - 1)

                        x_values = [x_values[idx] for idx in indices]
                        y_values = [y_values[idx] for idx in indices]

                        logger.info(
                            f"Subsampled {len(metrics)} points to {len(x_values)} points for run {run.info.run_id}"
                        )

                    # Collect all points
                    all_x_values.extend(x_values)
                    all_y_values.extend(y_values)
                    runs_plotted += 1

                except Exception as e:
                    logger.warning(
                        f"Error processing run {run.info.run_id} from experiment '{exp_name}': {e}"
                    )
                    continue

            # Properly convert Step to Epoch
            all_x_values = list(np.arange(0, len(runs), len(runs) / len(all_x_values)))

            # Plot all points together for epoch mode
            if all_x_values and all_y_values:
                # Sort by x-values to ensure proper line connections
                combined_data = list(zip(all_x_values, all_y_values))
                combined_data.sort(key=lambda x: x[0])
                sorted_x, sorted_y = zip(*combined_data)

                print(
                    f"Epoch mode combined x_values: {sorted_x[:10]}..."
                )  # Debug print

                ax.plot(
                    sorted_x,
                    sorted_y,
                    linestyle=current_style,
                    marker=current_marker,
                    color=current_color,
                    markersize=marker_size,
                    linewidth=2.0,
                    alpha=0.8,
                    label=f"{legend_name.title()} ({runs_plotted} runs)",
                    markerfacecolor="white",
                    markeredgecolor=current_color,
                    markeredgewidth=6.0,
                )

            x_label = "Epoch"  # Actually showing steps, but representing epochs

        else:
            # Step mode - plot each run separately (original behavior)
            for run_idx, run in enumerate(runs):
                try:
                    # Get metric history
                    metrics = client.get_metric_history(run.info.run_id, metric_key)
                    if not metrics:
                        logger.warning(
                            f"No metric '{metric_key}' found for run {run.info.run_id}"
                        )
                        continue

                    # Sort metrics by step/timestamp to ensure correct order
                    metrics = sorted(metrics, key=lambda m: m.step)

                    # Extract x and y values
                    x_values = [m.step for m in metrics]
                    y_values = [m.value for m in metrics]

                    # NEW: Subsample data points to reduce density
                    if len(x_values) > subsample_interval:
                        # Keep first and last points, then subsample in between
                        indices = [0]  # Always keep first point

                        # Add subsampled points
                        for idx in range(
                            subsample_interval, len(x_values) - 1, subsample_interval
                        ):
                            indices.append(idx)

                        # Always keep last point
                        if len(x_values) - 1 not in indices:
                            indices.append(len(x_values) - 1)

                        x_values = [x_values[idx] for idx in indices]
                        y_values = [y_values[idx] for idx in indices]

                        logger.info(
                            f"Subsampled {len(metrics)} points to {len(x_values)} points for run {run.info.run_id}"
                        )

                    print(f"Step mode x_values: {x_values[:10]}...")  # Debug print

                    # Plot each run separately for step mode
                    ax.plot(
                        x_values,
                        y_values,
                        linestyle=current_style,
                        marker=current_marker,
                        color=current_color,
                        markersize=marker_size,
                        linewidth=2.0,
                        alpha=0.8,
                        label=f"{legend_name} (Run {run_idx+1})",
                        markerfacecolor="white",
                        markeredgecolor=current_color,
                        markeredgewidth=6.0,
                    )

                    runs_plotted += 1

                except Exception as e:
                    logger.warning(
                        f"Error processing run {run.info.run_id} from experiment '{exp_name}': {e}"
                    )
                    continue

            x_label = "Step"

        if runs_plotted > 0:
            experiments_found += 1
            # Add experiment to legend (using the current style as representative)
            legend_elements.append(
                mlines.Line2D(
                    [],
                    [],
                    color=current_color,
                    linestyle=current_style,
                    marker=current_marker,
                    markersize=marker_size,
                    markerfacecolor="white",
                    markeredgecolor=current_color,
                    linewidth=2.0,
                    markeredgewidth=6.0,
                    label=f"{legend_name.title()} ({runs_plotted} runs)",
                )
            )

    if experiments_found == 0:
        logger.warning("No data found for any of the specified experiments")
        plt.close(fig)
        return

    # Configure plot for print quality
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_metric.replace("_", " ").title(), fontsize=12, fontweight="bold")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    # else:
    #     ax.set_title(
    #         f'{metric_key.replace("_", " ").title()} Across Experiments',
    #         fontsize=14,
    #         fontweight="bold",
    #         pad=20,
    #     )

    # Enhanced legend for print clarity
    ax.legend(
        handles=legend_elements,
        loc="best",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=14,
        framealpha=0.9,
    )

    # Grid for better readability in print
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Improve layout
    plt.tight_layout()

    # Set artifact filename
    if artifact_file is not None:
        plt.savefig(artifact_file, dpi=dpi, bbox_inches="tight")
        logger.info(f"Plot saved as artifact: {artifact_file}")
    else:
        plt.show()

    # Clean up
    plt.close(fig)
    cleanup_plots()


def export_mlflow_data_with_flattening(
    storage_path: str, output_file: str = "mlflow_all_experiments_runs.csv"
):
    """
    Export MLflow data with JSON flattening - adapted from original code.

    Args:
        storage_path: Path to MLflow storage folder
        output_file: Output CSV filename
    """
    # Set MLflow tracking URI
    if storage_path.startswith("/"):
        import mlflow

        mlflow.set_tracking_uri(f"file:{storage_path}")
    else:
        main_file = Path(os.getcwd()).resolve()
        mlflow_path = os.path.join(main_file, storage_path)
        os.makedirs(mlflow_path, exist_ok=True)
        import mlflow

        mlflow.set_tracking_uri(f"file:{mlflow_path}")

    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()

    all_runs_data = []

    for experiment in experiments:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if not runs.empty:
            # Add experiment name to each run
            runs["experiment_name"] = experiment.name
            all_runs_data.append(runs)

    if all_runs_data:
        combined_runs = pd.concat(all_runs_data, ignore_index=True)

        # Flatten JSON columns
        combined_runs_flattened = flatten_json_columns(combined_runs)

        combined_runs_flattened.to_csv(output_file, index=False)
        print(
            f"Data exported to {output_file} with {len(combined_runs_flattened)} runs"
        )
        return combined_runs_flattened
    else:
        print("No runs found")
        return pd.DataFrame()


def find_best_metrics(
    storage_path: str,
    experiment_names: list[str],
    metrics: list[str],
    min_or_max: list[str],
    run_mode: Literal["parent", "children", "both"] = "both",
    filter_string: str = None,
    aggregation_mode: Literal["all_runs", "best_run"] = "all_runs",
) -> pd.DataFrame:
    """
    Find the best values for specified metrics across experiments.
    Uses only pandas, json, and python - no MLflow calls.

    Args:
        storage_path: Path to MLflow storage folder or CSV file
        experiment_names: List of experiment names to search (supports regex patterns)
        metrics: List of metric names to find best values for (supports regex patterns)
        min_or_max: List of 'min' or 'max' corresponding to each metric
        run_mode: Filter runs by 'parent', 'children', or 'both'
        filter_string: Optional pandas query filter string
        aggregation_mode: 'all_runs' for current behavior, 'best_run' to keep only best run per metric

    Returns:
        pandas.DataFrame: Results with best values, steps, epochs, and run info
    """

    # Input validation
    if len(metrics) != len(min_or_max):
        raise ValueError("metrics and min_or_max must have the same length")

    for mode in min_or_max:
        if mode not in ["min", "max"]:
            raise ValueError("min_or_max values must be 'min' or 'max'")

    # Load data
    if storage_path.endswith(".csv"):
        df = pd.read_csv(storage_path)
    else:
        # Export data first if storage_path is MLflow folder
        df = export_mlflow_data_with_flattening(storage_path)

    if df.empty:
        return pd.DataFrame()

    # Flatten JSON columns if not already done
    df = flatten_json_columns(df)

    # Filter by experiment names (supports regex)
    if experiment_names:
        exp_pattern = "|".join(experiment_names)
        if "experiment_name" in df.columns:
            df = df[
                df["experiment_name"].str.contains(exp_pattern, regex=True, na=False)
            ]
        else:
            print("Warning: 'experiment_name' column not found. Available columns:")
            print(df.columns.tolist())
            return pd.DataFrame()

    # Filter by run mode
    if run_mode == "parent":
        # Parent runs don't have parentRunId
        if "tags.mlflow.parentRunId" in df.columns:
            df = df[df["tags.mlflow.parentRunId"].isna()]
    elif run_mode == "children":
        # Child runs have parentRunId
        if "tags.mlflow.parentRunId" in df.columns:
            df = df[df["tags.mlflow.parentRunId"].notna()]

    # Apply custom filter
    if filter_string:
        try:
            df = df.query(filter_string)
        except Exception as e:
            print(f"Warning: Filter string '{filter_string}' failed: {e}")

    # Filter successful runs
    if "status" in df.columns:
        df = df[df["status"] == "FINISHED"]

    if df.empty:
        return pd.DataFrame()

    # Find matching metric columns (supports regex)
    available_columns = df.columns.tolist()
    results = []

    for metric_pattern, optimization in zip(metrics, min_or_max):
        # Find columns matching the metric pattern
        matching_cols = [
            col for col in available_columns if re.search(metric_pattern, col)
        ]

        if not matching_cols:
            print(f"Warning: No columns found matching pattern '{metric_pattern}'")
            continue

        for metric_col in matching_cols:
            # Skip if metric column doesn't exist or is all NaN
            if metric_col not in df.columns or df[metric_col].isna().all():
                continue

            df_metric = df[df[metric_col].notna()].copy()

            if df_metric.empty:
                continue

            if aggregation_mode == "all_runs":
                # Return all runs sorted by metric
                ascending = optimization == "min"
                df_sorted = df_metric.sort_values(by=metric_col, ascending=ascending)

                # Add metric info
                df_sorted = df_sorted.copy()
                df_sorted["target_metric"] = metric_col
                df_sorted["optimization"] = optimization
                df_sorted["metric_value"] = df_sorted[metric_col]

                results.append(df_sorted)

            elif aggregation_mode == "best_run":
                # Find best run for this metric
                if optimization == "min":
                    best_idx = df_metric[metric_col].idxmin()
                else:
                    best_idx = df_metric[metric_col].idxmax()

                best_run = df_metric.loc[[best_idx]].copy()
                best_run["target_metric"] = metric_col
                best_run["optimization"] = optimization
                best_run["metric_value"] = best_run[metric_col]

                results.append(best_run)

    if not results:
        return pd.DataFrame()

    # Combine results
    final_df = pd.concat(results, ignore_index=True)

    # Add useful columns if they exist
    useful_cols = [
        "experiment_name",
        "run_id",
        "status",
        "start_time",
        "end_time",
        "tags.mlflow.runName",
        "tags.mlflow.parentRunId",
        "target_metric",
        "optimization",
        "metric_value",
    ]

    # Keep only existing columns
    existing_useful_cols = [col for col in useful_cols if col in final_df.columns]

    # Keep metric columns and useful columns
    metric_columns = [col for col in final_df.columns if col.startswith("metrics.")]
    param_columns = [col for col in final_df.columns if col.startswith("params.")]

    columns_to_keep = existing_useful_cols + metric_columns + param_columns
    columns_to_keep = list(
        dict.fromkeys(columns_to_keep)
    )  # Remove duplicates while preserving order

    # Filter columns that actually exist in the dataframe
    columns_to_keep = [col for col in columns_to_keep if col in final_df.columns]

    final_df = final_df[columns_to_keep]

    return final_df
