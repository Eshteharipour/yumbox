import os
import subprocess
import sys
import tempfile
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Literal, Optional, Union

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from omegaconf import DictConfig, OmegaConf

from yumbox.cache import BFG

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
) -> list[mlflow.entities.Run]:
    """Get runs based on experiment name, status, and hierarchy level.

    Args:
        experiment_name (str): Name of the MLflow experiment
        status (str): Run status - "success" for FINISHED runs, "failed" for FAILED runs,
                     None for any status
        level (str): Run hierarchy level - "parent" for parent runs only, "child" for child runs only,
                    None for all runs regardless of hierarchy

    Returns:
        List[mlflow.entities.Run]: List of runs matching the criteria, sorted by start_time DESC
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

    filter_string = " AND ".join(filter_conditions) if filter_conditions else ""

    # Search runs with status filter
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
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
            f"No runs found for experiment '{experiment_name}' with status='{status}' and level='{level}'"
        )
        return []

    logger.info(
        f"Found {len(runs)} run(s) for experiment '{experiment_name}' with status='{status}' and level='{level}'"
    )
    return runs

    # except Exception as e:
    #     logger.error(f"Error retrieving runs for '{experiment_name}': {str(e)}")
    #     return []


# Helper functions for backward compatibility and convenience
def get_last_successful_run(experiment_name: str) -> Optional[mlflow.entities.Run]:
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


def get_last_run_failed(experiment_name: str) -> Optional[mlflow.entities.Run]:
    """Find the most recent run and return it if it's failed.

    Args:
        experiment_name (str): Name of the MLflow experiment

    Returns:
        Optional[mlflow.entities.Run]: Most recent run if it' has failed or None
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
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
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
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
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
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
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
        ax.legend()
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
    metrics_to_mean: list[str],
    mean_metric_name: str,
    sort_metric: str,
    aggregate_all_runs: bool = False,
    run_mode: Literal["parent", "children", "both"] = "both",
    filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Process MLflow experiments to calculate mean metrics and sort results.

    Args:
        storage_path: Path to MLflow storage folder
        select_metrics: List of metric names to collect
        metrics_to_mean: List of metric names to calculate mean (subset of select_metrics)
        mean_metric_name: Name for the new mean metric
        sort_metric: Metric to sort results by
        aggregate_all_runs: If True, get all successful runs; if False, get last run
        run_mode: Filter runs by 'parent', 'children', or 'both'
        filter: Optional MLflow filter string (e.g., "params.dataset = 'lip'")

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
    experiments = client.search_experiments(
        view_type=mlflow.entities.ViewType.ACTIVE_ONLY
    )
    if not experiments:
        print("No active experiments found")
        return pd.DataFrame()

    results = []

    for exp in experiments:
        # Get runs based on mode and filter
        filter_string = "status = 'FINISHED'"
        if filter:
            filter_string = f"{filter_string} and {filter}"

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=filter_string,
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
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
    if sort_metric in df.columns:
        df = df.sort_values(by=sort_metric, ascending=False)
    else:
        print(f"WARNING: sort metric {sort_metric} not valid.")

    return df


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
    mode: Literal["step", "epoch"] = "step",
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
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"{legend_name} ({runs_plotted} runs)",
                    markerfacecolor="white",
                    markeredgecolor=current_color,
                    markeredgewidth=1.5,
                )

            x_label = "Step"  # Actually showing steps, but representing epochs

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
                        linewidth=1.5,
                        alpha=0.8,
                        label=f"{legend_name} (Run {run_idx+1})",
                        markerfacecolor="white",
                        markeredgecolor=current_color,
                        markeredgewidth=1.5,
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
                    markeredgewidth=1.5,
                    label=f"{legend_name} ({runs_plotted} runs)",
                )
            )

    if experiments_found == 0:
        logger.warning("No data found for any of the specified experiments")
        plt.close(fig)
        return

    # Configure plot for print quality
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_key.replace("_", " ").title(), fontsize=12, fontweight="bold")

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
        fontsize=10,
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


def find_best_metrics(
    storage_path: str,
    experiment_names: list[str],
    metrics: list[str],
    min_or_max: list[str],
    run_mode: Literal["parent", "children", "both"] = "both",
    filter_string: str = None,
) -> pd.DataFrame:
    """
    Find the best values for specified metrics across experiments.

    Args:
        storage_path: Path to MLflow storage folder
        experiment_names: List of experiment names to search
        metrics: List of metric names to find best values for
        min_or_max: List of 'min' or 'max' corresponding to each metric
        run_mode: Filter runs by 'parent', 'children', or 'both'
        filter_string: Optional MLflow filter string

    Returns:
        pandas.DataFrame: Results with best values, steps, epochs, and run info
    """
    from mlflow import MlflowClient

    from yumbox.mlflow import get_mlflow_runs, set_tracking_uri

    if len(metrics) != len(min_or_max):
        raise ValueError("metrics and min_or_max lists must have the same length")

    if not all(direction in ["min", "max"] for direction in min_or_max):
        raise ValueError("min_or_max values must be either 'min' or 'max'")

    # Set MLflow tracking URI
    set_tracking_uri(storage_path)
    client = MlflowClient()

    results = []

    for exp_name in experiment_names:
        print(f"Processing experiment: {exp_name}")

        # Get runs for this experiment
        runs = get_mlflow_runs(exp_name, status="success", level=None)

        if not runs:
            print(f"No successful runs found for experiment '{exp_name}'")
            continue

        # Apply run mode filtering
        if run_mode == "children":
            runs = [run for run in runs if "mlflow.parentRunId" in run.data.tags]
        elif run_mode == "parent":
            runs = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]

        if not runs:
            print(
                f"No runs found for experiment '{exp_name}' with run_mode '{run_mode}'"
            )
            continue

        # Process each metric
        for metric_name, direction in zip(metrics, min_or_max):
            best_value = None
            best_step = None
            best_epoch = None
            best_run_id = None
            best_run_name = None

            print(f"  Finding best {direction} for metric: {metric_name}")

            for run in runs:
                try:
                    # Get metric history for this run
                    metric_history = client.get_metric_history(
                        run.info.run_id, metric_name
                    )

                    if not metric_history:
                        continue

                    # Find best value in this run's history
                    if direction == "min":
                        run_best_metric = min(metric_history, key=lambda x: x.value)
                    else:  # max
                        run_best_metric = max(metric_history, key=lambda x: x.value)

                    # Check if this is the overall best
                    if best_value is None:
                        best_value = run_best_metric.value
                        best_step = run_best_metric.step
                        best_epoch = run.data.metrics.get(
                            "epoch", None
                        )  # Try to get epoch from run metrics
                        best_run_id = run.info.run_id
                        best_run_name = run.data.tags.get(
                            "mlflow.runName", run.info.run_id
                        )
                    else:
                        if direction == "min" and run_best_metric.value < best_value:
                            best_value = run_best_metric.value
                            best_step = run_best_metric.step
                            best_epoch = run.data.metrics.get("epoch", None)
                            best_run_id = run.info.run_id
                            best_run_name = run.data.tags.get(
                                "mlflow.runName", run.info.run_id
                            )
                        elif direction == "max" and run_best_metric.value > best_value:
                            best_value = run_best_metric.value
                            best_step = run_best_metric.step
                            best_epoch = run.data.metrics.get("epoch", None)
                            best_run_id = run.info.run_id
                            best_run_name = run.data.tags.get(
                                "mlflow.runName", run.info.run_id
                            )

                except Exception as e:
                    print(
                        f"    Warning: Error processing metric '{metric_name}' for run {run.info.run_id}: {e}"
                    )
                    continue

            # Add result if we found a best value
            if best_value is not None:
                # Try to get epoch from the specific step if not available from run metrics
                if best_epoch is None:
                    print("Epoch is none, falling back to second approach.")
                    # Look for epoch metric at the same step
                    try:
                        epoch_history = client.get_metric_history(best_run_id, "epoch")
                        if epoch_history:
                            # Find epoch value closest to our best step
                            closest_epoch_metric = min(
                                epoch_history, key=lambda x: abs(x.step - best_step)
                            )
                            best_epoch = closest_epoch_metric.value
                    except:
                        best_epoch = None

                results.append(
                    {
                        "experiment_name": exp_name,
                        "metric_name": metric_name,
                        "optimization": direction,
                        "best_value": best_value,
                        "step": best_step,
                        "epoch": best_epoch,
                        "run_id": best_run_id,
                        "run_name": best_run_name,
                    }
                )

                print(
                    f"    Best {direction}: {best_value:.6f} at step {best_step} (epoch {best_epoch}) from run {best_run_name}"
                )
            else:
                print(
                    f"    No data found for metric '{metric_name}' in experiment '{exp_name}'"
                )

    # Create DataFrame
    if not results:
        print("No results found for any metrics")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Sort by experiment name, then by metric name
    df = df.sort_values(["experiment_name", "metric_name"])

    return df
