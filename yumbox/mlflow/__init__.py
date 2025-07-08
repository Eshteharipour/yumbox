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
    experiments = client.search_experiments(view_type=entities.ViewType.ACTIVE_ONLY)
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
    aggregation_mode: Literal["all_runs", "best_run"] = "all_runs",
) -> pd.DataFrame:
    """
    Find the best values for specified metrics across experiments.

    Args:
        storage_path: Path to MLflow storage folder
        experiment_names: List of experiment names to search (supports regex patterns)
        metrics: List of metric names to find best values for (supports regex patterns)
        min_or_max: List of 'min' or 'max' corresponding to each metric
        run_mode: Filter runs by 'parent', 'children', or 'both'
        filter_string: Optional MLflow filter string
        aggregation_mode: 'all_runs' for current behavior, 'best_run' to keep only best run per metric

    Returns:
        pandas.DataFrame: Results with metrics as columns, experiments/runs as rows
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

    # Get all available experiments
    all_experiments = client.search_experiments()

    # Match experiment names using regex
    matched_experiments = []
    for exp_pattern in experiment_names:
        if exp_pattern is None or exp_pattern == "none":
            # Include all experiments
            matched_experiments.extend([exp.name for exp in all_experiments])
        else:
            try:
                regex_pattern = re.compile(exp_pattern)
                matched_exp_names = [
                    exp.name
                    for exp in all_experiments
                    if regex_pattern.search(exp.name)
                ]
                if matched_exp_names:
                    matched_experiments.extend(matched_exp_names)
                else:
                    print(f"No experiments found matching pattern: {exp_pattern}")
            except re.error as e:
                print(f"Invalid regex pattern '{exp_pattern}': {e}")
                continue

    # Remove duplicates while preserving order
    matched_experiments = list(dict.fromkeys(matched_experiments))

    if not matched_experiments:
        print("No experiments found matching the provided patterns")
        return pd.DataFrame()

    # First, collect all metrics across all experiments to get the full set
    all_matched_metrics = set()
    metric_directions = {}  # metric_name -> direction

    for exp_name in matched_experiments:
        print(f"Processing experiment: {exp_name}")

        # Get runs for this experiment
        runs = get_mlflow_runs(
            exp_name, status="success", level=None, filter=filter_string
        )

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

        # Get all available metrics from runs to support regex matching
        all_metrics_in_exp = set()
        for run in runs:
            all_metrics_in_exp.update(run.data.metrics.keys())
            # Also get metric history keys
            try:
                for metric_key in run.data.metrics.keys():
                    metric_history = client.get_metric_history(
                        run.info.run_id, metric_key
                    )
                    if metric_history:
                        all_metrics_in_exp.add(metric_key)
            except:
                pass

        # Match metrics using regex for this experiment
        for i, metric_pattern in enumerate(metrics):
            if metric_pattern is None or metric_pattern == "none":
                for metric_name in all_metrics_in_exp:
                    all_matched_metrics.add(metric_name)
                    metric_directions[metric_name] = min_or_max[i]
            else:
                try:
                    regex_pattern = re.compile(metric_pattern)
                    matched_metric_names = [
                        m for m in all_metrics_in_exp if regex_pattern.search(m)
                    ]
                    for metric_name in matched_metric_names:
                        all_matched_metrics.add(metric_name)
                        metric_directions[metric_name] = min_or_max[i]
                except re.error as e:
                    print(f"Invalid regex pattern '{metric_pattern}': {e}")
                    continue

    if not all_matched_metrics:
        print("No metrics found matching the provided patterns")
        return pd.DataFrame()

    # Now collect results for each experiment/run combination
    results = []

    for exp_name in matched_experiments:
        print(f"Processing experiment: {exp_name}")

        # Get runs for this experiment
        runs = get_mlflow_runs(
            exp_name, status="success", level=None, filter=filter_string
        )

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

        # For each run, collect best values for all metrics
        for run in runs:
            run_result = {
                "experiment_name": exp_name,
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", run.info.run_id),
            }

            # Add each metric as a column
            for metric_name in all_matched_metrics:
                direction = metric_directions[metric_name]

                try:
                    # Get metric history for this run
                    metric_history = client.get_metric_history(
                        run.info.run_id, metric_name
                    )

                    if not metric_history:
                        run_result[metric_name] = np.nan
                        run_result[f"{metric_name}_step"] = np.nan
                        run_result[f"{metric_name}_epoch"] = np.nan
                        continue

                    # Find best value in this run's history
                    if direction == "min":
                        best_metric = min(metric_history, key=lambda x: x.value)
                    else:  # max
                        best_metric = max(metric_history, key=lambda x: x.value)

                    run_result[metric_name] = best_metric.value
                    run_result[f"{metric_name}_step"] = best_metric.step

                    # Try to get epoch
                    epoch_value = run.data.metrics.get("epoch", None)
                    if epoch_value is None:
                        # Look for epoch metric at the same step
                        try:
                            epoch_history = client.get_metric_history(
                                run.info.run_id, "epoch"
                            )
                            if epoch_history:
                                valid_epochs = [
                                    e
                                    for e in epoch_history
                                    if e.step <= best_metric.step
                                ]
                                if valid_epochs:
                                    closest_epoch_metric = min(
                                        valid_epochs,
                                        key=lambda x: abs(x.step - best_metric.step),
                                    )
                                    epoch_value = closest_epoch_metric.value
                        except Exception as e:
                            epoch_value = None

                    run_result[f"{metric_name}_epoch"] = epoch_value

                except Exception as e:
                    print(
                        f"    Warning: Error processing metric '{metric_name}' for run {run.info.run_id}: {e}"
                    )
                    run_result[metric_name] = np.nan
                    run_result[f"{metric_name}_step"] = np.nan
                    run_result[f"{metric_name}_epoch"] = np.nan

            results.append(run_result)

    # Create DataFrame
    if not results:
        print("No results found for any metrics")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Apply aggregation mode
    if aggregation_mode == "best_run":
        # For each metric, keep only the best run across all experiments
        best_runs = []

        for metric_name in all_matched_metrics:
            direction = metric_directions[metric_name]

            # Filter out rows where this metric is NaN
            metric_df = df[df[metric_name].notna()]

            if len(metric_df) == 0:
                continue

            if direction == "min":
                best_row = metric_df.loc[metric_df[metric_name].idxmin()]
            else:  # max
                best_row = metric_df.loc[metric_df[metric_name].idxmax()]

            best_runs.append(best_row)

        # If we have best runs, create a new DataFrame with unique runs
        if best_runs:
            df = pd.DataFrame(best_runs).drop_duplicates(subset=["run_id"])

    # Sort by experiment name, then by run name
    df = df.sort_values(["experiment_name", "run_name"])

    # Sort columns to move _epoch and _step columns to the end
    if not df.empty:
        # Get all column names
        all_columns = df.columns.tolist()

        # Separate columns into regular and step/epoch columns
        regular_columns = []
        step_epoch_columns = []

        for col in all_columns:
            if col.endswith("_step") or col.endswith("_epoch"):
                step_epoch_columns.append(col)
            else:
                regular_columns.append(col)

        # Sort step/epoch columns for consistent ordering
        step_epoch_columns.sort()

        # Reorder DataFrame columns
        df = df[regular_columns + step_epoch_columns]

    return df
