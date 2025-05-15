import os
import subprocess
import sys
import tempfile
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Literal, Optional

import mlflow
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

    tqdm_default = os.getenv("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"

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

    if tqdm_default is not None:
        os.environ["TQDM_DISABLE"] = tqdm_default


def get_last_successful_run(experiment_name: str) -> Optional[mlflow.entities.Run]:
    """Find the most recent successful run to continue training from.

    Args:
        experiment_name (str): Name of the MLflow experiment

    Returns:
        Optional[mlflow.entities.Run]: Most recent successful run or None if not found
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
        filter_string="status = 'FINISHED'",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        order_by=["start_time DESC"],
    )
    runs = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]

    if not runs:
        logger.info(f"No successful runs found for experiment '{experiment_name}'")
        return None

    logger.info(
        f"Found successful run {runs[0].info.run_id} for experiment '{experiment_name}'"
    )
    return runs[0]
    # except Exception as e:
    #     logger.error(
    #         f"Error retrieving last successful run for '{experiment_name}': {str(e)}"
    #     )
    #     return None


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
