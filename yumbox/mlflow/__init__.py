import os
import subprocess
import sys
import tempfile
from logging import Logger
from pathlib import Path
from typing import Literal, Optional

import mlflow
from mlflow import MlflowClient
from omegaconf import DictConfig, OmegaConf

from yumbox.cache import BFG

DATE_TIME_FORMAT = "%Y-%m-%dT%H-%M-%S%z"


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


def log_scores_dict(name, scores_dict):
    for k, v in scores_dict.items():
        mlflow.log_metric(name + "_" + k, v)


def log_config(cfg: DictConfig, logger: None | Logger):
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "config.yaml")

    # OmegaConf.to_yaml(CFG)
    OmegaConf.save(cfg, config_path)
    mlflow.log_artifact(config_path)

    os.remove(config_path)
    os.rmdir(temp_dir)

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
):
    logger = BFG["logger"]

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

        os.environ["TQDM_DISABLE"] = "1"

        cmd = [executable, script, config_arg, config_file]
        if extra_args:
            cmd.extend(extra_args)

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            while process.poll() is None:
                output = process.stdout.readline()
                if output:
                    logger.info(output.strip())

            return_code = process.wait()
            if return_code != 0:
                error_output = process.stderr.read()
                logger.info(f"✗ Failed: {config_file}")
                logger.info(f"Error: {error_output}")
            else:
                logger.info(f"✓ Completed: {config_file}")

        except Exception as e:
            logger.info(f"✗ Failed: {config_file}")
            logger.info(f"Error: {str(e)}")

        logger.info("-" * 50)


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


def set_tracking_uri(path: str):
    # If path is absolute
    if path.startswith("/"):
        mlflow.set_tracking_uri(f"file:{path}")
    # Otherwise get parent dir of entrypoint script
    else:
        main_file = Path(sys.argv[0]).parent.resolve()
        mlflow_path = os.path.join(main_file, path)
        mlflow.set_tracking_uri(f"file:{mlflow_path}")
