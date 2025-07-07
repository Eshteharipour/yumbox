#!/usr/bin/env python

import argparse


def analyze_metrics(args):
    import pandas as pd

    from yumbox.mlflow import process_experiment_metrics, visualize_metrics

    """Process and visualize MLflow experiment metrics in one command."""
    # Run process_experiment_metrics
    df = process_experiment_metrics(
        storage_path=args.storage_path,
        select_metrics=args.select_metrics,
        metrics_to_mean=args.metrics_to_mean,
        mean_metric_name=args.mean_metric_name,
        sort_metric=args.sort_metric,
        aggregate_all_runs=args.aggregate_all_runs,
        run_mode=args.run_mode,
        filter=args.filter,
    )

    if df.empty:
        print("No data to visualize. Exiting.")
        return

    # Set pandas display options
    pd.set_option("display.max_columns", None)  # Show all columns
    pd.set_option("display.max_colwidth", None)  # Show full content of each cell
    pd.set_option("display.width", None)  # Auto-detect terminal width
    pd.set_option("display.colheader_justify", "left")
    try:
        from tabulate import tabulate

        print(
            tabulate(
                df,
                headers="keys",
                tablefmt="github",  # psql
                showindex=False,
                colalign=("left",),
            )
        )
    except ImportError:
        print(df)

    # Save to CSV if output path is provided
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"Processed metrics saved to {args.output_csv}")

    # Run visualize_metrics with optional file output
    visualize_metrics(
        df=df,
        x_metric=args.x_metric,
        y_metric=args.y_metric,
        color_metric=args.color_metric,
        title=args.title,
        theme=args.theme,
        output_file=args.output_plot,
    )


def compare_experiments(args):
    """Compare metric across multiple experiments with print-friendly visualization."""
    from yumbox.mlflow import plot_metric_across_experiments, set_tracking_uri

    set_tracking_uri(args.storage_path)

    plot_metric_across_experiments(
        experiment_names=args.experiment_names,
        metric_key=args.metric,
        mode=args.mode,
        legend_names=args.legend_names,
        artifact_file=args.output_file,
        title=args.title,
        figsize=tuple(args.figsize) if args.figsize else (10, 6),
        dpi=args.dpi,
    )

    print(f"Experiment comparison plot generated for metric '{args.metric}'")
    print(f"Experiments compared: {', '.join(args.experiment_names)}")
    if args.legend_names:
        print(f"Legend names: {', '.join(args.legend_names)}")


def manage_checkpoints(args):
    """Analyze and manage checkpoint files based on MLflow experiment data."""
    from yumbox.mlflow.checkpoint_helpers import (
        analyze_checkpoint_status,
        execute_checkpoint_removal,
        format_checkpoint_report,
    )

    # Define metric direction mapping
    # You can expand this dict as needed for your specific metrics
    metric_direction_map = {
        "loss": "min",
        "error": "min",
        "mse": "min",
        "rmse": "min",
        "mae": "min",
        "accuracy": "max",
        "acc": "max",
        "precision": "max",
        "recall": "max",
        "f1": "max",
        "auc": "max",
        "dice": "max",
        "iou": "max",
        "bleu": "max",
        "rouge": "max",
        "precision": "max",
        "pr_auc": "max",
        "roc_auc": "max",
        "neg_dist": "max",
        "pos_dist": "min",
    }

    # Add custom metric directions if provided
    if args.custom_metrics:
        for metric_spec in args.custom_metrics:
            if ":" not in metric_spec:
                print(
                    f"ERROR: Invalid custom metric format '{metric_spec}'. Use 'metric_name:min' or 'metric_name:max'"
                )
                return
            metric_name, direction = metric_spec.split(":", 1)
            if direction not in ["min", "max"]:
                print(
                    f"ERROR: Invalid direction '{direction}' for metric '{metric_name}'. Use 'min' or 'max'"
                )
                return
            metric_direction_map[metric_name.strip()] = direction.strip()

    # Define and combine ignored metrics
    default_ignored_metrics = {"lr", "step"}
    user_ignored_metrics = set(args.ignore_metrics)
    ignored_metrics = default_ignored_metrics.union(user_ignored_metrics)
    print(f"Ignoring metrics: {', '.join(sorted(list(ignored_metrics)))}")

    try:
        # Analyze checkpoint status
        keep_set, remove_set, deleted_set, non_referenced_set, reasons = (
            analyze_checkpoint_status(
                checkpoints_dir=args.checkpoints_dir,
                storage_path=args.storage_path,
                metric_direction_map=metric_direction_map,
                ignore_metrics=ignored_metrics,
            )
        )

        # Generate and display report
        report = format_checkpoint_report(
            keep_set=keep_set,
            remove_set=remove_set,
            deleted_set=deleted_set,
            non_referenced_set=non_referenced_set,
            reasons=reasons,
            checkpoints_dir=args.checkpoints_dir,
        )

        print(report)

        # Save report to file if requested
        if args.output_report:
            with open(args.output_report, "w") as f:
                f.write(report)
            print(f"Report saved to: {args.output_report}")

        # Execute removal if requested
        if remove_set and args.remove:
            execute_checkpoint_removal(remove_set=remove_set, dry_run=args.dry_run)

    except ValueError as e:
        print(f"ERROR: {e}")
        print(
            "\nPlease update your metric_direction_map or use --custom-metrics to specify directions for missing metrics."
        )
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")


def find_best_metrics_command(args):
    """CLI command to find best metrics across experiments."""
    import pandas as pd

    from yumbox.mlflow import find_best_metrics

    # Process the best metrics
    df = find_best_metrics(
        storage_path=args.storage_path,
        experiment_names=args.experiment_names,
        metrics=args.metrics,
        min_or_max=args.min_or_max,
        run_mode=args.run_mode,
        filter_string=args.filter,
    )

    if df.empty:
        print("No data found. Exiting.")
        return

    # Set pandas display options for better output
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)
    pd.set_option("display.colheader_justify", "left")

    # Print results using tabulate if available
    try:
        from tabulate import tabulate

        print("\nBest Metrics Results:")
        print("=" * 80)
        print(
            tabulate(
                df,
                headers="keys",
                tablefmt="github",
                showindex=False,
                colalign=("left",),
                floatfmt=".6f",
            )
        )
    except ImportError:
        print("\nBest Metrics Results:")
        print("=" * 80)
        print(df)

    # Save to CSV if requested
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to {args.output_csv}")

    # Print summary statistics
    print(f"\nSummary:")
    print(f"- Total experiments processed: {df['experiment_name'].nunique()}")
    print(f"- Total metrics analyzed: {df['metric_name'].nunique()}")
    print(f"- Total best values found: {len(df)}")


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool to analyze MLflow experiment metrics by processing and visualizing them."
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # Subparser for analyze
    analyze_parser = subparsers.add_parser(
        "analyze", help="Process MLflow experiment metrics and generate a visualization"
    )
    analyze_parser.add_argument(
        "--storage-path",
        type=str,
        required=True,
        help="Path to the MLflow storage folder containing experiment data (e.g., './mlflow').",
    )
    analyze_parser.add_argument(
        "--select-metrics",
        type=str,
        nargs="+",
        required=True,
        help="Space-separated list of metric names to collect from MLflow runs (e.g., 'acc loss').",
    )
    analyze_parser.add_argument(
        "--metrics-to-mean",
        type=str,
        nargs="+",
        required=True,
        help="Space-separated list of metric names to calculate the mean for (e.g., 'acc loss'). Must be a subset of select-metrics.",
    )
    analyze_parser.add_argument(
        "--mean-metric-name",
        type=str,
        required=True,
        help="Name for the calculated mean metric (e.g., 'avg_score').",
    )
    analyze_parser.add_argument(
        "--sort-metric",
        type=str,
        required=True,
        help="Metric to sort the results by (e.g., 'acc'). Must be one of select-metrics or mean-metric-name.",
    )
    analyze_parser.add_argument(
        "--aggregate-all-runs",
        action="store_true",
        help="If set, process all successful runs; otherwise, process only the most recent run.",
    )
    analyze_parser.add_argument(
        "--run-mode",
        type=str,
        choices=["parent", "children", "both"],
        default="both",
        help="Filter runs: 'parent' (parent runs only), 'children' (child runs only), or 'both' (all runs). Default: 'both'.",
    )
    analyze_parser.add_argument(
        "--filter",
        type=str,
        help="MLflow filter string to select runs (e.g., \"params.dataset = 'lip'\"). Optional.",
    )
    analyze_parser.add_argument(
        "--output-csv",
        type=str,
        help="Path to save the processed metrics as a CSV file (e.g., 'metrics.csv'). Optional.",
    )
    analyze_parser.add_argument(
        "--output-plot",
        type=str,
        help="Path to save the plot as an image file (e.g., 'plot.png', 'plot.html'). Supports PNG, JPG, SVG, PDF, and HTML formats. If not provided, plot will be displayed interactively. Optional.",
    )
    analyze_parser.add_argument(
        "--x-metric",
        type=str,
        required=True,
        help="Metric to use for the x-axis in the visualization (e.g., 'acc'). Must be in the processed metrics.",
    )
    analyze_parser.add_argument(
        "--y-metric",
        type=str,
        required=True,
        help="Metric to use for the y-axis in the visualization (e.g., 'loss'). Must be in the processed metrics.",
    )
    analyze_parser.add_argument(
        "--color-metric",
        type=str,
        help="Metric to use for the color scale in the visualization (e.g., 'avg_score'). Optional.",
    )
    analyze_parser.add_argument(
        "--title",
        type=str,
        default="Experiment Metrics Visualization",
        help="Title for the Plotly visualization. Default: 'Experiment Metrics Visualization'.",
    )
    analyze_parser.add_argument(
        "--theme",
        type=str,
        default="plotly_dark",
        help="Plotly theme for the visualization (e.g., 'plotly', 'plotly_dark', 'plotly_white'). Default: 'plotly_dark'.",
    )

    # Subparser for compare-experiments
    compare_parser = subparsers.add_parser(
        "compare-experiments",
        help="Compare a single metric across multiple experiments with print-friendly visualization",
    )
    compare_parser.add_argument(
        "--storage-path",
        type=str,
        required=True,
        help="Path to the MLflow storage folder containing experiment data (e.g., './mlflow').",
    )
    compare_parser.add_argument(
        "--experiment-names",
        type=str,
        nargs="+",
        required=True,
        help="Space-separated list of experiment names to compare (e.g., 'exp1' 'exp2' 'exp3').",
    )
    compare_parser.add_argument(
        "--legend-names",
        type=str,
        nargs="+",
        help="Space-separated list of custom names for legend in same order as experiment-names (e.g., 'Baseline' 'Method A' 'Method B'). Optional.",
    )
    compare_parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="Single metric name to plot across experiments (e.g., 'train_loss', 'val_accuracy').",
    )
    compare_parser.add_argument(
        "--mode",
        type=str,
        choices=["step", "epoch"],
        default="step",
        help="X-axis mode: 'step' for training steps or 'epoch' for epochs. Default: 'step'.",
    )
    compare_parser.add_argument(
        "--output-file",
        type=str,
        help="Custom filename for the output plot artifact (e.g., 'my_comparison.png'). Optional.",
    )
    compare_parser.add_argument(
        "--title",
        type=str,
        help="Custom title for the plot. If not provided, will auto-generate based on metric name.",
    )
    compare_parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches as two values: width height (e.g., 12 8). Default: 10 6.",
    )
    compare_parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for high-quality output suitable for printing. Default: 300.",
    )

    # Subparser for manage-checkpoints
    checkpoint_parser = subparsers.add_parser(
        "manage-checkpoints",
        help="Analyze MLflow experiments to suggest which checkpoints to keep or remove",
    )
    checkpoint_parser.add_argument(
        "--checkpoints-dir",
        type=str,
        required=True,
        help="Path to the directory containing checkpoint files to analyze (e.g., './checkpoints').",
    )
    checkpoint_parser.add_argument(
        "--storage-path",
        type=str,
        required=True,
        help="Path to the MLflow storage folder containing experiment data (e.g., './mlflow').",
    )
    checkpoint_parser.add_argument(
        "--custom-metrics",
        type=str,
        nargs="*",
        help="Custom metric direction mappings in format 'metric_name:min' or 'metric_name:max' (e.g., 'custom_loss:min' 'my_score:max'). Optional.",
    )
    checkpoint_parser.add_argument(
        "--ignore-metrics",
        type=str,
        nargs="*",
        default=[],
        help="Space-separated list of metric names to ignore during analysis (e.g., 'epoch'). 'lr' is ignored by default.",
    )
    checkpoint_parser.add_argument(
        "--output-report",
        type=str,
        help="Path to save the analysis report as a text file (e.g., 'checkpoint_report.txt'). Optional.",
    )
    checkpoint_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what checkpoints would be removed without actually removing them.",
    )
    checkpoint_parser.add_argument(
        "--remove",
        action="store_true",
        help="Actually remove the suggested checkpoint files. Use with caution!",
    )

    # Subparser for best-metrics
    best_parser = subparsers.add_parser(
        "best-metrics",
        help="Find the best values for specified metrics across multiple experiments",
    )
    best_parser.add_argument(
        "--storage-path",
        type=str,
        required=True,
        help="Path to the MLflow storage folder containing experiment data (e.g., './mlflow').",
    )
    best_parser.add_argument(
        "--experiment-names",
        type=str,
        nargs="+",
        required=True,
        help="Space-separated list of experiment names to search (e.g., 'exp1' 'exp2' 'exp3').",
    )
    best_parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        required=True,
        help="Space-separated list of metric names to find best values for (e.g., 'loss' 'accuracy' 'f1_score').",
    )
    best_parser.add_argument(
        "--min-or-max",
        type=str,
        nargs="+",
        required=True,
        choices=["min", "max"],
        help="Space-separated list of 'min' or 'max' for each metric, indicating whether lower or higher values are better. Must match the order and length of --metrics.",
    )
    best_parser.add_argument(
        "--run-mode",
        type=str,
        choices=["parent", "children", "both"],
        default="parent",
        help="Filter runs: 'parent' (parent runs only), 'children' (child runs only), or 'both' (all runs). Default: 'parent'.",
    )
    best_parser.add_argument(
        "--filter",
        type=str,
        help="MLflow filter string to select runs (e.g., \"params.dataset = 'lip'\"). Optional.",
    )
    best_parser.add_argument(
        "--output-csv",
        type=str,
        help="Path to save the results as a CSV file (e.g., 'best_metrics.csv'). Optional.",
    )

    args = parser.parse_args()

    if args.command == "analyze":
        analyze_metrics(args)
    elif args.command == "compare-experiments":
        compare_experiments(args)
    elif args.command == "manage-checkpoints":
        manage_checkpoints(args)
    elif args.command == "best-metrics":
        find_best_metrics_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
