#!/usr/bin/env python

import argparse

import pandas as pd

from yumbox.mlflow import process_experiment_metrics, visualize_metrics


def analyze_metrics(args):
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
                tablefmt="psql",
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

    # Run visualize_metrics
    visualize_metrics(
        df=df,
        x_metric=args.x_metric,
        y_metric=args.y_metric,
        color_metric=args.color_metric,
        title=args.title,
        theme=args.theme,
    )


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool to analyze MLflow experiment metrics by processing and visualizing them."
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # Subparser for analyze (wrapper)
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

    args = parser.parse_args()

    if args.command == "analyze":
        analyze_metrics(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
