# metrics-cli-helper.py
"""
Helper utility for metrics-cli commands.
Run with: python metrics-cli-helper.py
Or integrate into your CLI with: metrics-cli help
"""

import argparse


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_subheader(title: str):
    """Print a formatted subheader."""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")


def get_command_info() -> dict[str, dict]:
    """Return comprehensive command information."""
    return {
        "analyze": {
            "description": "Process MLflow experiment metrics and generate visualizations",
            "purpose": "Analyze experiment metrics, calculate mean values, and create scatter plots",
            "use_cases": [
                "Compare model performance across experiments",
                "Visualize metric relationships (e.g., accuracy vs loss)",
                "Generate reports with calculated mean metrics",
                "Export processed data to CSV for further analysis",
            ],
            "key_features": [
                "Calculate mean across specified metrics",
                "Sort results by any metric",
                "Filter by run type (parent/children/both)",
                "Export to CSV and plot files",
                "Interactive or static visualizations",
            ],
            "example": {
                "command": "metrics-cli analyze",
                "args": [
                    "--storage-path ./mlflow",
                    "--select-metrics accuracy loss f1_score precision recall",
                    "--metrics-to-mean accuracy f1_score precision recall",
                    "--mean-metric-name avg_performance",
                    "--sort-metric avg_performance",
                    "--aggregate-all-runs",
                    "--run-mode both",
                    "--filter \"params.dataset = 'validation'\"",
                    "--output-csv experiment_analysis.csv",
                    "--output-plot performance_plot.html",
                    "--x-metric accuracy",
                    "--y-metric loss",
                    "--color-metric avg_performance",
                    '--title "Model Performance Analysis"',
                    "--theme plotly_dark",
                ],
            },
        },
        "compare-experiments": {
            "description": "Compare a single metric across multiple experiments",
            "purpose": "Generate print-friendly comparison plots showing metric evolution",
            "use_cases": [
                "Compare training curves across different model architectures",
                "Analyze convergence patterns between experiments",
                "Create publication-ready comparison charts",
                "Track metric improvements over time",
            ],
            "key_features": [
                "Print-friendly visualization with distinct line styles",
                "Support for step or epoch-based x-axis",
                "Custom legend names for experiments",
                "High-DPI output for publications",
                "Automatic subsampling for cleaner plots",
            ],
            "example": {
                "command": "metrics-cli compare-experiments",
                "args": [
                    "--storage-path ./mlflow",
                    "--experiment-names baseline_cnn resnet50_transfer bert_finetuned",
                    '--legend-names "Baseline CNN" "ResNet50 Transfer" "BERT Fine-tuned"',
                    "--metric val_accuracy",
                    "--mode epoch",
                    "--output-file model_comparison.png",
                    '--title "Validation Accuracy Comparison"',
                    "--figsize 12 8",
                    "--dpi 300",
                ],
            },
        },
        "manage-checkpoints": {
            "description": "Analyze and manage checkpoint files based on MLflow data",
            "purpose": "Identify which checkpoints to keep or remove based on performance",
            "use_cases": [
                "Clean up disk space by removing suboptimal checkpoints",
                "Identify best-performing model checkpoints",
                "Audit checkpoint-MLflow consistency",
                "Maintain only essential model saves",
            ],
            "key_features": [
                "Smart checkpoint analysis based on metrics",
                "Safety with dry-run mode",
                "Custom metric direction mapping",
                "Detailed reporting of actions",
                "Handles missing/orphaned checkpoints",
            ],
            "example": {
                "command": "metrics-cli manage-checkpoints",
                "args": [
                    "--checkpoints-dir ./saved_models",
                    "--storage-path ./mlflow",
                    '--custom-metrics "custom_loss:min" "weighted_f1:max" "perplexity:min"',
                    "--ignore-metrics epoch lr_scheduler_step",
                    "--output-report checkpoint_analysis.txt",
                    "--dry-run",
                    # "--remove"  # Uncomment to actually remove files
                ],
            },
        },
        "best-metrics": {
            "description": "Find the best values for specified metrics across experiments",
            "purpose": "Identify top-performing runs and extract their best metric values",
            "use_cases": [
                "Find best hyperparameter combinations",
                "Extract top-performing model configurations",
                "Compare peak performance across experiments",
                "Generate performance leaderboards",
            ],
            "key_features": [
                "Multi-metric optimization support",
                "Configurable min/max optimization per metric",
                "Experiment-wide search capabilities",
                "Detailed run information in results",
                "CSV export for further analysis",
            ],
            "example": {
                "command": "metrics-cli best-metrics",
                "args": [
                    "--storage-path ./mlflow",
                    "--experiment-names text_classification image_segmentation nlp_tasks",
                    "--metrics accuracy loss f1_score auc_score",
                    "--min-or-max max min max max",
                    "--run-mode parent",
                    "--filter \"params.model_type != 'baseline'\"",
                    "--output-csv best_models_summary.csv",
                ],
            },
        },
    }


def print_command_list():
    """Print a summary of all available commands."""
    print_header("METRICS-CLI COMMANDS")

    commands = get_command_info()

    for cmd_name, cmd_info in commands.items():
        print(f"\n📊 {cmd_name.upper()}")
        print(f"   {cmd_info['description']}")
        print(f"   Purpose: {cmd_info['purpose']}")


def print_command_details(command_name: str = None):
    """Print detailed information about a specific command or all commands."""
    commands = get_command_info()

    if command_name and command_name in commands:
        # Print details for specific command
        cmd_info = commands[command_name]
        print_header(f"COMMAND: {command_name.upper()}")

        print(f"\n📝 DESCRIPTION:")
        print(f"   {cmd_info['description']}")

        print(f"\n🎯 PURPOSE:")
        print(f"   {cmd_info['purpose']}")

        print(f"\n💡 USE CASES:")
        for use_case in cmd_info["use_cases"]:
            print(f"   • {use_case}")

        print(f"\n✨ KEY FEATURES:")
        for feature in cmd_info["key_features"]:
            print(f"   • {feature}")

        print_subheader("EXAMPLE USAGE")
        example = cmd_info["example"]
        print(f"\n{example['command']} \\")
        for i, arg in enumerate(example["args"]):
            if i < len(example["args"]) - 1:
                print(f"  {arg} \\")
            else:
                print(f"  {arg}")

        print_subheader("COPY-PASTE READY")
        full_command = f"{example['command']} {' '.join(example['args'])}"
        print(f"\n{full_command}")

    else:
        # Print details for all commands
        print_header("DETAILED COMMAND REFERENCE")

        for cmd_name, cmd_info in commands.items():
            print_subheader(f"{cmd_name.upper()} COMMAND")

            print(f"\n📝 {cmd_info['description']}")
            print(f"🎯 {cmd_info['purpose']}")

            print(f"\n💡 Use Cases:")
            for use_case in cmd_info["use_cases"]:
                print(f"   • {use_case}")

            print(f"\n✨ Key Features:")
            for feature in cmd_info["key_features"]:
                print(f"   • {feature}")

            print(f"\n🔧 Example:")
            example = cmd_info["example"]
            print(f"   {example['command']} \\")
            for i, arg in enumerate(example["args"]):
                if i < len(example["args"]) - 1:
                    print(f"     {arg} \\")
                else:
                    print(f"     {arg}")


def print_quick_start():
    """Print a quick start guide."""
    print_header("QUICK START GUIDE")

    print(
        """
🚀 GETTING STARTED:

1. First, ensure you have MLflow experiments with logged metrics
2. Identify your MLflow storage path (usually './mlflow' or './mlruns')
3. Choose the appropriate command based on your needs:

   📊 ANALYZE: For comprehensive metric analysis and visualization
   📈 COMPARE-EXPERIMENTS: For comparing metrics across experiments  
   🗂️  MANAGE-CHECKPOINTS: For checkpoint cleanup and management
   🏆 BEST-METRICS: For finding top-performing runs

4. Start with the 'analyze' command to get familiar with your data
5. Use --dry-run flag with manage-checkpoints before actual cleanup

💡 TIPS:
   • Use --help with any command for detailed argument descriptions
   • Start with smaller datasets to test your commands
   • Always backup important checkpoints before cleanup
   • Use filters to narrow down your analysis scope
"""
    )


def print_common_patterns():
    """Print common usage patterns and tips."""
    print_header("COMMON USAGE PATTERNS")

    patterns = [
        {
            "title": "🔍 EXPLORATORY ANALYSIS",
            "description": "Start by analyzing your experiments to understand the data",
            "command": "metrics-cli analyze --storage-path ./mlflow --select-metrics accuracy loss --metrics-to-mean accuracy --mean-metric-name avg_acc --sort-metric avg_acc --x-metric accuracy --y-metric loss",
        },
        {
            "title": "📊 MODEL COMPARISON",
            "description": "Compare training curves across different model architectures",
            "command": "metrics-cli compare-experiments --storage-path ./mlflow --experiment-names exp1 exp2 exp3 --metric val_loss --mode epoch --output-file comparison.png",
        },
        {
            "title": "🧹 CHECKPOINT CLEANUP",
            "description": "Clean up disk space by removing suboptimal checkpoints",
            "command": "metrics-cli manage-checkpoints --checkpoints-dir ./models --storage-path ./mlflow --dry-run --output-report cleanup_report.txt",
        },
        {
            "title": "🏆 LEADERBOARD CREATION",
            "description": "Find the best performing models across experiments",
            "command": "metrics-cli best-metrics --storage-path ./mlflow --experiment-names exp1 exp2 --metrics accuracy f1_score --min-or-max max max --output-csv leaderboard.csv",
        },
        {
            "title": "🎯 FILTERED ANALYSIS",
            "description": "Analyze only specific runs matching certain criteria",
            "command": 'metrics-cli analyze --storage-path ./mlflow --select-metrics accuracy --metrics-to-mean accuracy --mean-metric-name avg_acc --sort-metric avg_acc --filter "params.learning_rate > 0.001" --x-metric accuracy --y-metric loss',
        },
    ]

    for pattern in patterns:
        print(f"\n{pattern['title']}")
        print(f"   {pattern['description']}")
        print(f"   Command: {pattern['command']}")


def print_troubleshooting():
    """Print common troubleshooting tips."""
    print_header("TROUBLESHOOTING")

    print(
        """
🔧 COMMON ISSUES & SOLUTIONS:

❌ "No experiments found"
   ✅ Check your --storage-path points to correct MLflow directory
   ✅ Ensure experiments have been logged to MLflow

❌ "No runs found with specified metrics"
   ✅ Verify metric names match those logged in MLflow
   ✅ Check if runs completed successfully (status = 'FINISHED')
   ✅ Adjust --run-mode filter (parent/children/both)

❌ "Checkpoint files not found"
   ✅ Verify --checkpoints-dir path is correct
   ✅ Ensure checkpoint files have expected extensions (.pt, .pth, .ckpt)

❌ "Invalid metric direction"
   ✅ Use --custom-metrics to specify metric optimization direction
   ✅ Format: "metric_name:min" or "metric_name:max"

❌ "Plot not displaying"
   ✅ Install required packages: plotly, kaleido for static images
   ✅ Use --output-plot to save instead of displaying

❌ "Permission denied during checkpoint removal"
   ✅ Check file permissions in checkpoint directory
   ✅ Ensure files are not being used by other processes

💡 DEBUGGING TIPS:
   • Use --dry-run with manage-checkpoints to preview actions
   • Start with small datasets to test commands
   • Check MLflow UI to verify experiment structure
   • Use --filter to narrow down problematic runs
"""
    )


def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Helper utility for metrics-cli commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "action",
        nargs="?",
        choices=["list", "details", "quick-start", "patterns", "troubleshooting"],
        default="list",
        help="Action to perform (default: list)",
    )

    parser.add_argument(
        "--command",
        "-c",
        choices=[
            "analyze",
            "compare-experiments",
            "manage-checkpoints",
            "best-metrics",
        ],
        help="Show details for specific command",
    )

    args = parser.parse_args()

    if args.action == "list":
        print_command_list()
    elif args.action == "details":
        print_command_details(args.command)
    elif args.action == "quick-start":
        print_quick_start()
    elif args.action == "patterns":
        print_common_patterns()
    elif args.action == "troubleshooting":
        print_troubleshooting()

    # Always show a helpful footer
    print(f"\n{'='*60}")
    print("🔗 For more help:")
    print("   metrics-cli <command> --help    # Detailed argument descriptions")
    print(
        "   python metrics-cli-helper.py details --command <name>  # Specific command help"
    )
    print(
        "   python metrics-cli-helper.py quick-start              # Getting started guide"
    )
    print(
        "   python metrics-cli-helper.py patterns                 # Common usage patterns"
    )
    print(
        "   python metrics-cli-helper.py troubleshooting          # Common issues & solutions"
    )
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
