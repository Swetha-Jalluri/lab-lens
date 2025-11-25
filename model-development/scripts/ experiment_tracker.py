"""
Experiment Tracker and Comparison for BioBART Training
Author: Lab Lens Team
Description: Analyzes MLflow experiments, compares training runs, generates visualizations
             Provides insights into model performance across different configurations
             
Required by Rubric: Section 4 - Experiment Tracking and Results

This script:
1. Connects to MLflow tracking server
2. Retrieves all experiment runs
3. Compares performance across runs
4. Generates comparison visualizations (loss curves, metric comparisons)
5. Creates summary report with best model selection rationale
6. Exports experiment data for documentation
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Data processing and visualization
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow
import mlflow
from mlflow.tracking import MlflowClient


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Configure logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_tracker_{timestamp}.log"
    
    logger = logging.getLogger("ExperimentTracker")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_experiments(tracking_uri: str, experiment_name: str, logger: logging.Logger) -> List[Dict]:
    """
    Load all runs from a specific MLflow experiment.
    
    Args:
        tracking_uri: MLflow tracking URI
        experiment_name: Name of experiment to load
        logger: Logger instance
        
    Returns:
        List of experiment run dictionaries
    """
    logger.info(f"Connecting to MLflow: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    
    client = MlflowClient(tracking_uri=tracking_uri)
    
    # Get experiment
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.error(f"Experiment '{experiment_name}' not found")
            return []
        
        experiment_id = experiment.experiment_id
        logger.info(f"✓ Found experiment: {experiment_name} (ID: {experiment_id})")
        
    except Exception as e:
        logger.error(f"Failed to get experiment: {e}")
        return []
    
    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"]
    )
    
    logger.info(f"✓ Loaded {len(runs)} runs from experiment")
    
    # Convert to list of dictionaries
    runs_data = []
    for run in runs:
        run_data = {
            'run_id': run.info.run_id,
            'run_name': run.data.tags.get('mlflow.runName', 'unnamed'),
            'status': run.info.status,
            'start_time': datetime.fromtimestamp(run.info.start_time / 1000),
            'end_time': datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
            'params': run.data.params,
            'metrics': run.data.metrics,
        }
        runs_data.append(run_data)
    
    return runs_data


def create_comparison_table(runs_data: List[Dict], logger: logging.Logger) -> pd.DataFrame:
    """
    Create comparison table of all experiments.
    
    Args:
        runs_data: List of experiment run dictionaries
        logger: Logger instance
        
    Returns:
        DataFrame with comparison data
    """
    logger.info("Creating experiment comparison table...")
    
    comparison_data = []
    
    for run in runs_data:
        if run['status'] != 'FINISHED':
            continue
        
        row = {
            'run_name': run['run_name'],
            'run_id': run['run_id'][:8],  # Short ID
            'start_time': run['start_time'].strftime('%Y-%m-%d %H:%M'),
        }
        
        # Add parameters
        for param_name, param_value in run['params'].items():
            row[f'param_{param_name}'] = param_value
        
        # Add metrics
        for metric_name, metric_value in run['metrics'].items():
            row[f'metric_{metric_name}'] = metric_value
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    logger.info(f"✓ Created comparison table with {len(df)} runs")
    
    return df


def generate_comparison_plots(df: pd.DataFrame, output_dir: Path, logger: logging.Logger):
    """
    Generate visualizations comparing experiments.
    
    Args:
        df: DataFrame with experiment comparison data
        output_dir: Directory to save plots
        logger: Logger instance
    """
    logger.info("Generating comparison visualizations...")
    
    # Extract metric columns
    metric_cols = [col for col in df.columns if col.startswith('metric_')]
    
    if not metric_cols:
        logger.warning("No metrics found to visualize")
        return
    
    # Plot 1: Metric comparison across all runs
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics_to_plot = [col for col in metric_cols if any(x in col for x in ['rouge', 'loss'])]
    
    x = range(len(df))
    width = 0.8 / len(metrics_to_plot)
    
    for i, metric_col in enumerate(metrics_to_plot):
        offset = (i - len(metrics_to_plot) / 2) * width
        values = pd.to_numeric(df[metric_col], errors='coerce').fillna(0)
        ax.bar([xi + offset for xi in x], values, width, 
               label=metric_col.replace('metric_', ''), alpha=0.8)
    
    ax.set_xlabel('Experiment Run', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Experiment Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['run_name'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'experiment_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved comparison plot: {plot_path}")
    
    # Plot 2: Best metrics summary
    if 'metric_rougeL' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # ROUGE scores
        ax = axes[0]
        rouge_metrics = ['metric_rouge1', 'metric_rouge2', 'metric_rougeL']
        rouge_data = df[rouge_metrics].apply(pd.to_numeric, errors='coerce')
        
        rouge_data.plot(kind='bar', ax=ax, alpha=0.7)
        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylabel('ROUGE Score', fontsize=12)
        ax.set_title('ROUGE Scores Across Experiments', fontsize=13, fontweight='bold')
        ax.legend(['ROUGE-1', 'ROUGE-2', 'ROUGE-L'])
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(range(len(df)), rotation=0)
        
        # Loss comparison
        ax = axes[1]
        if 'metric_eval_loss' in df.columns:
            loss_data = pd.to_numeric(df['metric_eval_loss'], errors='coerce')
            ax.plot(loss_data, marker='o', linewidth=2, markersize=8, color='coral')
            ax.set_xlabel('Experiment', fontsize=12)
            ax.set_ylabel('Validation Loss', fontsize=12)
            ax.set_title('Validation Loss Across Experiments', fontsize=13, fontweight='bold')
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / 'metrics_summary.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved metrics summary plot: {plot_path}")


def identify_best_model(df: pd.DataFrame, logger: logging.Logger) -> Dict:
    """
    Identify best performing model from experiments.
    
    Args:
        df: DataFrame with experiment results
        logger: Logger instance
        
    Returns:
        Dictionary with best model information
    """
    logger.info("=" * 80)
    logger.info("IDENTIFYING BEST MODEL")
    logger.info("=" * 80)
    
    if 'metric_rougeL' not in df.columns:
        logger.warning("ROUGE-L metric not found")
        return {}
    
    # Convert to numeric
    df['metric_rougeL_numeric'] = pd.to_numeric(df['metric_rougeL'], errors='coerce')
    
    # Find best run
    best_idx = df['metric_rougeL_numeric'].idxmax()
    best_run = df.iloc[best_idx]
    
    logger.info(f"\nBest Model:")
    logger.info(f"  Run: {best_run['run_name']}")
    logger.info(f"  Run ID: {best_run['run_id']}")
    logger.info(f"  ROUGE-L: {best_run['metric_rougeL']}")
    
    # Extract parameters
    param_cols = [col for col in df.columns if col.startswith('param_')]
    
    if param_cols:
        logger.info(f"\nBest Hyperparameters:")
        for col in param_cols:
            logger.info(f"  {col.replace('param_', '')}: {best_run[col]}")
    
    best_model_info = {
        'run_id': best_run['run_id'],
        'run_name': best_run['run_name'],
        'rougeL': float(best_run['metric_rougeL']),
        'params': {col.replace('param_', ''): best_run[col] for col in param_cols}
    }
    
    return best_model_info


def save_experiment_summary(
    df: pd.DataFrame,
    best_model_info: Dict,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Save comprehensive experiment summary report.
    
    Args:
        df: Comparison DataFrame
        best_model_info: Information about best model
        output_dir: Directory to save report
        logger: Logger instance
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save CSV
    csv_path = output_dir / f'experiment_comparison_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"\n✓ Experiment data saved: {csv_path}")
    
    # Save summary text
    summary_path = output_dir / f'experiment_summary_{timestamp}.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENT TRACKING SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Experiments: {len(df)}\n\n")
        
        if best_model_info:
            f.write("BEST MODEL:\n")
            f.write(f"  Run: {best_model_info['run_name']}\n")
            f.write(f"  Run ID: {best_model_info['run_id']}\n")
            f.write(f"  ROUGE-L: {best_model_info['rougeL']:.4f}\n\n")
            
            if best_model_info.get('params'):
                f.write("BEST HYPERPARAMETERS:\n")
                for param, value in best_model_info['params'].items():
                    f.write(f"  {param}: {value}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("ALL EXPERIMENTS:\n")
        f.write("=" * 80 + "\n\n")
        f.write(df.to_string(index=False))
    
    logger.info(f"✓ Summary saved: {summary_path}")


def main():
    """
    Main experiment tracking pipeline.
    
    Orchestrates experiment analysis:
    1. Connect to MLflow tracking server
    2. Load all experiment runs
    3. Create comparison table
    4. Generate visualizations
    5. Identify best model
    6. Save comprehensive summary
    """
    parser = argparse.ArgumentParser(
        description="Track and compare MLflow experiments"
    )
    parser.add_argument(
        '--tracking-uri',
        type=str,
        default='model-development/logs/mlruns',
        help='MLflow tracking URI'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='biobart_hyperparameter_tuning',
        help='Name of MLflow experiment to analyze'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='model-development/experiment_reports',
        help='Directory to save reports'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    
    # Set up logging
    logger = setup_logging(log_dir, args.log_level)
    logger.info("=" * 80)
    logger.info("EXPERIMENT TRACKING PIPELINE - Lab Lens Project")
    logger.info("=" * 80)
    logger.info(f"Tracking URI: {args.tracking_uri}")
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load experiments
        runs_data = load_experiments(args.tracking_uri, args.experiment_name, logger)
        
        if not runs_data:
            logger.warning("\n⚠️ No experiment runs found!")
            logger.info("\nTo populate MLflow with experiments:")
            logger.info("  1. Run hyperparameter tuning:")
            logger.info("     python scripts/hyperparameter_tuning.py --search-type quick")
            logger.info("  2. Or run training:")
            logger.info("     python scripts/train_biobart.py")
            logger.info("\nThen run this script again to see comparisons.")
            return 0
        
        # Create comparison table
        comparison_df = create_comparison_table(runs_data, logger)
        
        # Generate visualizations
        generate_comparison_plots(comparison_df, output_dir, logger)
        
        # Identify best model
        best_model = identify_best_model(comparison_df, logger)
        
        # Save summary
        save_experiment_summary(comparison_df, best_model, output_dir, logger)
        
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT TRACKING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Reports saved to: {output_dir}")
        logger.info("\nTo view experiments interactively:")
        logger.info(f"  mlflow ui --backend-store-uri {args.tracking_uri}")
        logger.info("  Open: http://localhost:5000")
        
        return 0
        
    except Exception as e:
        logger.error(f"Experiment tracking failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    sys.exit(main())