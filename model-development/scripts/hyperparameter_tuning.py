"""
Hyperparameter Tuning Script for BioBART Medical Summarization
Author: Lab Lens Team
Description: Systematic hyperparameter search to optimize model performance
             Tests different learning rates, batch sizes, and other parameters
             
Required by Rubric: Section 3 - Hyperparameter Tuning

This script:
1. Defines hyperparameter search space
2. Runs multiple training experiments with different configurations
3. Tracks all experiments in MLflow
4. Compares results across all configurations
5. Identifies best hyperparameter combination
6. Generates comparison visualizations
7. Saves best configuration for production use
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import itertools
import warnings
warnings.filterwarnings('ignore')

# Data processing
import pandas as pd
import numpy as np

# PyTorch and transformers
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)

# Evaluation
import evaluate

# MLflow for experiment tracking
import mlflow
import mlflow.pytorch

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Configure logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"hyperparameter_tuning_{timestamp}.log"
    
    logger = logging.getLogger("HyperparameterTuning")
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


def load_config(config_path: str) -> Dict:
    """Load model configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")


class MIMICSummarizationDataset(Dataset):
    """PyTorch Dataset for MIMIC-III summarization - simplified version."""
    
    def __init__(self, dataframe, tokenizer, max_input_length=512, max_target_length=150):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        input_text = str(row['input_text'])
        target_text = str(row['target_summary'])
        
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels
        }


def compute_metrics(eval_pred, tokenizer):
    """Compute ROUGE metrics for evaluation."""
    rouge = evaluate.load('rouge')
    predictions, labels = eval_pred
    
    # Convert predictions to numpy array and handle overflow
    # This fixes "out of range integral type conversion" error
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Ensure predictions are in valid range for int64
    predictions = np.array(predictions, dtype=np.int64)
    predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Handle labels similarly
    labels = np.array(labels, dtype=np.int64)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    return {
        'rouge1': result['rouge1'],
        'rouge2': result['rouge2'],
        'rougeL': result['rougeL']
    }


def define_search_space(search_type: str) -> Dict:
    """
    Define hyperparameter search space.
    
    Args:
        search_type: Type of search ('quick', 'medium', 'full')
        
    Returns:
        Dictionary with hyperparameter options
    """
    if search_type == 'quick':
        # Quick search - 4 combinations
        return {
            'learning_rate': [2e-5, 5e-5],
            'batch_size': [4, 8],
            'num_epochs': [3]  # Fixed for speed
        }
    
    elif search_type == 'medium':
        # Medium search - 12 combinations
        return {
            'learning_rate': [1e-5, 2e-5, 5e-5],
            'batch_size': [4, 8],
            'num_epochs': [3, 5]
        }
    
    else:  # full
        # Full search - 24 combinations
        return {
            'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
            'batch_size': [4, 8, 16],
            'num_epochs': [3, 5]
        }


def generate_hyperparameter_combinations(search_space: Dict) -> List[Dict]:
    """
    Generate all combinations of hyperparameters.
    
    Args:
        search_space: Dictionary with hyperparameter options
        
    Returns:
        List of hyperparameter dictionaries
    """
    keys = search_space.keys()
    values = search_space.values()
    
    combinations = []
    for combo in itertools.product(*values):
        param_dict = dict(zip(keys, combo))
        combinations.append(param_dict)
    
    return combinations


def train_with_hyperparameters(
    train_dataset: Dataset,
    val_dataset: Dataset,
    hyperparams: Dict,
    model_name: str,
    tokenizer,
    device: str,
    output_dir: Path,
    logger: logging.Logger
) -> Dict:
    """
    Train model with specific hyperparameter configuration.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        hyperparams: Hyperparameter configuration to test
        model_name: Base model name
        tokenizer: Model tokenizer
        device: Device for training
        output_dir: Output directory for this experiment
        logger: Logger instance
        
    Returns:
        Dictionary with training results and metrics
    """
    logger.info(f"Training with hyperparameters: {hyperparams}")
    
    # Initialize fresh model for each experiment
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    
    # Create training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=hyperparams['num_epochs'],
        per_device_train_batch_size=hyperparams['batch_size'],
        per_device_eval_batch_size=hyperparams['batch_size'],
        learning_rate=hyperparams['learning_rate'],
        weight_decay=0.01,
        warmup_steps=100,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='rougeL',
        greater_is_better=True,
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=150,
        fp16=False,
        report_to=["mlflow"],
        seed=42,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    train_result = trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    
    results = {
        'hyperparameters': hyperparams,
        'final_train_loss': train_result.training_loss,
        'final_eval_loss': eval_results['eval_loss'],
        'rouge1': eval_results['eval_rouge1'],
        'rouge2': eval_results['eval_rouge2'],
        'rougeL': eval_results['eval_rougeL'],
    }
    
    logger.info(f"  Results: ROUGE-L={results['rougeL']:.4f}, Loss={results['final_eval_loss']:.4f}")
    
    return results


def visualize_results(all_results: List[Dict], output_dir: Path, logger: logging.Logger):
    """
    Create visualizations comparing all hyperparameter experiments.
    
    Args:
        all_results: List of results from all experiments
        output_dir: Directory to save plots
        logger: Logger instance
    """
    logger.info("Creating comparison visualizations...")
    
    # Create DataFrame from results
    df = pd.DataFrame([
        {
            'lr': r['hyperparameters']['learning_rate'],
            'batch_size': r['hyperparameters']['batch_size'],
            'epochs': r['hyperparameters']['num_epochs'],
            'rougeL': r['rougeL'],
            'rouge1': r['rouge1'],
            'eval_loss': r['final_eval_loss']
        }
        for r in all_results
    ])
    
    # Plot 1: ROUGE-L vs Learning Rate
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ROUGE-L by learning rate
    ax = axes[0, 0]
    for bs in df['batch_size'].unique():
        subset = df[df['batch_size'] == bs]
        ax.plot(subset['lr'], subset['rougeL'], marker='o', label=f'Batch={bs}')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('ROUGE-L')
    ax.set_title('ROUGE-L vs Learning Rate')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    
    # ROUGE-L by batch size
    ax = axes[0, 1]
    for lr in df['lr'].unique():
        subset = df[df['lr'] == lr]
        ax.plot(subset['batch_size'], subset['rougeL'], marker='o', label=f'LR={lr:.0e}')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('ROUGE-L')
    ax.set_title('ROUGE-L vs Batch Size')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Loss by learning rate
    ax = axes[1, 0]
    for bs in df['batch_size'].unique():
        subset = df[df['batch_size'] == bs]
        ax.plot(subset['lr'], subset['eval_loss'], marker='o', label=f'Batch={bs}')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Learning Rate')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    
    # Heatmap of ROUGE-L
    ax = axes[1, 1]
    pivot = df.pivot_table(values='rougeL', index='lr', columns='batch_size', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax)
    ax.set_title('ROUGE-L Heatmap (LR vs Batch Size)')
    ax.set_ylabel('Learning Rate')
    ax.set_xlabel('Batch Size')
    
    plt.tight_layout()
    plot_path = output_dir / 'hyperparameter_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved comparison plot: {plot_path}")
    
    # Save results table
    results_path = output_dir / 'hyperparameter_results.csv'
    df.to_csv(results_path, index=False)
    logger.info(f"✓ Saved results table: {results_path}")


def main():
    """
    Main hyperparameter tuning pipeline.
    
    Orchestrates systematic hyperparameter search:
    1. Load configuration and data
    2. Define search space
    3. Generate all hyperparameter combinations
    4. Train model for each combination
    5. Track all experiments in MLflow
    6. Compare results and identify best configuration
    7. Save best configuration
    """
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for BioBART model"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='model-development/configs/model_config.json',
        help='Path to base configuration file'
    )
    parser.add_argument(
        '--train-data',
        type=str,
        default='model-development/data/model_ready/train.csv',
        help='Path to training data'
    )
    parser.add_argument(
        '--val-data',
        type=str,
        default='model-development/data/model_ready/validation.csv',
        help='Path to validation data'
    )
    parser.add_argument(
        '--search-type',
        type=str,
        default='quick',
        choices=['quick', 'medium', 'full'],
        help='Search space size (quick=4, medium=12, full=24 combinations)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='model-development/hyperparameter_tuning',
        help='Directory to save tuning results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device for training'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Set up paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    
    # Set up logging
    logger = setup_logging(log_dir, args.log_level)
    logger.info("=" * 80)
    logger.info("HYPERPARAMETER TUNING PIPELINE - Lab Lens Project")
    logger.info("=" * 80)
    logger.info(f"Search type: {args.search_type}")
    logger.info(f"Output directory: {output_dir}")
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    try:
        # Initialize MLflow
        mlflow_tracking_uri = str(output_dir / "mlruns")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("biobart_hyperparameter_tuning")
        
        logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")
        
        # Load data
        logger.info("Loading training and validation data...")
        train_df = pd.read_csv(args.train_data)
        val_df = pd.read_csv(args.val_data)
        
        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Validation samples: {len(val_df)}")
        
        # Load tokenizer
        model_name = config['model_config']['model_name']
        logger.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Define search space
        search_space = define_search_space(args.search_type)
        combinations = generate_hyperparameter_combinations(search_space)
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"HYPERPARAMETER SEARCH SPACE")
        logger.info(f"{'=' * 80}")
        logger.info(f"Learning rates: {search_space['learning_rate']}")
        logger.info(f"Batch sizes: {search_space['batch_size']}")
        logger.info(f"Epochs: {search_space['num_epochs']}")
        logger.info(f"Total combinations: {len(combinations)}")
        
        # Run experiments
        all_results = []
        
        for i, hyperparams in enumerate(combinations, 1):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"EXPERIMENT {i}/{len(combinations)}")
            logger.info(f"{'=' * 80}")
            
            # Create experiment-specific output directory
            exp_dir = output_dir / f"exp_{i}_lr{hyperparams['learning_rate']}_bs{hyperparams['batch_size']}"
            exp_dir.mkdir(exist_ok=True)
            
            # Start MLflow run
            with mlflow.start_run(run_name=f"exp_{i}"):
                # Log hyperparameters
                mlflow.log_params(hyperparams)
                
                # Create datasets
                train_dataset = MIMICSummarizationDataset(
                    train_df,
                    tokenizer,
                    max_input_length=config['data_config']['max_input_length'],
                    max_target_length=config['data_config']['max_target_length']
                )
                
                val_dataset = MIMICSummarizationDataset(
                    val_df,
                    tokenizer,
                    max_input_length=config['data_config']['max_input_length'],
                    max_target_length=config['data_config']['max_target_length']
                )
                
                # Train with these hyperparameters
                results = train_with_hyperparameters(
                    train_dataset,
                    val_dataset,
                    hyperparams,
                    model_name,
                    tokenizer,
                    device,
                    exp_dir,
                    logger
                )
                
                # Log metrics to MLflow
                mlflow.log_metric("rouge1", results['rouge1'])
                mlflow.log_metric("rouge2", results['rouge2'])
                mlflow.log_metric("rougeL", results['rougeL'])
                mlflow.log_metric("eval_loss", results['final_eval_loss'])
                mlflow.log_metric("train_loss", results['final_train_loss'])
                
                all_results.append(results)
        
        # Find best configuration
        logger.info(f"\n{'=' * 80}")
        logger.info("TUNING RESULTS SUMMARY")
        logger.info(f"{'=' * 80}")
        
        best_result = max(all_results, key=lambda x: x['rougeL'])
        
        logger.info(f"\nBest Configuration:")
        logger.info(f"  Learning Rate: {best_result['hyperparameters']['learning_rate']}")
        logger.info(f"  Batch Size: {best_result['hyperparameters']['batch_size']}")
        logger.info(f"  Epochs: {best_result['hyperparameters']['num_epochs']}")
        logger.info(f"\nBest Performance:")
        logger.info(f"  ROUGE-L: {best_result['rougeL']:.4f}")
        logger.info(f"  ROUGE-1: {best_result['rouge1']:.4f}")
        logger.info(f"  ROUGE-2: {best_result['rouge2']:.4f}")
        
        # Save best configuration
        best_config_path = output_dir / 'best_hyperparameters.json'
        with open(best_config_path, 'w') as f:
            json.dump(best_result['hyperparameters'], f, indent=2)
        logger.info(f"\n✓ Best configuration saved: {best_config_path}")
        
        # Save all results
        all_results_path = output_dir / 'all_results.json'
        with open(all_results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"✓ All results saved: {all_results_path}")
        
        # Create visualizations
        visualize_results(all_results, output_dir, logger)
        
        logger.info(f"\n{'=' * 80}")
        logger.info("HYPERPARAMETER TUNING COMPLETE")
        logger.info(f"{'=' * 80}")
        logger.info(f"Tested {len(combinations)} configurations")
        logger.info(f"Best ROUGE-L: {best_result['rougeL']:.4f}")
        logger.info(f"\nView experiments in MLflow:")
        logger.info(f"  mlflow ui --backend-store-uri {mlflow_tracking_uri}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    sys.exit(main())