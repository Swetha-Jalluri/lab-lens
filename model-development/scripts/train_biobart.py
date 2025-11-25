"""
BioBART Training Script for Lab-Lens Medical Summarization
Author: Lab Lens Team
Description: Production training script for fine-tuning BioBART on MIMIC-III discharge summaries
             Loads processed data from data pipeline, trains model, logs to MLflow, saves checkpoints

This script converts Swetha's notebook into a production-ready training pipeline.
Integrates with: Data Pipeline -> Model Training -> MLflow -> Model Registry
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Core dependencies
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Transformers and HuggingFace
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)

# Evaluation metrics
import evaluate

# MLflow for experiment tracking
import mlflow
import mlflow.pytorch

# Set up logging
def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging to both file and console.
    Creates timestamped log files in the specified directory.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("BioBART_Training")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # File handler with detailed formatting
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler with simpler formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_config(config_path: str) -> Dict:
    """
    Load training configuration from JSON file.
    This centralizes all hyperparameters and paths.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")


class MIMICSummarizationDataset(Dataset):
    """
    PyTorch Dataset for MIMIC-III discharge summary summarization.
    Converts raw text into tokenized format for BioBART model.
    
    This class handles:
    - Loading preprocessed data from CSV
    - Tokenizing input discharge summaries
    - Tokenizing target summaries
    - Creating proper input format for sequence-to-sequence models
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 150,
        input_column: str = "text",
        target_column: str = "summary"
    ):
        """
        Initialize the dataset.
        
        Args:
            dataframe: Pandas DataFrame with discharge summaries
            tokenizer: HuggingFace tokenizer for BioBART
            max_input_length: Maximum tokens for input text
            max_target_length: Maximum tokens for target summary
            input_column: Name of column containing input text
            target_column: Name of column containing target summaries
        """
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.input_column = input_column
        self.target_column = target_column
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing input_ids, attention_mask, and labels
        """
        row = self.data.iloc[idx]
        
        # Get input text (discharge summary) and target summary
        input_text = str(row[self.input_column])
        target_text = str(row[self.target_column])
        
        # Tokenize input text
        # This converts text to token IDs that the model can process
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target summary
        # Labels are what the model should learn to generate
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels (replace padding token ids with -100 so they're ignored in loss)
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels
        }


def load_and_prepare_data(
    data_path: str,
    config: Dict,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load processed data from the data pipeline and split into train/val/test.
    
    This function:
    - Loads the CSV file created by the data pipeline
    - Validates required columns exist
    - Performs stratified split to maintain demographic balance
    - Returns three DataFrames for training, validation, and testing
    
    Args:
        data_path: Path to processed CSV file from data pipeline
        config: Configuration dictionary
        logger: Logger instance for tracking
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info(f"Loading data from: {data_path}")
    
    # Load the processed data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples from data pipeline")
    
    # Get data configuration
    data_config = config.get('data_config', {})
    train_split = data_config.get('train_split', 0.70)
    val_split = data_config.get('validation_split', 0.15)
    test_split = data_config.get('test_split', 0.15)
    random_seed = data_config.get('random_seed', 42)
    
    # Validate required columns
    required_columns = ['input_text', 'target_summary']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Filter incomplete records if configured
    if data_config.get('filter_incomplete_records', True):
        initial_len = len(df)
        df = df.dropna(subset=['input_text', 'target_summary'])
        df = df[df['input_text'].str.len() >= data_config.get('min_input_length', 100)]
        logger.info(f"Filtered {initial_len - len(df)} incomplete records")
    
    # Perform stratified split if demographic data available
    stratify_column = data_config.get('stratify_by', None)
    stratify_data = None
    
    if stratify_column and stratify_column in df.columns:
        stratify_data = df[stratify_column]
        logger.info(f"Using stratified split by: {stratify_column}")
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_split,
        random_state=random_seed,
        stratify=stratify_data
    )
    
    # Second split: separate train and validation
    relative_val_size = val_split / (train_split + val_split)
    
    if stratify_data is not None:
        stratify_train_val = train_val_df[stratify_column]
    else:
        stratify_train_val = None
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=random_seed,
        stratify=stratify_train_val
    )
    
    logger.info(f"Data split complete:")
    logger.info(f"  Training samples: {len(train_df)}")
    logger.info(f"  Validation samples: {len(val_df)}")
    logger.info(f"  Test samples: {len(test_df)}")
    
    return train_df, val_df, test_df


def compute_metrics(eval_pred, tokenizer):
    """
    Compute evaluation metrics for summarization.
    
    This function calculates:
    - ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) for n-gram overlap
    - Average summary length
    
    Args:
        eval_pred: Tuple of (predictions, labels) from the model
        tokenizer: Tokenizer to decode predictions back to text
        
    Returns:
        Dictionary of computed metrics
    """
    # Load ROUGE metric
    rouge = evaluate.load('rouge')
    
    predictions, labels = eval_pred
    
    # Decode predictions from token IDs back to text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels with pad token (they were masked during training)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute ROUGE scores
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    # Extract key metrics and convert to percentages
    result = {
        'rouge1': result['rouge1'] * 100,
        'rouge2': result['rouge2'] * 100,
        'rougeL': result['rougeL'] * 100
    }
    
    # Add average summary length
    prediction_lens = [len(pred.split()) for pred in decoded_preds]
    result['avg_summary_length'] = np.mean(prediction_lens)
    
    return result


def initialize_mlflow(config: Dict, logger: logging.Logger):
    """
    Initialize MLflow for experiment tracking.
    
    Sets up MLflow to track:
    - Training parameters (learning rate, batch size, etc.)
    - Performance metrics (ROUGE scores, loss)
    - Model artifacts (checkpoints)
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    mlflow_config = config.get('mlflow_config', {})
    
    if not mlflow_config.get('tracking_enabled', True):
        logger.info("MLflow tracking is disabled")
        return
    
    # Set tracking URI (where MLflow stores experiment data)
    tracking_uri = config['model_config'].get('mlflow_tracking_uri', 'mlruns')
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set experiment name
    experiment_name = config['model_config'].get('experiment_name', 'biobart_training')
    mlflow.set_experiment(experiment_name)
    
    logger.info(f"MLflow initialized:")
    logger.info(f"  Tracking URI: {tracking_uri}")
    logger.info(f"  Experiment: {experiment_name}")


def train_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    model,
    tokenizer,
    config: Dict,
    logger: logging.Logger,
    output_dir: Path
) -> Seq2SeqTrainer:
    """
    Train the BioBART model using HuggingFace Trainer.
    
    This is the main training loop that:
    - Configures training hyperparameters
    - Sets up evaluation strategy
    - Implements early stopping
    - Saves checkpoints
    - Logs metrics to MLflow
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        model: BioBART model to train
        tokenizer: Tokenizer for the model
        config: Configuration dictionary
        logger: Logger instance
        output_dir: Directory to save model checkpoints
        
    Returns:
        Trained Seq2SeqTrainer instance
    """
    logger.info("Setting up training configuration...")
    
    # Get training configuration
    train_config = config.get('training_config', {})
    
    # Create training arguments
    # These control all aspects of the training process
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        
        # Training hyperparameters
        num_train_epochs=train_config.get('num_epochs', 5),
        per_device_train_batch_size=train_config.get('batch_size', 8),
        per_device_eval_batch_size=train_config.get('batch_size', 8),
        learning_rate=train_config.get('learning_rate', 2e-5),
        weight_decay=train_config.get('weight_decay', 0.01),
        warmup_steps=train_config.get('warmup_steps', 500),
        
        # Gradient accumulation (effective batch size = batch_size * gradient_accumulation_steps)
        gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 2),
        max_grad_norm=train_config.get('max_grad_norm', 1.0),
        
        # Mixed precision training (fp16) for faster training on GPU
        # Set to False for CPU or if you encounter stability issues
        fp16=train_config.get('fp16', False) and torch.cuda.is_available(),
        
        # Evaluation and saving strategy
        eval_strategy=train_config.get('evaluation_strategy', 'epoch'),
        save_strategy=train_config.get('save_strategy', 'epoch'),
        save_total_limit=train_config.get('save_total_limit', 2),
        load_best_model_at_end=train_config.get('load_best_model_at_end', True),
        metric_for_best_model=train_config.get('metric_for_best_model', 'rougeL'),
        greater_is_better=train_config.get('greater_is_better', True),
        
        # Logging
        logging_dir=str(output_dir / "logs"),
        logging_steps=train_config.get('logging_steps', 100),
        logging_first_step=True,
        
        # Generation settings for validation
        predict_with_generate=True,
        generation_max_length=config.get('data_config', {}).get('max_target_length', 150),
        generation_num_beams=config.get('generation_config', {}).get('num_beams', 4),
        
        # Reproducibility
        seed=config.get('data_config', {}).get('random_seed', 42),
        
        # Other settings
        report_to=["mlflow"] if config.get('mlflow_config', {}).get('tracking_enabled', True) else [],
        push_to_hub=False,
        remove_unused_columns=False,
    )
    
    # Data collator (handles batching and padding)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Create callbacks for early stopping
    early_stopping_patience = train_config.get('early_stopping_patience', 2)
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
    ]
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
        callbacks=callbacks
    )
    
    logger.info("Starting training...")
    logger.info(f"  Number of training epochs: {training_args.num_train_epochs}")
    logger.info(f"  Batch size per device: {training_args.per_device_train_batch_size}")
    logger.info(f"  Total optimization steps: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")
    
    # Start training
    train_result = trainer.train()
    
    logger.info("Training completed!")
    logger.info(f"  Final training loss: {train_result.training_loss:.4f}")
    
    return trainer


def save_model_and_artifacts(
    trainer: Seq2SeqTrainer,
    tokenizer,
    config: Dict,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Save the trained model, tokenizer, and training artifacts.
    
    This function saves:
    - Model weights and configuration
    - Tokenizer files
    - Training statistics
    - Configuration used for training
    
    Args:
        trainer: Trained Seq2SeqTrainer
        tokenizer: Tokenizer used
        config: Configuration dictionary
        output_dir: Directory to save artifacts
        logger: Logger instance
    """
    logger.info(f"Saving model to: {output_dir}")
    
    # Save model and tokenizer
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save training configuration
    config_save_path = output_dir / "training_config.json"
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Model and artifacts saved successfully")
    
    # Log to MLflow if enabled
    mlflow_config = config.get('mlflow_config', {})
    if mlflow_config.get('log_model', True) and mlflow.active_run():
        logger.info("Logging model to MLflow...")
        mlflow.pytorch.log_model(trainer.model, "model")
        mlflow.log_artifact(str(config_save_path))


def main():
    """
    Main training pipeline.
    
    This orchestrates the entire training process:
    1. Parse command-line arguments
    2. Load configuration
    3. Set up logging
    4. Initialize MLflow
    5. Load data
    6. Initialize model and tokenizer
    7. Create datasets
    8. Train model
    9. Save results
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train BioBART model for medical summarization"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='model-development/configs/model_config.json',
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to processed data CSV (overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for model (overrides config)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to use for training'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override config with command-line arguments if provided
    if args.data_path:
        config['model_config']['input_data_path'] = args.data_path
    if args.output_dir:
        config['model_config']['output_model_path'] = args.output_dir
    
    # Set up output directories
    output_dir = Path(config['model_config']['output_model_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(config['model_config']['logs_path'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(log_dir, args.log_level)
    logger.info("=" * 80)
    logger.info("BioBART Training Pipeline - Lab Lens Project")
    logger.info("=" * 80)
    logger.info(f"Configuration: {args.config}")
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
        initialize_mlflow(config, logger)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"biobart_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log configuration parameters
            mlflow.log_params({
                'model_name': config['model_config']['model_name'],
                'learning_rate': config['training_config']['learning_rate'],
                'batch_size': config['training_config']['batch_size'],
                'num_epochs': config['training_config']['num_epochs'],
                'max_input_length': config['data_config']['max_input_length'],
                'max_target_length': config['data_config']['max_target_length'],
            })
            
            # Load and prepare data
            data_path = config['model_config']['input_data_path']
            train_df, val_df, test_df = load_and_prepare_data(data_path, config, logger)
            
            # Initialize tokenizer and model
            logger.info("Loading BioBART model and tokenizer...")
            model_name = config['model_config']['model_name']
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.to(device)
            
            logger.info(f"Model loaded: {model_name}")
            logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Create datasets
            logger.info("Creating datasets...")
            train_dataset = MIMICSummarizationDataset(
                train_df,
                tokenizer,
                max_input_length=config['data_config']['max_input_length'],
                max_target_length=config['data_config']['max_target_length'],
                input_column="input_text",
                target_column="target_summary"
            )
            
            val_dataset = MIMICSummarizationDataset(
                val_df,
                tokenizer,
                max_input_length=config['data_config']['max_input_length'],
                max_target_length=config['data_config']['max_target_length'],
                input_column="input_text",
                target_column="target_summary"
            )
            
            # Train model
            trainer = train_model(
                train_dataset,
                val_dataset,
                model,
                tokenizer,
                config,
                logger,
                output_dir
            )
            
            # Evaluate on validation set
            logger.info("Running final evaluation...")
            eval_results = trainer.evaluate()
            
            logger.info("Evaluation results:")
            for key, value in eval_results.items():
                logger.info(f"  {key}: {value:.4f}")
                mlflow.log_metric(f"final_{key}", value)
            
            # Save model and artifacts
            save_model_and_artifacts(trainer, tokenizer, config, output_dir, logger)
            
            logger.info("=" * 80)
            logger.info("Training pipeline completed successfully!")
            logger.info(f"Model saved to: {output_dir}")
            logger.info("=" * 80)
            
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()