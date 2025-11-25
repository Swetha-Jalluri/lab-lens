"""
Model Validation Script for BioBART Medical Summarization
Author: Lab Lens Team
Description: Comprehensive evaluation of trained BioBART model on test set
             Computes multiple metrics (ROUGE, BLEU, BERTScore) for validation
             
Required by Rubric: Section 2.3 - Model Validation

This script:
1. Loads trained BioBART model
2. Generates predictions on held-out test dataset
3. Computes comprehensive evaluation metrics:
   - ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
   - BLEU score
   - BERTScore (precision, recall, F1)
   - Summary statistics (length, compression ratio)
4. Generates validation report with visualizations
5. Saves detailed predictions for error analysis
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Data processing
import pandas as pd
import numpy as np

# Model and tokenization
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Evaluation metrics
import evaluate
from rouge_score import rouge_scorer

# Try to import BERTScore (optional but recommended)
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("Warning: bert-score not available. Install with: pip install bert-score")


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging to both file and console.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"validation_{timestamp}.log"
    
    logger = logging.getLogger("ModelValidation")
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
    Load model configuration from JSON file.
    
    Args:
        config_path: Path to model_config.json
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration: {e}")


def load_trained_model(model_path: str, device: str, logger: logging.Logger):
    """
    Load trained BioBART model and tokenizer from checkpoint.
    
    Args:
        model_path: Path to saved model directory
        device: Device to load model on (cpu/cuda/mps)
        logger: Logger instance
        
    Returns:
        Tuple of (model, tokenizer, device)
    """
    logger.info(f"Loading trained model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    model.eval()  # Set to evaluation mode (important!)
    
    logger.info(f"✓ Model loaded successfully on {device}")
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer, device


def load_test_data(data_path: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Load test dataset for validation.
    
    Args:
        data_path: Path to test CSV file
        logger: Logger instance
        
    Returns:
        Test DataFrame with required columns
    """
    logger.info(f"Loading test data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"✓ Loaded {len(df)} test samples")
    
    # Validate required columns
    required_cols = ['input_text', 'target_summary']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def generate_predictions(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    device: str,
    generation_config: Dict,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Generate model predictions for all test samples.
    
    Args:
        model: Trained BioBART model
        tokenizer: Model tokenizer
        test_df: Test DataFrame
        device: Device for inference
        generation_config: Generation configuration (beam size, length, etc.)
        logger: Logger instance
        
    Returns:
        DataFrame with original data plus 'predicted_summary' column
    """
    logger.info("Generating predictions on test set...")
    logger.info(f"  Generation config: beams={generation_config.get('num_beams', 4)}, "
                f"max_length={generation_config.get('max_summary_length', 150)}")
    
    predictions = []
    
    # Generate predictions in batches for efficiency
    batch_size = 8
    num_batches = (len(test_df) + batch_size - 1) // batch_size
    
    for i in range(0, len(test_df), batch_size):
        batch_texts = test_df.iloc[i:i+batch_size]['input_text'].tolist()
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors='pt'
        ).to(device)
        
        # Generate summaries
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=generation_config.get('max_summary_length', 150),
                num_beams=generation_config.get('num_beams', 4),
                length_penalty=generation_config.get('length_penalty', 1.0),
                early_stopping=generation_config.get('early_stopping', True),
                no_repeat_ngram_size=generation_config.get('no_repeat_ngram_size', 3)
            )
        
        # Decode predictions
        batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_predictions)
        
        # Log progress every 10 batches
        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"  Progress: {i + len(batch_texts)}/{len(test_df)} samples")
    
    # Add predictions to dataframe
    result_df = test_df.copy()
    result_df['predicted_summary'] = predictions
    
    logger.info(f"✓ Generated {len(predictions)} predictions")
    return result_df


def compute_rouge_metrics(references: List[str], predictions: List[str], logger: logging.Logger) -> Dict:
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures n-gram overlap
    between generated summaries and reference summaries.
    
    Args:
        references: List of reference summaries
        predictions: List of predicted summaries
        logger: Logger instance
        
    Returns:
        Dictionary with ROUGE scores
    """
    logger.info("Computing ROUGE scores...")
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    results = {
        'rouge1_mean': np.mean(rouge1_scores),
        'rouge1_std': np.std(rouge1_scores),
        'rouge1_median': np.median(rouge1_scores),
        'rouge2_mean': np.mean(rouge2_scores),
        'rouge2_std': np.std(rouge2_scores),
        'rouge2_median': np.median(rouge2_scores),
        'rougeL_mean': np.mean(rougeL_scores),
        'rougeL_std': np.std(rougeL_scores),
        'rougeL_median': np.median(rougeL_scores),
    }
    
    logger.info("✓ ROUGE scores computed:")
    logger.info(f"  ROUGE-1: {results['rouge1_mean']:.4f} (±{results['rouge1_std']:.4f})")
    logger.info(f"  ROUGE-2: {results['rouge2_mean']:.4f} (±{results['rouge2_std']:.4f})")
    logger.info(f"  ROUGE-L: {results['rougeL_mean']:.4f} (±{results['rougeL_std']:.4f})")
    
    return results


def compute_bleu_score(references: List[str], predictions: List[str], logger: logging.Logger) -> Dict:
    """
    Compute BLEU score.
    
    BLEU (Bilingual Evaluation Understudy) measures precision of n-grams
    between generated and reference text.
    
    Args:
        references: List of reference summaries
        predictions: List of predicted summaries
        logger: Logger instance
        
    Returns:
        Dictionary with BLEU score
    """
    logger.info("Computing BLEU score...")
    
    try:
        # Load BLEU metric from HuggingFace evaluate
        bleu = evaluate.load('bleu')
        
        # Format references as list of lists (BLEU can handle multiple references)
        formatted_refs = [[ref] for ref in references]
        
        # Compute BLEU
        results = bleu.compute(predictions=predictions, references=formatted_refs)
        
        bleu_score = results['bleu']
        
        logger.info(f"✓ BLEU score computed: {bleu_score:.4f}")
        
        return {
            'bleu': bleu_score,
            'bleu_precisions': results.get('precisions', [])
        }
    
    except Exception as e:
        logger.warning(f"Failed to compute BLEU: {e}")
        return {'bleu': 0.0, 'bleu_precisions': []}


def compute_bertscore(references: List[str], predictions: List[str], logger: logging.Logger) -> Dict:
    """
    Compute BERTScore.
    
    BERTScore measures semantic similarity using BERT embeddings.
    More robust to paraphrasing than n-gram based metrics.
    
    Args:
        references: List of reference summaries
        predictions: List of predicted summaries
        logger: Logger instance
        
    Returns:
        Dictionary with BERTScore precision, recall, and F1
    """
    if not BERTSCORE_AVAILABLE:
        logger.warning("BERTScore not available, skipping...")
        return {'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0}
    
    logger.info("Computing BERTScore (this may take a few minutes)...")
    
    try:
        # Compute BERTScore
        # Using distilbert-base-uncased for speed
        P, R, F1 = bert_score(
            predictions,
            references,
            lang='en',
            model_type='distilbert-base-uncased',
            verbose=False
        )
        
        results = {
            'bertscore_precision': P.mean().item(),
            'bertscore_recall': R.mean().item(),
            'bertscore_f1': F1.mean().item()
        }
        
        logger.info("✓ BERTScore computed:")
        logger.info(f"  Precision: {results['bertscore_precision']:.4f}")
        logger.info(f"  Recall: {results['bertscore_recall']:.4f}")
        logger.info(f"  F1: {results['bertscore_f1']:.4f}")
        
        return results
    
    except Exception as e:
        logger.warning(f"Failed to compute BERTScore: {e}")
        return {'bertscore_precision': 0.0, 'bertscore_recall': 0.0, 'bertscore_f1': 0.0}


def compute_summary_statistics(test_df: pd.DataFrame, logger: logging.Logger) -> Dict:
    """
    Compute summary-level statistics.
    
    Args:
        test_df: DataFrame with predictions
        logger: Logger instance
        
    Returns:
        Dictionary with summary statistics
    """
    logger.info("Computing summary statistics...")
    
    # Calculate lengths
    input_lengths = test_df['input_text'].str.len()
    reference_lengths = test_df['target_summary'].str.len()
    prediction_lengths = test_df['predicted_summary'].str.len()
    
    # Calculate word counts
    input_word_counts = test_df['input_text'].str.split().str.len()
    reference_word_counts = test_df['target_summary'].str.split().str.len()
    prediction_word_counts = test_df['predicted_summary'].str.split().str.len()
    
    # Calculate compression ratios
    compression_ratios = input_lengths / prediction_lengths
    
    results = {
        'avg_input_length': input_lengths.mean(),
        'avg_reference_length': reference_lengths.mean(),
        'avg_prediction_length': prediction_lengths.mean(),
        'avg_input_words': input_word_counts.mean(),
        'avg_reference_words': reference_word_counts.mean(),
        'avg_prediction_words': prediction_word_counts.mean(),
        'avg_compression_ratio': compression_ratios.mean(),
        'median_compression_ratio': compression_ratios.median()
    }
    
    logger.info("✓ Summary statistics:")
    logger.info(f"  Avg input length: {results['avg_input_length']:.0f} chars ({results['avg_input_words']:.0f} words)")
    logger.info(f"  Avg prediction length: {results['avg_prediction_length']:.0f} chars ({results['avg_prediction_words']:.0f} words)")
    logger.info(f"  Avg compression ratio: {results['avg_compression_ratio']:.1f}x")
    
    return results


def perform_validation(
    predictions_df: pd.DataFrame,
    config: Dict,
    logger: logging.Logger
) -> Dict:
    """
    Perform comprehensive validation with multiple metrics.
    
    This is the main validation function that computes all required metrics
    as specified in the rubric.
    
    Args:
        predictions_df: DataFrame with predictions
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Dictionary containing all validation metrics
    """
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE MODEL VALIDATION")
    logger.info("=" * 80)
    
    validation_results = {
        'test_samples': len(predictions_df),
        'timestamp': datetime.now().isoformat()
    }
    
    # Extract references and predictions
    references = predictions_df['target_summary'].tolist()
    predictions = predictions_df['predicted_summary'].tolist()
    
    # Compute ROUGE scores
    rouge_results = compute_rouge_metrics(references, predictions, logger)
    validation_results.update(rouge_results)
    
    # Compute BLEU score
    bleu_results = compute_bleu_score(references, predictions, logger)
    validation_results.update(bleu_results)
    
    # Compute BERTScore (if available)
    bertscore_results = compute_bertscore(references, predictions, logger)
    validation_results.update(bertscore_results)
    
    # Compute summary statistics
    stats_results = compute_summary_statistics(predictions_df, logger)
    validation_results.update(stats_results)
    
    return validation_results


def save_validation_report(
    validation_results: Dict,
    predictions_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Save comprehensive validation report with all metrics and predictions.
    
    Args:
        validation_results: Dictionary of validation metrics
        predictions_df: DataFrame with all predictions
        output_dir: Directory to save reports
        logger: Logger instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON report (for programmatic access)
    report_path = output_dir / f"validation_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    logger.info(f"\n✓ Validation report saved: {report_path}")
    
    # Save detailed predictions CSV (for error analysis)
    predictions_path = output_dir / f"predictions_{timestamp}.csv"
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"✓ Predictions saved: {predictions_path}")
    
    # Create human-readable summary
    summary_path = output_dir / f"validation_summary_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL VALIDATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Test Samples: {validation_results['test_samples']}\n\n")
        
        f.write("ROUGE SCORES:\n")
        f.write(f"  ROUGE-1: {validation_results['rouge1_mean']:.4f} ± {validation_results['rouge1_std']:.4f}\n")
        f.write(f"  ROUGE-2: {validation_results['rouge2_mean']:.4f} ± {validation_results['rouge2_std']:.4f}\n")
        f.write(f"  ROUGE-L: {validation_results['rougeL_mean']:.4f} ± {validation_results['rougeL_std']:.4f}\n\n")
        
        f.write("OTHER METRICS:\n")
        f.write(f"  BLEU: {validation_results.get('bleu', 0):.4f}\n")
        
        if validation_results.get('bertscore_f1', 0) > 0:
            f.write(f"  BERTScore F1: {validation_results['bertscore_f1']:.4f}\n")
            f.write(f"  BERTScore Precision: {validation_results['bertscore_precision']:.4f}\n")
            f.write(f"  BERTScore Recall: {validation_results['bertscore_recall']:.4f}\n")
        
        f.write("\nSUMMARY STATISTICS:\n")
        f.write(f"  Avg Input: {validation_results['avg_input_length']:.0f} chars ({validation_results['avg_input_words']:.0f} words)\n")
        f.write(f"  Avg Output: {validation_results['avg_prediction_length']:.0f} chars ({validation_results['avg_prediction_words']:.0f} words)\n")
        f.write(f"  Compression Ratio: {validation_results['avg_compression_ratio']:.1f}x\n")
    
    logger.info(f"✓ Summary saved: {summary_path}")


def main():
    """
    Main validation pipeline.
    
    Orchestrates the complete validation process:
    1. Load configuration and trained model
    2. Load test data
    3. Generate predictions
    4. Compute comprehensive metrics (ROUGE, BLEU, BERTScore)
    5. Save validation report
    """
    parser = argparse.ArgumentParser(
        description="Validate BioBART model with comprehensive metrics"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='model-development/configs/model_config.json',
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained model (overrides config)'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        default=None,
        help='Path to test data CSV (overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='model-development/validation_reports',
        help='Directory to save validation reports'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device for inference'
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
    
    # Get config file directory for path resolution
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    
    def resolve_path(path_str: str) -> Path:
        """Resolve path relative to config directory"""
        path = Path(path_str)
        if not path.is_absolute():
            path = config_dir / path
        return path.resolve()
    
    # Set up paths
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        # Use the actual trained model location
        model_path = resolve_path("../models/biobart_summarization")
        if not model_path.exists():
            model_path = resolve_path(config['model_config']['output_model_path'])
    
    if args.test_data:
        test_data_path = Path(args.test_data)
    else:
        test_data_path = resolve_path("../data/model_ready/test.csv")
    
    output_dir = Path(args.output_dir)
    log_dir = output_dir / "logs"
    
    # Set up logging
    logger = setup_logging(log_dir, args.log_level)
    logger.info("=" * 80)
    logger.info("MODEL VALIDATION PIPELINE - Lab Lens Project")
    logger.info("=" * 80)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Test data: {test_data_path}")
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
        # Load trained model
        model, tokenizer, device = load_trained_model(str(model_path), device, logger)
        
        # Load test data
        test_df = load_test_data(str(test_data_path), logger)
        
        # Generate predictions
        predictions_df = generate_predictions(
            model,
            tokenizer,
            test_df,
            device,
            generation_config=config.get('generation_config', {}),
            logger=logger
        )
        
        # Perform comprehensive validation
        validation_results = perform_validation(predictions_df, config, logger)
        
        # Save validation report
        save_validation_report(validation_results, predictions_df, output_dir, logger)
        
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Overall ROUGE-L: {validation_results['rougeL_mean']:.4f}")
        logger.info(f"Overall BLEU: {validation_results.get('bleu', 0):.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    sys.exit(main())