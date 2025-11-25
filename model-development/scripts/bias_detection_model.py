"""
Bias Detection Script for BioBART Medical Summarization Model
Author: Lab Lens Team
Description: Detects and analyzes bias in model predictions across demographic groups
             Uses data slicing techniques to evaluate fairness across age, gender, and ethnicity
             
Required by Rubric: Section 6 - Model Bias Detection (Using Slicing Techniques)

This script:
1. Loads trained BioBART model
2. Generates predictions on test data
3. Slices data by demographic groups (age_group, gender, ethnicity)
4. Computes metrics (ROUGE, summary length) for each slice
5. Identifies performance disparities across groups
6. Generates comprehensive bias report with visualizations
7. Suggests mitigation strategies if bias detected
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

# Fairness analysis
try:
    from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    print("Warning: fairlearn not available. Using basic bias metrics only.")


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
    log_file = log_dir / f"bias_detection_{timestamp}.log"
    
    logger = logging.getLogger("BiasDetection")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_config(config_path: str) -> Dict:
    """
    Load configuration from JSON file.
    
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
    model.eval()  # Set to evaluation mode
    
    logger.info(f"Model loaded successfully on {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer, device


def load_test_data(data_path: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Load test dataset for bias detection.
    
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
    logger.info(f"Loaded {len(df)} test samples")
    
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
    max_length: int,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Generate model predictions for all test samples.
    
    Args:
        model: Trained BioBART model
        tokenizer: Model tokenizer
        test_df: Test DataFrame
        device: Device for inference
        max_length: Maximum summary length
        logger: Logger instance
        
    Returns:
        DataFrame with original data plus 'predicted_summary' column
    """
    logger.info("Generating predictions for test set...")
    
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
                max_length=max_length,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        # Decode predictions
        batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_predictions)
        
        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"  Processed {i + len(batch_texts)}/{len(test_df)} samples")
    
    # Add predictions to dataframe
    result_df = test_df.copy()
    result_df['predicted_summary'] = predictions
    
    logger.info(f"Generated {len(predictions)} predictions")
    return result_df


def compute_rouge_scores(reference: str, prediction: str, scorer) -> Dict[str, float]:
    """
    Compute ROUGE scores for a single prediction.
    
    Args:
        reference: Ground truth summary
        prediction: Model-generated summary
        scorer: Rouge scorer instance
        
    Returns:
        Dictionary of ROUGE scores
    """
    scores = scorer.score(reference, prediction)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


def compute_metrics_for_slice(
    df: pd.DataFrame,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Compute performance metrics for a data slice.
    
    Args:
        df: DataFrame slice
        logger: Logger instance
        
    Returns:
        Dictionary of computed metrics
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    summary_lengths = []
    compression_ratios = []
    
    for _, row in df.iterrows():
        # Compute ROUGE scores
        scores = compute_rouge_scores(
            row['target_summary'],
            row['predicted_summary'],
            scorer
        )
        rouge1_scores.append(scores['rouge1'])
        rouge2_scores.append(scores['rouge2'])
        rougeL_scores.append(scores['rougeL'])
        
        # Compute summary characteristics
        pred_len = len(row['predicted_summary'])
        input_len = len(row['input_text'])
        summary_lengths.append(pred_len)
        compression_ratios.append(input_len / pred_len if pred_len > 0 else 0)
    
    return {
        'sample_count': len(df),
        'rouge1_mean': np.mean(rouge1_scores),
        'rouge1_std': np.std(rouge1_scores),
        'rouge2_mean': np.mean(rouge2_scores),
        'rouge2_std': np.std(rouge2_scores),
        'rougeL_mean': np.mean(rougeL_scores),
        'rougeL_std': np.std(rougeL_scores),
        'avg_summary_length': np.mean(summary_lengths),
        'avg_compression_ratio': np.mean(compression_ratios)
    }


def perform_bias_detection(
    predictions_df: pd.DataFrame,
    config: Dict,
    logger: logging.Logger
) -> Dict:
    """
    Perform comprehensive bias detection across demographic slices.
    
    This is the core function that implements data slicing techniques
    to detect bias as required by the rubric.
    
    Args:
        predictions_df: DataFrame with predictions and demographics
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Dictionary containing bias analysis results
    """
    logger.info("=" * 80)
    logger.info("PERFORMING BIAS DETECTION USING DATA SLICING")
    logger.info("=" * 80)
    
    bias_config = config.get('bias_detection_config', {})
    slice_by = bias_config.get('slice_by', ['age_group', 'gender', 'ethnicity_clean'])
    min_samples = bias_config.get('min_samples_per_slice', 30)
    alert_threshold = bias_config.get('alert_threshold', 0.05)
    
    bias_results = {
        'overall_metrics': {},
        'slice_metrics': {},
        'disparities': {},
        'bias_detected': False,
        'recommendations': []
    }
    
    # Compute overall metrics (baseline)
    logger.info("\nComputing overall baseline metrics...")
    overall_metrics = compute_metrics_for_slice(predictions_df, logger)
    bias_results['overall_metrics'] = overall_metrics
    
    logger.info(f"Overall Performance:")
    logger.info(f"  Samples: {overall_metrics['sample_count']}")
    logger.info(f"  ROUGE-1: {overall_metrics['rouge1_mean']:.4f} (±{overall_metrics['rouge1_std']:.4f})")
    logger.info(f"  ROUGE-2: {overall_metrics['rouge2_mean']:.4f} (±{overall_metrics['rouge2_std']:.4f})")
    logger.info(f"  ROUGE-L: {overall_metrics['rougeL_mean']:.4f} (±{overall_metrics['rougeL_std']:.4f})")
    
    # Perform slicing analysis for each demographic attribute
    for slice_attr in slice_by:
        if slice_attr not in predictions_df.columns:
            logger.warning(f"Slice attribute '{slice_attr}' not found in data, skipping...")
            continue
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"ANALYZING BIAS BY: {slice_attr.upper()}")
        logger.info(f"{'=' * 80}")
        
        slice_results = {}
        
        # Get unique values for this attribute
        unique_values = predictions_df[slice_attr].dropna().unique()
        logger.info(f"Found {len(unique_values)} unique values: {list(unique_values)}")
        
        # Compute metrics for each slice
        for value in unique_values:
            slice_df = predictions_df[predictions_df[slice_attr] == value]
            
            # Skip slices with too few samples
            if len(slice_df) < min_samples:
                logger.info(f"  {value}: {len(slice_df)} samples (< {min_samples} minimum, skipping)")
                continue
            
            # Compute metrics for this slice
            slice_metrics = compute_metrics_for_slice(slice_df, logger)
            slice_results[str(value)] = slice_metrics
            
            logger.info(f"\n  Group: {value}")
            logger.info(f"    Samples: {slice_metrics['sample_count']}")
            logger.info(f"    ROUGE-1: {slice_metrics['rouge1_mean']:.4f} (±{slice_metrics['rouge1_std']:.4f})")
            logger.info(f"    ROUGE-2: {slice_metrics['rouge2_mean']:.4f} (±{slice_metrics['rouge2_std']:.4f})")
            logger.info(f"    ROUGE-L: {slice_metrics['rougeL_mean']:.4f} (±{slice_metrics['rougeL_std']:.4f})")
            logger.info(f"    Avg Summary Length: {slice_metrics['avg_summary_length']:.1f}")
        
        bias_results['slice_metrics'][slice_attr] = slice_results
        
        # Detect disparities
        if len(slice_results) >= 2:
            disparities = detect_disparities(
                slice_results,
                overall_metrics,
                alert_threshold,
                slice_attr,
                logger
            )
            bias_results['disparities'][slice_attr] = disparities
            
            if disparities['has_significant_disparity']:
                bias_results['bias_detected'] = True
    
    # Generate recommendations if bias detected
    if bias_results['bias_detected']:
        bias_results['recommendations'] = generate_bias_recommendations(
            bias_results['disparities'],
            logger
        )
    else:
        logger.info("\n✓ No significant bias detected across demographic groups")
    
    return bias_results


def detect_disparities(
    slice_results: Dict,
    overall_metrics: Dict,
    threshold: float,
    slice_attr: str,
    logger: logging.Logger
) -> Dict:
    """
    Detect performance disparities between demographic slices.
    
    Args:
        slice_results: Metrics for each slice
        overall_metrics: Overall baseline metrics
        threshold: Threshold for significant disparity (e.g., 0.05 = 5%)
        slice_attr: Name of slicing attribute
        logger: Logger instance
        
    Returns:
        Dictionary with disparity analysis
    """
    logger.info(f"\n  Analyzing disparities for {slice_attr}...")
    
    # Extract ROUGE-L scores for all slices
    rougeL_scores = {
        group: metrics['rougeL_mean']
        for group, metrics in slice_results.items()
    }
    
    # Find min and max performance
    min_group = min(rougeL_scores.items(), key=lambda x: x[1])
    max_group = max(rougeL_scores.items(), key=lambda x: x[1])
    
    # Calculate disparity
    absolute_disparity = max_group[1] - min_group[1]
    relative_disparity = absolute_disparity / overall_metrics['rougeL_mean']
    
    has_disparity = absolute_disparity > threshold
    
    logger.info(f"  Lowest performing group: {min_group[0]} (ROUGE-L: {min_group[1]:.4f})")
    logger.info(f"  Highest performing group: {max_group[0]} (ROUGE-L: {max_group[1]:.4f})")
    logger.info(f"  Absolute disparity: {absolute_disparity:.4f}")
    logger.info(f"  Relative disparity: {relative_disparity:.2%}")
    
    if has_disparity:
        logger.warning(f"  ⚠ SIGNIFICANT DISPARITY DETECTED (>{threshold:.2%})")
    else:
        logger.info(f"  ✓ No significant disparity (threshold: {threshold:.2%})")
    
    return {
        'has_significant_disparity': has_disparity,
        'absolute_disparity': absolute_disparity,
        'relative_disparity': relative_disparity,
        'lowest_group': min_group[0],
        'lowest_score': min_group[1],
        'highest_group': max_group[0],
        'highest_score': max_group[1],
        'threshold_used': threshold
    }


def generate_bias_recommendations(
    disparities: Dict,
    logger: logging.Logger
) -> List[str]:
    """
    Generate mitigation recommendations based on detected bias.
    
    Args:
        disparities: Disparity analysis results
        logger: Logger instance
        
    Returns:
        List of recommendation strings
    """
    logger.info("\n" + "=" * 80)
    logger.info("BIAS MITIGATION RECOMMENDATIONS")
    logger.info("=" * 80)
    
    recommendations = []
    
    for attr, disparity_info in disparities.items():
        if disparity_info['has_significant_disparity']:
            recommendations.append(
                f"[{attr}] Significant disparity detected between '{disparity_info['lowest_group']}' "
                f"and '{disparity_info['highest_group']}' groups"
            )
            
            # Specific recommendations based on disparity type
            if 'age' in attr.lower():
                recommendations.append(
                    "  → Consider data augmentation for underrepresented age groups"
                )
                recommendations.append(
                    "  → Review if model has sufficient training examples across all age ranges"
                )
            
            elif 'gender' in attr.lower():
                recommendations.append(
                    "  → Ensure balanced representation of gender groups in training data"
                )
                recommendations.append(
                    "  → Consider re-weighting loss function to account for class imbalance"
                )
            
            elif 'ethnicity' in attr.lower():
                recommendations.append(
                    "  → Review training data for ethnic representation balance"
                )
                recommendations.append(
                    "  → Consider collecting more data from underrepresented groups"
                )
            
            # General recommendations
            recommendations.append(
                "  → Apply fairness constraints during training (e.g., demographic parity)"
            )
            recommendations.append(
                "  → Consider stratified sampling to ensure balanced evaluation"
            )
    
    if recommendations:
        logger.info("\nRecommended Actions:")
        for rec in recommendations:
            logger.info(rec)
    
    return recommendations


def save_bias_report(
    bias_results: Dict,
    predictions_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Save comprehensive bias detection report.
    
    Args:
        bias_results: Bias analysis results
        predictions_df: DataFrame with predictions
        output_dir: Directory to save report
        logger: Logger instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON report
    report_path = output_dir / f"bias_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(bias_results, f, indent=2, default=str)
    logger.info(f"\n✓ Bias report saved: {report_path}")
    
    # Save detailed predictions with slice information
    predictions_path = output_dir / f"predictions_with_slices_{timestamp}.csv"
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"✓ Predictions saved: {predictions_path}")
    
    # Create human-readable summary
    summary_path = output_dir / f"bias_summary_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BIAS DETECTION REPORT SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERALL METRICS:\n")
        om = bias_results['overall_metrics']
        f.write(f"  Total Samples: {om['sample_count']}\n")
        f.write(f"  ROUGE-1: {om['rouge1_mean']:.4f} (±{om['rouge1_std']:.4f})\n")
        f.write(f"  ROUGE-2: {om['rouge2_mean']:.4f} (±{om['rouge2_std']:.4f})\n")
        f.write(f"  ROUGE-L: {om['rougeL_mean']:.4f} (±{om['rougeL_std']:.4f})\n\n")
        
        f.write("BIAS DETECTION RESULTS:\n")
        if bias_results['bias_detected']:
            f.write("  ⚠ BIAS DETECTED\n\n")
            for attr, disp in bias_results['disparities'].items():
                if disp['has_significant_disparity']:
                    f.write(f"  [{attr}]:\n")
                    f.write(f"    Lowest: {disp['lowest_group']} ({disp['lowest_score']:.4f})\n")
                    f.write(f"    Highest: {disp['highest_group']} ({disp['highest_score']:.4f})\n")
                    f.write(f"    Disparity: {disp['absolute_disparity']:.4f} ({disp['relative_disparity']:.2%})\n\n")
            
            if bias_results['recommendations']:
                f.write("RECOMMENDATIONS:\n")
                for rec in bias_results['recommendations']:
                    f.write(f"  {rec}\n")
        else:
            f.write("  ✓ No significant bias detected\n")
    
    logger.info(f"✓ Summary saved: {summary_path}")


def main():
    """
    Main bias detection pipeline.
    
    This orchestrates the complete bias detection process:
    1. Load configuration and trained model
    2. Load test data
    3. Generate predictions
    4. Perform data slicing analysis
    5. Detect performance disparities
    6. Generate bias report with recommendations
    """
    parser = argparse.ArgumentParser(
        description="Detect bias in BioBART model across demographic groups"
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
        default='model-development/bias_reports',
        help='Directory to save bias reports'
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
        # Try the actual trained model location
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
    logger.info("BIAS DETECTION PIPELINE - Lab Lens Project")
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
            max_length=config['data_config'].get('max_target_length', 150),
            logger=logger
        )
        
        # Perform bias detection
        bias_results = perform_bias_detection(predictions_df, config, logger)
        
        # Save bias report
        save_bias_report(bias_results, predictions_df, output_dir, logger)
        
        logger.info("\n" + "=" * 80)
        logger.info("BIAS DETECTION COMPLETE")
        logger.info("=" * 80)
        
        if bias_results['bias_detected']:
            logger.warning("⚠ Bias detected - review report for mitigation strategies")
            return 1
        else:
            logger.info("✓ No significant bias detected across demographic groups")
            return 0
        
    except Exception as e:
        logger.error(f"Bias detection failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    sys.exit(main())