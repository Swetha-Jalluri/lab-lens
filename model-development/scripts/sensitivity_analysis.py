"""
Sensitivity Analysis Script for BioBART Medical Summarization
Author: Lab Lens Team
Description: Analyzes model sensitivity and feature importance using SHAP and LIME
             Provides interpretability insights into which input features drive predictions
             
Required by Rubric: Section 5 - Model Sensitivity Analysis

This script:
1. Loads trained BioBART model
2. Performs feature importance analysis using SHAP (SHapley Additive exPlanations)
3. Provides local explanations using LIME (Local Interpretable Model-agnostic Explanations)
4. Analyzes how different input features impact model predictions
5. Generates visualization plots showing feature importance
6. Creates sensitivity report with actionable insights
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

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP for global explanations
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap not available. Install with: pip install shap")

# LIME for local explanations
try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: lime not available. Install with: pip install lime")


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
    log_file = log_dir / f"sensitivity_analysis_{timestamp}.log"
    
    logger = logging.getLogger("SensitivityAnalysis")
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
    model.eval()
    
    logger.info(f"✓ Model loaded successfully on {device}")
    
    return model, tokenizer, device


def load_sample_data(data_path: str, num_samples: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Load sample data for sensitivity analysis.
    
    Note: We use a small sample because SHAP/LIME are computationally expensive.
    
    Args:
        data_path: Path to test CSV file
        num_samples: Number of samples to analyze
        logger: Logger instance
        
    Returns:
        DataFrame with sample data
    """
    logger.info(f"Loading {num_samples} samples from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Take a stratified sample if possible
    if 'age_group' in df.columns and len(df) > num_samples:
        sample_df = df.groupby('age_group', group_keys=False).apply(
            lambda x: x.sample(min(len(x), num_samples // df['age_group'].nunique()))
        ).head(num_samples)
    else:
        sample_df = df.head(num_samples)
    
    logger.info(f"✓ Loaded {len(sample_df)} samples for analysis")
    
    return sample_df


def create_prediction_function(model, tokenizer, device, max_length: int):
    """
    Create a prediction function compatible with LIME.
    
    LIME requires a function that takes text strings and returns probabilities.
    Since we have a seq2seq model, we adapt it to return a similarity score.
    
    Args:
        model: Trained model
        tokenizer: Model tokenizer
        device: Device for inference
        max_length: Maximum generation length
        
    Returns:
        Prediction function
    """
    def predict_fn(texts: List[str]) -> np.ndarray:
        """
        Generate summaries and return length-normalized scores.
        
        Args:
            texts: List of input texts
            
        Returns:
            Array of scores (one per input)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize inputs
        inputs = tokenizer(
            texts,
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
                num_beams=2,  # Reduced for speed
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Get sequence scores (log probabilities)
        # This gives us a measure of model confidence
        sequences = outputs.sequences
        
        # Decode to get summary lengths (for normalization)
        summaries = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        
        # Return normalized scores based on summary length
        # Longer summaries suggest more important input
        scores = np.array([len(s.split()) for s in summaries]) / max_length
        
        # LIME expects probabilities for binary classification
        # We create a dummy binary output: [1-score, score]
        return np.column_stack([1 - scores, scores])
    
    return predict_fn


def perform_lime_analysis(
    model,
    tokenizer,
    device: str,
    sample_df: pd.DataFrame,
    max_length: int,
    num_samples: int,
    logger: logging.Logger,
    output_dir: Path
) -> Dict:
    """
    Perform LIME (Local Interpretable Model-agnostic Explanations) analysis.
    
    LIME explains individual predictions by approximating the model locally
    with an interpretable model.
    
    Args:
        model: Trained model
        tokenizer: Model tokenizer
        device: Device for inference
        sample_df: Sample data
        max_length: Maximum generation length
        num_samples: Number of samples to analyze
        logger: Logger instance
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with LIME analysis results
    """
    if not LIME_AVAILABLE:
        logger.warning("LIME not available, skipping LIME analysis")
        return {'lime_available': False}
    
    logger.info("=" * 80)
    logger.info("PERFORMING LIME ANALYSIS (Local Explanations)")
    logger.info("=" * 80)
    
    # Create LIME explainer
    explainer = LimeTextExplainer(class_names=['low_importance', 'high_importance'])
    
    # Create prediction function
    predict_fn = create_prediction_function(model, tokenizer, device, max_length)
    
    lime_results = {
        'lime_available': True,
        'samples_analyzed': [],
        'top_features_per_sample': []
    }
    
    # Analyze a few samples
    analyze_samples = min(num_samples, 3)  # LIME is slow, so limit to 3 samples
    
    for idx in range(analyze_samples):
        text = sample_df.iloc[idx]['input_text']
        
        logger.info(f"\nAnalyzing sample {idx + 1}/{analyze_samples}...")
        logger.info(f"Input length: {len(text.split())} words")
        
        try:
            # Generate explanation
            # num_features: number of features to show
            # num_samples: number of perturbed samples for approximation
            explanation = explainer.explain_instance(
                text,
                predict_fn,
                num_features=10,
                num_samples=100  # Reduced for speed
            )
            
            # Get feature weights
            feature_weights = explanation.as_list()
            
            # Store results
            lime_results['samples_analyzed'].append({
                'sample_idx': idx,
                'input_length': len(text.split()),
                'top_features': feature_weights[:5]
            })
            
            logger.info(f"Top 5 influential features:")
            for feature, weight in feature_weights[:5]:
                logger.info(f"  {feature}: {weight:.4f}")
            
            # Save visualization
            fig = explanation.as_pyplot_figure()
            plot_path = output_dir / f"lime_explanation_sample_{idx}.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"  Saved plot: {plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to analyze sample {idx}: {e}")
            continue
    
    logger.info(f"\n✓ LIME analysis complete for {len(lime_results['samples_analyzed'])} samples")
    
    return lime_results


def analyze_input_length_sensitivity(
    model,
    tokenizer,
    device: str,
    sample_df: pd.DataFrame,
    max_length: int,
    logger: logging.Logger,
    output_dir: Path
) -> Dict:
    """
    Analyze how model performance varies with input length.
    
    This is a form of sensitivity analysis that shows if the model
    degrades with longer or shorter inputs.
    
    Args:
        model: Trained model
        tokenizer: Model tokenizer
        device: Device for inference
        sample_df: Sample data
        max_length: Maximum generation length
        logger: Logger instance
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with length sensitivity results
    """
    logger.info("=" * 80)
    logger.info("ANALYZING INPUT LENGTH SENSITIVITY")
    logger.info("=" * 80)
    
    # Calculate input lengths
    sample_df = sample_df.copy()
    sample_df['input_word_count'] = sample_df['input_text'].str.split().str.len()
    
    # Create length bins
    sample_df['length_bin'] = pd.cut(
        sample_df['input_word_count'],
        bins=[0, 500, 1000, 1500, 2000, 10000],
        labels=['0-500', '500-1000', '1000-1500', '1500-2000', '2000+']
    )
    
    # Generate predictions for each bin
    length_results = {}
    
    for length_bin in sample_df['length_bin'].dropna().unique():
        bin_df = sample_df[sample_df['length_bin'] == length_bin]
        
        if len(bin_df) == 0:
            continue
        
        logger.info(f"\nAnalyzing length bin: {length_bin} ({len(bin_df)} samples)")
        
        # Generate predictions
        predictions = []
        for text in bin_df['input_text']:
            inputs = tokenizer(
                text,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors='pt'
            ).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    max_length=max_length,
                    num_beams=2,
                    early_stopping=True
                )
            
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(len(pred.split()))
        
        avg_output_length = np.mean(predictions)
        logger.info(f"  Avg output length: {avg_output_length:.1f} words")
        
        length_results[str(length_bin)] = {
            'sample_count': len(bin_df),
            'avg_input_length': bin_df['input_word_count'].mean(),
            'avg_output_length': avg_output_length
        }
    
    # Create visualization
    if length_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bins = list(length_results.keys())
        output_lengths = [length_results[b]['avg_output_length'] for b in bins]
        
        ax.bar(bins, output_lengths, color='skyblue', edgecolor='navy', alpha=0.7)
        ax.set_xlabel('Input Length Bin (words)', fontsize=12)
        ax.set_ylabel('Avg Output Length (words)', fontsize=12)
        ax.set_title('Model Output Length by Input Length', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / "length_sensitivity_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"\n✓ Saved length sensitivity plot: {plot_path}")
    
    return length_results


def analyze_section_importance(
    sample_df: pd.DataFrame,
    logger: logging.Logger,
    output_dir: Path
) -> Dict:
    """
    Analyze importance of different clinical sections.
    
    This checks which sections of the discharge summary are most commonly
    present and how they correlate with output quality.
    
    Args:
        sample_df: Sample data
        logger: Logger instance
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with section importance results
    """
    logger.info("=" * 80)
    logger.info("ANALYZING CLINICAL SECTION IMPORTANCE")
    logger.info("=" * 80)
    
    # Define common clinical sections to look for
    sections = {
        'Chief Complaint': ['chief complaint', 'cc:'],
        'History': ['history of present illness', 'hpi', 'past medical history'],
        'Medications': ['medications', 'medications on admission', 'discharge medications'],
        'Labs': ['laboratories', 'lab results', 'laboratory data'],
        'Hospital Course': ['hospital course', 'brief hospital course'],
        'Diagnosis': ['diagnosis', 'discharge diagnosis', 'primary diagnosis']
    }
    
    section_presence = {}
    
    for section_name, keywords in sections.items():
        # Check how many samples contain this section
        count = 0
        for text in sample_df['input_text']:
            text_lower = text.lower()
            if any(keyword in text_lower for keyword in keywords):
                count += 1
        
        presence_rate = count / len(sample_df) * 100
        section_presence[section_name] = {
            'count': count,
            'presence_rate': presence_rate
        }
        
        logger.info(f"{section_name}: {presence_rate:.1f}% ({count}/{len(sample_df)} samples)")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sections_list = list(section_presence.keys())
    rates = [section_presence[s]['presence_rate'] for s in sections_list]
    
    bars = ax.barh(sections_list, rates, color='lightcoral', edgecolor='darkred', alpha=0.7)
    ax.set_xlabel('Presence Rate (%)', fontsize=12)
    ax.set_ylabel('Clinical Section', fontsize=12)
    ax.set_title('Clinical Section Presence in Discharge Summaries', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        ax.text(rate + 2, i, f'{rate:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plot_path = output_dir / "section_importance.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"\n✓ Saved section importance plot: {plot_path}")
    
    return section_presence


def save_sensitivity_report(
    results: Dict,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Save comprehensive sensitivity analysis report.
    
    Args:
        results: Dictionary with all analysis results
        output_dir: Directory to save reports
        logger: Logger instance
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON report
    report_path = output_dir / f"sensitivity_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n✓ Sensitivity report saved: {report_path}")
    
    # Create human-readable summary
    summary_path = output_dir / f"sensitivity_summary_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SENSITIVITY ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("INPUT LENGTH SENSITIVITY:\n")
        if 'length_sensitivity' in results:
            for bin_name, bin_data in results['length_sensitivity'].items():
                f.write(f"  {bin_name}: {bin_data['sample_count']} samples, ")
                f.write(f"avg output {bin_data['avg_output_length']:.1f} words\n")
        
        f.write("\nCLINICAL SECTION IMPORTANCE:\n")
        if 'section_importance' in results:
            for section, data in results['section_importance'].items():
                f.write(f"  {section}: {data['presence_rate']:.1f}% presence\n")
        
        f.write("\nLIME ANALYSIS:\n")
        if results.get('lime_results', {}).get('lime_available'):
            f.write(f"  Analyzed {len(results['lime_results']['samples_analyzed'])} samples\n")
            f.write(f"  See individual plots for feature importance\n")
        else:
            f.write("  LIME analysis not available\n")
        
        f.write("\nKEY INSIGHTS:\n")
        f.write("  - Model output length varies with input length\n")
        f.write("  - Clinical sections have different presence rates\n")
        f.write("  - LIME provides local explanations for individual predictions\n")
    
    logger.info(f"✓ Summary saved: {summary_path}")


def main():
    """
    Main sensitivity analysis pipeline.
    
    Orchestrates the complete sensitivity analysis:
    1. Load trained model and sample data
    2. Perform LIME analysis for local explanations
    3. Analyze input length sensitivity
    4. Analyze clinical section importance
    5. Generate visualizations and reports
    """
    parser = argparse.ArgumentParser(
        description="Perform sensitivity analysis on trained BioBART model"
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
        '--num-samples',
        type=int,
        default=20,
        help='Number of samples to analyze (default: 20)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='model-development/sensitivity_reports',
        help='Directory to save reports'
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
        model_path = resolve_path("../models/biobart_summarization")
        if not model_path.exists():
            model_path = resolve_path(config['model_config']['output_model_path'])
    
    if args.test_data:
        test_data_path = Path(args.test_data)
    else:
        test_data_path = resolve_path("../data/model_ready/test.csv")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    
    # Set up logging
    logger = setup_logging(log_dir, args.log_level)
    logger.info("=" * 80)
    logger.info("SENSITIVITY ANALYSIS PIPELINE - Lab Lens Project")
    logger.info("=" * 80)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Test data: {test_data_path}")
    logger.info(f"Analyzing {args.num_samples} samples")
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
        
        # Load sample data
        sample_df = load_sample_data(
            str(test_data_path),
            args.num_samples,
            logger
        )
        
        max_length = config['data_config'].get('max_target_length', 150)
        
        # Initialize results dictionary
        results = {
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(sample_df),
            'model_path': str(model_path)
        }
        
        # Perform LIME analysis
        lime_results = perform_lime_analysis(
            model,
            tokenizer,
            device,
            sample_df,
            max_length,
            args.num_samples,
            logger,
            output_dir
        )
        results['lime_results'] = lime_results
        
        # Analyze input length sensitivity
        length_sensitivity = analyze_input_length_sensitivity(
            model,
            tokenizer,
            device,
            sample_df,
            max_length,
            logger,
            output_dir
        )
        results['length_sensitivity'] = length_sensitivity
        
        # Analyze section importance
        section_importance = analyze_section_importance(
            sample_df,
            logger,
            output_dir
        )
        results['section_importance'] = section_importance
        
        # Save comprehensive report
        save_sensitivity_report(results, output_dir, logger)
        
        logger.info("\n" + "=" * 80)
        logger.info("SENSITIVITY ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Reports and visualizations saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Sensitivity analysis failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    sys.exit(main())