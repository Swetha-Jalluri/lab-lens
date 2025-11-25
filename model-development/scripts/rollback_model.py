"""
Model Rollback Script for BioBART Medical Summarization
Author: Lab Lens Team
Description: Compares new model with previous version and handles rollback if needed
             Ensures suboptimal models are not deployed to production
             
Required by Rubric: Section 7.6 - Rollback Mechanism

This script:
1. Loads current production model version
2. Loads newly trained model version
3. Compares performance metrics
4. Automatically rolls back if new model performs worse
5. Updates production pointers to safe version
6. Logs rollback decisions with justification
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
    log_file = log_dir / f"rollback_{timestamp}.log"
    
    logger = logging.getLogger("ModelRollback")
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


def load_registry_index(registry_path: Path, logger: logging.Logger) -> Dict:
    """
    Load the model registry index containing version history.
    
    Args:
        registry_path: Path to model registry directory
        logger: Logger instance
        
    Returns:
        Registry index dictionary
    """
    index_path = registry_path / 'registry_index.json'
    
    if not index_path.exists():
        logger.warning("Registry index not found, creating new one")
        return {
            'models': [],
            'latest_version': None,
            'production_version': None,
            'updated_at': None
        }
    
    try:
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        logger.info(f"✓ Loaded registry index: {len(index.get('models', []))} versions")
        return index
    
    except Exception as e:
        logger.error(f"Failed to load registry index: {e}")
        raise


def get_model_metrics(version_path: Path, logger: logging.Logger) -> Optional[Dict]:
    """
    Load metrics for a specific model version.
    
    Args:
        version_path: Path to model version directory
        logger: Logger instance
        
    Returns:
        Dictionary with model metrics, or None if not found
    """
    metadata_path = version_path / 'model_metadata.json'
    
    if not metadata_path.exists():
        logger.warning(f"Metadata not found for version: {version_path.name}")
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata.get('metrics', {})
    
    except Exception as e:
        logger.warning(f"Failed to load metadata: {e}")
        return None


def compare_models(
    current_metrics: Dict,
    new_metrics: Dict,
    logger: logging.Logger
) -> Tuple[bool, str]:
    """
    Compare current production model with new model.
    
    Args:
        current_metrics: Metrics from current production model
        new_metrics: Metrics from newly trained model
        logger: Logger instance
        
    Returns:
        Tuple of (should_rollback, reason)
    """
    logger.info("=" * 80)
    logger.info("COMPARING MODEL VERSIONS")
    logger.info("=" * 80)
    
    # Primary metric: ROUGE-L
    current_rouge = current_metrics.get('rougeL_mean', 0)
    new_rouge = new_metrics.get('rougeL_mean', 0)
    
    # Secondary metric: BLEU
    current_bleu = current_metrics.get('bleu', 0)
    new_bleu = new_metrics.get('bleu', 0)
    
    logger.info("\nCurrent Production Model:")
    logger.info(f"  ROUGE-L: {current_rouge:.4f}")
    logger.info(f"  BLEU: {current_bleu:.4f}")
    
    logger.info("\nNew Model:")
    logger.info(f"  ROUGE-L: {new_rouge:.4f}")
    logger.info(f"  BLEU: {new_bleu:.4f}")
    
    # Calculate performance change
    rouge_change = new_rouge - current_rouge
    bleu_change = new_bleu - current_bleu
    
    logger.info("\nPerformance Change:")
    logger.info(f"  ROUGE-L: {rouge_change:+.4f} ({rouge_change/current_rouge*100:+.2f}%)")
    logger.info(f"  BLEU: {bleu_change:+.4f} ({bleu_change/current_bleu*100:+.2f}%)")
    
    # Decision criteria
    # Rollback if primary metric (ROUGE-L) decreased by more than 2%
    rollback_threshold = -0.02  # 2% degradation
    
    should_rollback = False
    reason = ""
    
    if rouge_change < rollback_threshold:
        should_rollback = True
        reason = f"ROUGE-L degraded by {abs(rouge_change):.4f} ({abs(rouge_change/current_rouge*100):.1f}%), exceeding threshold"
    elif new_rouge < current_rouge * 0.95:  # More than 5% worse
        should_rollback = True
        reason = f"ROUGE-L dropped by {(1 - new_rouge/current_rouge)*100:.1f}%, significant degradation"
    else:
        reason = "New model performance is acceptable"
    
    logger.info(f"\nDecision: {'ROLLBACK' if should_rollback else 'PROCEED'}")
    logger.info(f"Reason: {reason}")
    
    return should_rollback, reason


def perform_rollback(
    registry_path: Path,
    current_version: str,
    logger: logging.Logger
) -> bool:
    """
    Perform rollback to previous model version.
    
    This restores the registry to point to the last known good version.
    
    Args:
        registry_path: Path to model registry
        current_version: Version to roll back to
        logger: Logger instance
        
    Returns:
        True if rollback successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("PERFORMING ROLLBACK")
    logger.info("=" * 80)
    
    logger.info(f"Rolling back to version: {current_version}")
    
    try:
        # Load registry index
        index_path = registry_path / 'registry_index.json'
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        # Update production version pointer
        old_production = index.get('production_version')
        index['production_version'] = current_version
        index['rollback_at'] = datetime.now().isoformat()
        index['rollback_from'] = old_production
        
        # Save updated index
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"✓ Rollback complete")
        logger.info(f"  Production version: {current_version}")
        logger.info(f"  Rolled back from: {old_production}")
        
        return True
    
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        return False


def update_production_pointer(
    registry_path: Path,
    new_version: str,
    logger: logging.Logger
) -> bool:
    """
    Update production pointer to new model version.
    
    Args:
        registry_path: Path to model registry
        new_version: New production version
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Updating production pointer to: {new_version}")
    
    try:
        index_path = registry_path / 'registry_index.json'
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        old_version = index.get('production_version')
        index['production_version'] = new_version
        index['production_updated_at'] = datetime.now().isoformat()
        
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"✓ Production pointer updated")
        logger.info(f"  Old version: {old_version}")
        logger.info(f"  New version: {new_version}")
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to update production pointer: {e}")
        return False


def main():
    """
    Main rollback pipeline.
    
    Orchestrates the rollback decision process:
    1. Load registry index
    2. Identify current production and new versions
    3. Load metrics for both versions
    4. Compare performance
    5. Make rollback decision
    6. Execute rollback or promotion
    7. Log decision and reasoning
    """
    parser = argparse.ArgumentParser(
        description="Model rollback and version management"
    )
    parser.add_argument(
        '--registry-path',
        type=str,
        default='model-development/model_registry',
        help='Path to model registry directory'
    )
    parser.add_argument(
        '--new-version',
        type=str,
        required=False,
        help='New model version to evaluate'
    )
    parser.add_argument(
        '--force-rollback',
        action='store_true',
        help='Force rollback to previous version'
    )
    parser.add_argument(
        '--force-promote',
        action='store_true',
        help='Force promotion of new version'
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
    registry_path = Path(args.registry_path)
    log_dir = registry_path / "logs"
    
    # Set up logging
    logger = setup_logging(log_dir, args.log_level)
    logger.info("=" * 80)
    logger.info("MODEL ROLLBACK PIPELINE - Lab Lens Project")
    logger.info("=" * 80)
    logger.info(f"Registry path: {registry_path}")
    
    try:
        # Load registry index
        index = load_registry_index(registry_path, logger)
        
        if not index.get('models'):
            logger.warning("No models in registry, nothing to rollback")
            return 0
        
        # Identify versions
        current_production = index.get('production_version')
        latest_version = index.get('latest_version')
        new_version = args.new_version or latest_version
        
        logger.info(f"\nVersion Status:")
        logger.info(f"  Current Production: {current_production or 'Not set'}")
        logger.info(f"  Latest Version: {latest_version}")
        logger.info(f"  Evaluating: {new_version}")
        
        # Handle force flags
        if args.force_rollback:
            logger.warning("Force rollback requested")
            if current_production:
                perform_rollback(registry_path, current_production, logger)
                return 0
            else:
                logger.error("No production version to rollback to")
                return 1
        
        if args.force_promote:
            logger.warning("Force promotion requested")
            update_production_pointer(registry_path, new_version, logger)
            return 0
        
        # If no production version set, promote latest
        if not current_production:
            logger.info("No production version set, promoting latest version")
            update_production_pointer(registry_path, new_version, logger)
            return 0
        
        # Load metrics for both versions
        current_version_path = registry_path / current_production
        new_version_path = registry_path / new_version
        
        current_metrics = get_model_metrics(current_version_path, logger)
        new_metrics = get_model_metrics(new_version_path, logger)
        
        if not current_metrics or not new_metrics:
            logger.warning("Unable to compare models - missing metrics")
            logger.warning("Proceeding without comparison (use --force flags to override)")
            return 0
        
        # Compare models
        should_rollback, reason = compare_models(current_metrics, new_metrics, logger)
        
        # Execute decision
        if should_rollback:
            logger.warning("⚠️ ROLLBACK REQUIRED")
            logger.warning(f"Reason: {reason}")
            success = perform_rollback(registry_path, current_production, logger)
            
            if success:
                logger.info("✓ Rollback completed successfully")
                return 0
            else:
                logger.error("✗ Rollback failed")
                return 1
        else:
            logger.info("✓ NEW MODEL APPROVED")
            logger.info(f"Reason: {reason}")
            success = update_production_pointer(registry_path, new_version, logger)
            
            if success:
                logger.info("✓ Production updated to new version")
                return 0
            else:
                logger.error("✗ Failed to update production")
                return 1
        
    except Exception as e:
        logger.error(f"Rollback pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    sys.exit(main())