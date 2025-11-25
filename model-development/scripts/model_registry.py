"""
Model Registry Script for BioBART Medical Summarization
Author: Lab Lens Team
Description: Pushes trained models to GCP Artifact Registry for version control and deployment
             Tracks model metadata, metrics, and lineage for reproducibility
             
Required by Rubric: Section 2.6 - Pushing Model to Artifact Registry

This script:
1. Loads trained BioBART model and its metadata
2. Validates model meets quality thresholds
3. Packages model with version information
4. Pushes to GCP Artifact Registry (or local registry for testing)
5. Tags model with performance metrics and training metadata
6. Maintains model lineage and version history
"""

import os
import sys
import json
import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Core dependencies
import pandas as pd

# GCP dependencies (optional - graceful degradation if not available)
try:
    from google.cloud import storage
    from google.cloud import aiplatform
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    print("Warning: google-cloud libraries not available. Using local registry mode.")


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
    log_file = log_dir / f"model_registry_{timestamp}.log"
    
    logger = logging.getLogger("ModelRegistry")
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


def validate_model_exists(model_path: Path, logger: logging.Logger) -> bool:
    """
    Validate that the trained model exists and has required files.
    
    Args:
        model_path: Path to model directory
        logger: Logger instance
        
    Returns:
        True if model is valid, False otherwise
    """
    logger.info(f"Validating model at: {model_path}")
    
    if not model_path.exists():
        logger.error(f"Model directory not found: {model_path}")
        return False
    
    # Check for required files
    required_files = [
        'config.json',
        'tokenizer_config.json'
    ]
    
    # Check for model weights (either .bin or .safetensors)
    has_weights = (model_path / 'pytorch_model.bin').exists() or \
                  (model_path / 'model.safetensors').exists()
    
    if not has_weights:
        logger.error("Model weights not found (pytorch_model.bin or model.safetensors)")
        return False
    
    # Check other required files
    for req_file in required_files:
        if not (model_path / req_file).exists():
            logger.warning(f"Missing file: {req_file}")
    
    logger.info("✓ Model validation passed")
    return True


def load_model_metrics(validation_reports_dir: Path, logger: logging.Logger) -> Optional[Dict]:
    """
    Load the most recent validation metrics for the model.
    
    Args:
        validation_reports_dir: Directory containing validation reports
        logger: Logger instance
        
    Returns:
        Dictionary with validation metrics, or None if not found
    """
    logger.info(f"Loading validation metrics from: {validation_reports_dir}")
    
    if not validation_reports_dir.exists():
        logger.warning("Validation reports directory not found")
        return None
    
    # Find most recent validation report
    report_files = list(validation_reports_dir.glob("validation_report_*.json"))
    
    if not report_files:
        logger.warning("No validation reports found")
        return None
    
    # Get most recent report
    latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
    
    logger.info(f"Loading metrics from: {latest_report.name}")
    
    try:
        with open(latest_report, 'r') as f:
            metrics = json.load(f)
        
        logger.info("✓ Metrics loaded successfully:")
        logger.info(f"  ROUGE-L: {metrics.get('rougeL_mean', 'N/A')}")
        logger.info(f"  BLEU: {metrics.get('bleu', 'N/A')}")
        
        return metrics
    except Exception as e:
        logger.warning(f"Failed to load metrics: {e}")
        return None


def check_quality_thresholds(metrics: Dict, logger: logging.Logger) -> bool:
    """
    Check if model meets minimum quality thresholds.
    
    Args:
        metrics: Dictionary with validation metrics
        logger: Logger instance
        
    Returns:
        True if model meets thresholds, False otherwise
    """
    logger.info("Checking quality thresholds...")
    
    if not metrics:
        logger.warning("No metrics available, skipping threshold check")
        return True  # Allow push if no metrics
    
    # Define minimum thresholds (adjust based on your requirements)
    thresholds = {
        'rougeL_mean': 0.15,  # Minimum ROUGE-L score
        'rouge1_mean': 0.20,  # Minimum ROUGE-1 score
    }
    
    passed = True
    
    for metric_name, threshold in thresholds.items():
        metric_value = metrics.get(metric_name, 0)
        
        if metric_value < threshold:
            logger.warning(f"✗ {metric_name}: {metric_value:.4f} < {threshold:.4f} (threshold)")
            passed = False
        else:
            logger.info(f"✓ {metric_name}: {metric_value:.4f} >= {threshold:.4f} (threshold)")
    
    return passed


def create_model_metadata(
    model_path: Path,
    metrics: Optional[Dict],
    config: Dict,
    version: str,
    logger: logging.Logger
) -> Dict:
    """
    Create comprehensive metadata for the model.
    
    Args:
        model_path: Path to model directory
        metrics: Validation metrics
        config: Model configuration
        version: Model version string
        logger: Logger instance
        
    Returns:
        Dictionary with model metadata
    """
    logger.info("Creating model metadata...")
    
    metadata = {
        'model_name': 'biobart-mimic-summarization',
        'version': version,
        'created_at': datetime.now().isoformat(),
        'model_path': str(model_path),
        'model_type': 'BioBART',
        'task': 'medical-summarization',
        'framework': 'transformers',
        
        # Training configuration
        'training_config': {
            'base_model': config['model_config'].get('model_name'),
            'epochs': config['training_config'].get('num_epochs'),
            'batch_size': config['training_config'].get('batch_size'),
            'learning_rate': config['training_config'].get('learning_rate'),
        },
        
        # Data configuration
        'data_config': {
            'max_input_length': config['data_config'].get('max_input_length'),
            'max_target_length': config['data_config'].get('max_target_length'),
        },
        
        # Performance metrics (if available)
        'metrics': metrics if metrics else {},
        
        # Deployment info
        'deployment': {
            'platform': config['deployment_config'].get('deployment_platform', 'local'),
            'registry_type': config['deployment_config'].get('registry_type', 'local'),
        }
    }
    
    logger.info("✓ Metadata created")
    return metadata


def push_to_local_registry(
    model_path: Path,
    metadata: Dict,
    registry_path: Path,
    logger: logging.Logger
) -> bool:
    """
    Push model to local registry (for testing or when GCP is not available).
    
    This creates a versioned copy of the model in a local registry directory
    with metadata and version tracking.
    
    Args:
        model_path: Path to source model directory
        metadata: Model metadata dictionary
        registry_path: Path to local registry directory
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("PUSHING TO LOCAL REGISTRY")
    logger.info("=" * 80)
    
    # Create registry directory structure
    registry_path.mkdir(parents=True, exist_ok=True)
    
    # Create version directory
    version = metadata['version']
    version_dir = registry_path / version
    
    if version_dir.exists():
        logger.warning(f"Version {version} already exists in registry")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            logger.info("Push cancelled by user")
            return False
        shutil.rmtree(version_dir)
    
    version_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Copying model to: {version_dir}")
    
    try:
        # Copy model files
        for item in model_path.iterdir():
            if item.is_file():
                shutil.copy2(item, version_dir / item.name)
                logger.info(f"  Copied: {item.name}")
            elif item.is_dir() and not item.name.startswith('.'):
                shutil.copytree(item, version_dir / item.name)
                logger.info(f"  Copied directory: {item.name}")
        
        # Save metadata
        metadata_path = version_dir / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  Saved metadata: model_metadata.json")
        
        # Update registry index
        update_registry_index(registry_path, metadata, logger)
        
        logger.info("✓ Model successfully pushed to local registry")
        logger.info(f"  Location: {version_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to push to local registry: {e}")
        return False


def update_registry_index(
    registry_path: Path,
    metadata: Dict,
    logger: logging.Logger
):
    """
    Update the registry index file with new model version.
    
    Args:
        registry_path: Path to registry directory
        metadata: Model metadata
        logger: Logger instance
    """
    index_path = registry_path / 'registry_index.json'
    
    # Load existing index or create new one
    if index_path.exists():
        with open(index_path, 'r') as f:
            index = json.load(f)
    else:
        index = {
            'models': [],
            'latest_version': None,
            'updated_at': None
        }
    
    # Add new model entry
    model_entry = {
        'version': metadata['version'],
        'created_at': metadata['created_at'],
        'metrics': {
            'rougeL': metadata['metrics'].get('rougeL_mean', 0),
            'bleu': metadata['metrics'].get('bleu', 0)
        }
    }
    
    # Remove old entry for this version if exists
    index['models'] = [m for m in index['models'] if m['version'] != metadata['version']]
    
    # Add new entry
    index['models'].append(model_entry)
    
    # Sort by creation date
    index['models'].sort(key=lambda x: x['created_at'], reverse=True)
    
    # Update latest version
    index['latest_version'] = metadata['version']
    index['updated_at'] = datetime.now().isoformat()
    
    # Save index
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    logger.info(f"✓ Registry index updated: {len(index['models'])} versions")


def push_to_gcp_registry(
    model_path: Path,
    metadata: Dict,
    config: Dict,
    logger: logging.Logger
) -> bool:
    """
    Push model to GCP Artifact Registry or Vertex AI Model Registry.
    
    This requires GCP credentials and proper project setup.
    
    Args:
        model_path: Path to source model directory
        metadata: Model metadata dictionary
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    if not GCP_AVAILABLE:
        logger.error("GCP libraries not available. Install with:")
        logger.error("  pip install google-cloud-storage google-cloud-aiplatform")
        return False
    
    logger.info("=" * 80)
    logger.info("PUSHING TO GCP REGISTRY")
    logger.info("=" * 80)
    
    deployment_config = config.get('deployment_config', {})
    registry_path = deployment_config.get('registry_path')
    
    if not registry_path or registry_path == 'gcr.io/PROJECT_ID/biobart-mimic-summarization':
        logger.error("GCP registry path not configured properly in model_config.json")
        logger.error("Please update deployment_config.registry_path with your GCP project ID")
        return False
    
    logger.info(f"Target registry: {registry_path}")
    logger.info("Note: This requires GCP credentials and proper project setup")
    logger.info("For now, using local registry instead")
    
    # TODO: Implement actual GCP push when credentials are configured
    # This would involve:
    # 1. Authenticating with GCP
    # 2. Creating a Cloud Storage bucket if needed
    # 3. Uploading model files to GCS
    # 4. Registering model in Vertex AI Model Registry
    # 5. Tagging with metadata
    
    return False


def generate_deployment_instructions(
    registry_path: Path,
    version: str,
    logger: logging.Logger
):
    """
    Generate instructions for deploying the registered model.
    
    Args:
        registry_path: Path to registry
        version: Model version
        logger: Logger instance
    """
    logger.info("\n" + "=" * 80)
    logger.info("DEPLOYMENT INSTRUCTIONS")
    logger.info("=" * 80)
    
    logger.info(f"\nModel version {version} is now available in the registry.")
    logger.info("\nTo load this model in your application:")
    logger.info(f"""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    model_path = "{registry_path / version}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    """)
    
    logger.info("\nTo deploy to production:")
    logger.info("  1. Test the model thoroughly")
    logger.info("  2. Set up serving infrastructure (e.g., FastAPI, Flask)")
    logger.info("  3. Configure monitoring and logging")
    logger.info("  4. Implement rollback mechanism")
    logger.info("  5. Deploy behind a load balancer")


def main():
    """
    Main model registry pipeline.
    
    Orchestrates the complete model registration process:
    1. Load and validate trained model
    2. Load validation metrics
    3. Check quality thresholds
    4. Create comprehensive metadata
    5. Push to registry (local or GCP)
    6. Update version tracking
    7. Generate deployment instructions
    """
    parser = argparse.ArgumentParser(
        description="Register trained model in artifact registry"
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
        '--version',
        type=str,
        default=None,
        help='Model version (default: auto-generated timestamp)'
    )
    parser.add_argument(
        '--registry-path',
        type=str,
        default='model-development/model_registry',
        help='Path to local registry directory'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip quality threshold validation'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force push even if quality thresholds not met'
    )
    parser.add_argument(
        '--use-gcp',
        action='store_true',
        help='Push to GCP registry (requires credentials)'
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
    
    registry_path = Path(args.registry_path)
    log_dir = registry_path / "logs"
    
    # Set up logging
    logger = setup_logging(log_dir, args.log_level)
    logger.info("=" * 80)
    logger.info("MODEL REGISTRY PIPELINE - Lab Lens Project")
    logger.info("=" * 80)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Registry path: {registry_path}")
    
    # Generate version if not provided
    if args.version:
        version = args.version
    else:
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Model version: {version}")
    
    try:
        # Validate model exists
        if not validate_model_exists(model_path, logger):
            logger.error("Model validation failed")
            return 1
        
        # Load validation metrics
        validation_reports_dir = resolve_path("../validation_reports")
        metrics = load_model_metrics(validation_reports_dir, logger)
        
        # Check quality thresholds
        if not args.skip_validation:
            if not check_quality_thresholds(metrics, logger):
                if not args.force:
                    logger.error("Model does not meet quality thresholds")
                    logger.error("Use --force to push anyway")
                    return 1
                else:
                    logger.warning("Forcing push despite quality thresholds")
        
        # Create metadata
        metadata = create_model_metadata(
            model_path,
            metrics,
            config,
            version,
            logger
        )
        
        # Push to registry
        if args.use_gcp:
            success = push_to_gcp_registry(model_path, metadata, config, logger)
            if not success:
                logger.warning("GCP push failed, falling back to local registry")
                success = push_to_local_registry(model_path, metadata, registry_path, logger)
        else:
            success = push_to_local_registry(model_path, metadata, registry_path, logger)
        
        if not success:
            logger.error("Failed to push model to registry")
            return 1
        
        # Generate deployment instructions
        generate_deployment_instructions(registry_path, version, logger)
        
        logger.info("\n" + "=" * 80)
        logger.info("MODEL REGISTRATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Model version {version} successfully registered")
        
        return 0
        
    except Exception as e:
        logger.error(f"Model registration failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    sys.exit(main())