# Lab Lens - Model Development

## Overview

Complete MLOps pipeline for training, validating, and deploying BioBART medical summarization models. All scripts work cross-platform (macOS M4, Windows, Linux) with automatic path resolution.

## Scripts Created

### Core Training & Evaluation
1. **train_biobart.py** - Production training pipeline
2. **validate_model.py** - Comprehensive metrics evaluation
3. **bias_detection_model.py** - Demographic fairness analysis
4. **sensitivity_analysis.py** - Feature importance & interpretability
5. **model_registry.py** - Version control & deployment
6. **rollback_model.py** - Automatic version management

### Supporting Files
7. **.github/workflows/model_training.yml** - CI/CD automation

## Quick Start

### 1. Setup Environment

```bash
cd lab-lens
source venv/bin/activate  # macOS/Linux
# or
.\venv\Scripts\Activate.ps1  # Windows

pip install -r model-development/requirements.txt
```

### 2. Prepare Training Data

```bash
# Run data pipeline first
python data-pipeline/scripts/main_pipeline.py

# Prepare model-ready data
python model-development/scripts/prepare_model_data.py
```

### 3. Run Complete Pipeline

```bash
# From lab-lens root directory

# Step 1: Train model (or use existing from Colab)
python model-development/scripts/train_biobart.py

# Step 2: Validate model
python model-development/scripts/validate_model.py

# Step 3: Check for bias
python model-development/scripts/bias_detection_model.py

# Step 4: Sensitivity analysis
python model-development/scripts/sensitivity_analysis.py --num-samples 20

# Step 5: Register model
python model-development/scripts/model_registry.py --version v1.0
```

## Script Details

### train_biobart.py

**Purpose:** Fine-tune BioBART on MIMIC-III discharge summaries

**Usage:**
```bash
python model-development/scripts/train_biobart.py \
    --config model-development/configs/model_config.json \
    --device auto
```

**Outputs:**
- Trained model in `models/biobart_summarization/`
- Training logs in `logs/`
- MLflow experiments in `logs/mlruns/`

**Time:** 2-3 hours on M4 MPS, 1-2 hours on T4 GPU

---

### validate_model.py

**Purpose:** Comprehensive model evaluation with multiple metrics

**Usage:**
```bash
python model-development/scripts/validate_model.py
```

**Metrics Computed:**
- ROUGE-1, ROUGE-2, ROUGE-L
- BLEU score
- BERTScore (if installed)
- Summary statistics

**Outputs:**
- `validation_reports/validation_report_*.json`
- `validation_reports/predictions_*.csv`
- `validation_reports/validation_summary_*.txt`

**Time:** 5-10 minutes

---

### bias_detection_model.py

**Purpose:** Detect bias across demographic groups using data slicing

**Usage:**
```bash
python model-development/scripts/bias_detection_model.py
```

**Analysis:**
- Slices by age_group, gender, ethnicity
- Computes ROUGE scores per slice
- Identifies performance disparities
- Suggests mitigation strategies

**Outputs:**
- `bias_reports/bias_report_*.json`
- `bias_reports/predictions_with_slices_*.csv`
- `bias_reports/bias_summary_*.txt`

**Time:** 5-10 minutes

---

### sensitivity_analysis.py

**Purpose:** Model interpretability and feature importance

**Usage:**
```bash
python model-development/scripts/sensitivity_analysis.py --num-samples 20
```

**Analysis:**
- LIME explanations for individual predictions
- Input length sensitivity
- Clinical section importance

**Outputs:**
- `sensitivity_reports/lime_explanation_*.png`
- `sensitivity_reports/length_sensitivity_analysis.png`
- `sensitivity_reports/section_importance.png`
- `sensitivity_reports/sensitivity_report_*.json`

**Time:** 3-5 minutes

---

### model_registry.py

**Purpose:** Version control and deployment preparation

**Usage:**
```bash
python model-development/scripts/model_registry.py --version v1.0
```

**Features:**
- Quality threshold validation
- Metadata tracking
- Version history
- Local registry (GCP-ready)

**Outputs:**
- `model_registry/v{version}/` - Versioned model copy
- `model_registry/registry_index.json` - Version tracking

**Time:** 1-2 minutes

---

### rollback_model.py

**Purpose:** Automatic rollback if new model underperforms

**Usage:**
```bash
python model-development/scripts/rollback_model.py
```

**Features:**
- Compares new vs production model
- Automatic rollback decision
- 2% degradation threshold
- Updates production pointers

**Time:** < 1 minute

---

## CI/CD Pipeline

### GitHub Actions Workflow

**Location:** `.github/workflows/model_training.yml`

**Triggers:**
- Push to `model-development` or `main` branch
- Pull requests
- Manual workflow dispatch

**Jobs:**
1. **Setup & Prepare** - Data pipeline, data preparation
2. **Train Model** - Model training (optional, can skip)
3. **Validate** - Metrics evaluation with thresholds
4. **Bias Detection** - Fairness checks
5. **Deploy** - Push to registry if all checks pass
6. **Rollback** - Automatic rollback on failure

**Setup:**
```bash
# Copy workflow file
mkdir -p .github/workflows
cp model_training.yml .github/workflows/

# Commit and push
git add .github/workflows/model_training.yml
git commit -m "Add model training CI/CD pipeline"
git push
```

---

## Directory Structure

```
model-development/
├── scripts/
│   ├── train_biobart.py              ← Training pipeline
│   ├── validate_model.py             ← Validation metrics
│   ├── bias_detection_model.py       ← Bias analysis
│   ├── sensitivity_analysis.py       ← Interpretability
│   ├── model_registry.py             ← Version control
│   └── rollback_model.py             ← Rollback management
│
├── configs/
│   └── model_config.json             ← Configuration
│
├── data/
│   └── model_ready/                  ← Training data
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── models/
│   └── biobart_summarization/        ← Trained model
│
├── logs/
│   └── mlruns/                       ← MLflow experiments
│
├── validation_reports/               ← Validation outputs
├── bias_reports/                     ← Bias detection outputs
├── sensitivity_reports/              ← Sensitivity analysis outputs
└── model_registry/                   ← Versioned models
    ├── v20251124_190450/
    └── registry_index.json
```

---

## Rubric Compliance

### ✅ Section 2: Model Development and ML Code
1. ✅ Load data from pipeline - `train_biobart.py`
2. ✅ Train and select best model - `train_biobart.py` with early stopping
3. ✅ Model validation - `validate_model.py` with ROUGE, BLEU, BERTScore
4. ✅ Bias detection - `bias_detection_model.py` with demographic slicing
5. ✅ Code to check bias - Comprehensive bias reports
6. ✅ Push to registry - `model_registry.py`

### ✅ Section 3: Hyperparameter Tuning
- MLflow tracks all experiments
- Configuration-driven tuning
- Experiment comparison in MLflow UI

### ✅ Section 4: Experiment Tracking
- Complete MLflow integration
- Parameters, metrics, and artifacts logged
- Visualization in MLflow UI

### ✅ Section 5: Sensitivity Analysis
- Feature importance with LIME
- Input length sensitivity
- Clinical section analysis

### ✅ Section 6: Bias Detection
- Data slicing by demographics
- Metrics across slices
- Disparity detection
- Mitigation recommendations

### ✅ Section 7: CI/CD Pipeline
- GitHub Actions workflow
- Automated training & validation
- Bias detection in pipeline
- Quality gates
- Automatic rollback
- Notifications

### ✅ Section 8: Code Implementation
- All scripts production-ready
- Cross-platform compatible
- Configuration-driven
- Comprehensive logging
- Error handling

---

## Common Commands

### View MLflow Experiments
```bash
cd model-development
mlflow ui --backend-store-uri logs/mlruns
# Open http://localhost:5000
```

### Check Registry Versions
```bash
cat model-development/model_registry/registry_index.json
```

### Compare Two Models
```bash
python model-development/scripts/rollback_model.py \
    --registry-path model-development/model_registry
```

### Force Rollback
```bash
python model-development/scripts/rollback_model.py --force-rollback
```

---

## Troubleshooting

### Issue: Out of Memory
**Solution:** Reduce batch size in `model_config.json`
```json
"batch_size": 4
```

### Issue: LIME fails with MPS
**Solution:** Run sensitivity analysis on CPU
```bash
python scripts/sensitivity_analysis.py --device cpu --num-samples 10
```

### Issue: Path not found
**Solution:** Always run from `lab-lens` root directory

### Issue: Module not found
**Solution:** Activate virtual environment
```bash
source venv/bin/activate  # macOS/Linux
.\venv\Scripts\Activate.ps1  # Windows
```

---

## Team Workflow

### For New Team Members

1. Clone repository
2. Set up virtual environment
3. Install dependencies
4. Run validation on existing model
5. Review reports in respective directories

### For Model Updates

1. Create feature branch
2. Modify training code or config
3. Run full pipeline locally
4. Push to GitHub (triggers CI/CD)
5. Review automated checks
6. Merge if all checks pass

### For Production Deployment

1. Validate model meets thresholds
2. Run bias detection
3. Register model with version
4. Review deployment instructions
5. Deploy with monitoring
6. Keep rollback ready

---

## Next Steps

After completing model development:

1. **Set up GCP** - Configure credentials for cloud deployment
2. **Create API** - FastAPI endpoint for model serving
3. **Add monitoring** - Track model performance in production
4. **Implement caching** - Speed up inference
5. **Load testing** - Verify scalability

---

## Support

For issues:
1. Check logs in respective `*/logs/` directories
2. Review MLflow experiments
3. Check this README
4. Contact team members

---

**Model Development Pipeline Complete!** 🎉