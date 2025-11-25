# Dockerfile for Lab Lens Model Development
# Author: Lab Lens Team
# Description: Containerized environment for BioBART training and inference
# Required by Rubric: Section 8.1 - Docker Format
#
# This Dockerfile:
# - Creates reproducible environment for all team members
# - Works on macOS M4, Windows, and Linux
# - Includes all dependencies for training, validation, and bias detection
# - Supports both CPU and GPU execution
# - Ensures consistent results across different machines

# Use official Python runtime as base image
# Using slim variant to reduce image size
FROM python:3.9-slim

# Set metadata
LABEL maintainer="Lab Lens Team"
LABEL description="BioBART Medical Summarization Model Development"
LABEL version="1.0"

# Set working directory inside container
WORKDIR /app

# Set environment variables
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1
# Disable tokenizers parallelism warning
ENV TOKENIZERS_PARALLELISM=false

# Install system dependencies
# These are needed for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY model-development/requirements.txt /app/requirements.txt

# Install Python dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy model development code
COPY model-development/ /app/model-development/

# Copy data pipeline code (needed for data loading)
COPY data-pipeline/ /app/data-pipeline/

# Copy configuration files
COPY model-development/configs/ /app/model-development/configs/

# Create necessary directories
RUN mkdir -p /app/model-development/models && \
    mkdir -p /app/model-development/logs && \
    mkdir -p /app/model-development/data && \
    mkdir -p /app/model-development/validation_reports && \
    mkdir -p /app/model-development/bias_reports && \
    mkdir -p /app/model-development/sensitivity_reports && \
    mkdir -p /app/model-development/model_registry

# Set permissions
RUN chmod -R 755 /app/model-development/scripts

# Expose port for MLflow UI
EXPOSE 5000

# Default command: Show help
CMD ["python", "model-development/scripts/train_biobart.py", "--help"]

# ============================================================================
# Usage Examples:
# ============================================================================
#
# Build the image:
#   docker build -t lab-lens-model-dev -f model-development/Dockerfile .
#
# Run training:
#   docker run -v $(pwd)/data:/app/data lab-lens-model-dev \
#       python model-development/scripts/train_biobart.py
#
# Run validation:
#   docker run -v $(pwd)/model-development:/app/model-development lab-lens-model-dev \
#       python model-development/scripts/validate_model.py
#
# Run bias detection:
#   docker run -v $(pwd)/model-development:/app/model-development lab-lens-model-dev \
#       python model-development/scripts/bias_detection_model.py
#
# Start MLflow UI:
#   docker run -p 5000:5000 -v $(pwd)/model-development:/app/model-development \
#       lab-lens-model-dev mlflow ui --host 0.0.0.0 \
#       --backend-store-uri /app/model-development/logs/mlruns
#
# Interactive shell:
#   docker run -it -v $(pwd):/app lab-lens-model-dev /bin/bash
#
# ============================================================================