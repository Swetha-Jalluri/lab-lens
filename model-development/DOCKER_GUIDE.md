# Docker Setup Guide - Lab Lens Model Development

## Overview

Docker ensures everyone on the team has the exact same environment, eliminating "works on my machine" problems.

## Prerequisites

### Install Docker

**macOS:**
- Download Docker Desktop: https://www.docker.com/products/docker-desktop
- Install and start Docker Desktop
- Verify: `docker --version`

**Windows:**
- Download Docker Desktop: https://www.docker.com/products/docker-desktop
- Install and start Docker Desktop
- Enable WSL 2 backend (recommended)
- Verify: `docker --version`

## Quick Start

### 1. Build Docker Image

From `lab-lens` root directory:

```bash
docker build -t lab-lens-model-dev -f model-development/Dockerfile .
```

**Time:** 5-10 minutes (first build)

### 2. Run MLflow UI

```bash
docker-compose up mlflow
```

Open browser: http://localhost:5000

### 3. Run Validation

```bash
docker-compose run validation
```

### 4. Run Bias Detection

```bash
docker-compose run bias-detection
```

## Docker Compose Commands

### Start All Services

```bash
docker-compose up
```

### Start Specific Service

```bash
# MLflow only
docker-compose up mlflow

# Training only
docker-compose up model-training
```

### Run Validation Pipeline

```bash
docker-compose --profile validation up
```

### Stop All Services

```bash
docker-compose down
```

### View Logs

```bash
docker-compose logs -f mlflow
docker-compose logs -f model-training
```

## Direct Docker Commands

### Run Training

```bash
docker run -v $(pwd)/model-development:/app/model-development \
    lab-lens-model-dev \
    python model-development/scripts/train_biobart.py
```

### Run Validation

```bash
docker run -v $(pwd)/model-development:/app/model-development \
    lab-lens-model-dev \
    python model-development/scripts/validate_model.py
```

### Interactive Shell

```bash
docker run -it -v $(pwd):/app lab-lens-model-dev /bin/bash
```

Then inside container:
```bash
cd model-development
python scripts/validate_model.py
```

## Volume Mounts Explained

Volumes allow Docker containers to access your local files:

```bash
-v $(pwd)/model-development:/app/model-development
```

This means:
- Left side: Your local `model-development/` folder
- Right side: `/app/model-development/` inside container
- Changes in container appear on your local machine

## Troubleshooting

### Docker build fails

```bash
# Clean build (removes cache)
docker build --no-cache -t lab-lens-model-dev -f model-development/Dockerfile .
```

### Permission errors on Linux

```bash
# Add your user to docker group
sudo usermod -aG docker $USER
# Logout and login again
```

### Port 5000 already in use

```bash
# Use different port
docker-compose up mlflow -p 5001:5000
```

### Container can't find files

Make sure you're running from `lab-lens` root directory!

## Team Collaboration

### Share Image

```bash
# Save image
docker save lab-lens-model-dev > lab-lens-model-dev.tar

# Load on another machine
docker load < lab-lens-model-dev.tar
```

### Push to Docker Hub (Optional)

```bash
# Tag image
docker tag lab-lens-model-dev yourusername/lab-lens-model-dev:v1.0

# Push
docker push yourusername/lab-lens-model-dev:v1.0

# Team members pull
docker pull yourusername/lab-lens-model-dev:v1.0
```

## Benefits of Docker

✅ **Reproducibility** - Same environment for everyone
✅ **Isolation** - No conflicts with other projects
✅ **Portability** - Works on Mac, Windows, Linux
✅ **Easy setup** - One command to get started
✅ **Version control** - Tag and share images

## Quick Reference

```bash
# Build
docker build -t lab-lens-model-dev -f model-development/Dockerfile .

# MLflow UI
docker-compose up mlflow

# Validation
docker-compose run validation

# Training
docker-compose up model-training

# Clean up
docker-compose down
docker system prune -a  # Remove unused images/containers
```

---

**Docker setup complete!** Your team can now run everything in containers.