#!/usr/bin/env bash
set -e

pip install --upgrade pip

# Install CPU-only PyTorch FIRST (no CUDA/NVIDIA libs = ~180MB vs ~3GB)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps — sentence-transformers finds torch already present, skips reinstall
pip install -r requirements.txt
