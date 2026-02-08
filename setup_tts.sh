#!/bin/bash
# Setup script for Qwen3-TTS Multi-Voice Audiobook Generator

set -e

echo "=== Qwen3-TTS Setup ==="
echo ""

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Miniconda or Anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "Creating conda environment 'qwen3-tts' with Python 3.12..."
conda create -n qwen3-tts python=3.12 -y

# Activate environment
echo ""
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3-tts

# Install PyTorch (detect GPU availability)
echo ""
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, installing CUDA version..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
else
    echo "No NVIDIA GPU detected, installing CPU version..."
    pip install torch torchaudio
fi

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Optional: Flash Attention (GPU only)
if command -v nvidia-smi &> /dev/null; then
    echo ""
    read -p "Install Flash Attention 2 for better GPU performance? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install flash-attn --no-build-isolation
    fi
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To use the TTS generator:"
echo "  1. Activate the environment: conda activate qwen3-tts"
echo "  2. Run the generator: python tts_generator.py --help"
echo ""
echo "Quick start commands:"
echo "  python tts_generator.py --list-characters    # See all characters"
echo "  python tts_generator.py --dry-run            # Test parsing"
echo "  python tts_generator.py --chapter 1          # Process chapter 1"
echo "  python tts_generator.py --assemble           # Generate & combine audio"
