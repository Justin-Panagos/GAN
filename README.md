# Django GAN

This project is a work in progress (WIP) and subject to change. It is not yet a complete or production-ready application but serves as an evolving experiment with Generative Adversarial Networks (GANs). The current implementation uses a Wasserstein GAN with Gradient Penalty (WGAN-GP) integrated with TensorFlow Datasets for image generation in a Django web app.

## Iterations
- Traditional GAN
- Wasserstein GAN (WGAN)
- Wasserstein GAN with Gradient Penalty (WGAN-GP) *(current setup with TensorFlow and PyTorch)*

## Overview
This is a Django web application that allows users to generate images of cats or dogs using a pre-trained conditional WGAN-GP model. The model is trained on the `cats_vs_dogs` dataset from TensorFlow Datasets, enabling text-conditioned image generation (e.g., "fluffy cat" or "happy dog"). The project leverages PyTorch for the GAN architecture and TensorFlow for dataset handling, running on Python 3.11.11 with Django for the web interface.

## Features
- Generate cat or dog images based on text input (e.g., "cat", "dog", or descriptive phrases like "fluffy white cat").
- Train the WGAN-GP model on the `cats_vs_dogs` dataset (25,000+ images, 192x192 resolution).
- Web interface for real-time image generation using Django.
- Support for GPU acceleration with NVIDIA GPUs (e.g., GeForce GTX 750 Ti) or CPU fallback.
- Built with Django, PyTorch (for the GAN), and TensorFlow (for datasets).

## Requirements
- Python == 3.11.11
- Django >= 5.1.6
- PyTorch >= 2.6.0
- Torchvision >= 0.21.0
- TensorFlow == 2.14.0
- TensorFlow Datasets == 4.9.7
- NumPy == 1.26.4
- Pillow >= 11.1.0
- Other dependencies listed in `pyproject.toml`

## Setup

### 1. Install Dependencies
This project uses `uv` (a fast Python package manager) for dependency management. Follow these steps:

#### Install `uv` (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
