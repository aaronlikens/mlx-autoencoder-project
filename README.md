# MLX Autoencoder for Dimensionality Reduction

This project implements an autoencoder using Apple's MLX framework for dimensionality reduction tasks. The implementation is optimized for performance and can run on both CPU and GPU (if available).

## Project Structure

```
mlx_autoencoder_project/
├── autoencoder_mlx/               # Main package
│   ├── __init__.py                # Package initialization
│   └── autoencoder_mlx.py         # Autoencoder implementation
├── data/                          # Directory for your data files
│   └── S01_RUN1_VoxelExport.csv   # Example data file (not included)
├── models/                        # Directory where trained models will be saved
├── example_usage.py               # Example script to demonstrate usage
└── README.md                      # This file
```

## Requirements

- Python 3.8+
- mlx
- polars
- numpy
- matplotlib

## Installation

Clone this repository and install the required packages:

```bash
pip install mlx polars numpy matplotlib
```

## Usage

1. Place your CSV data files in the `data/` directory.
2. Run the example script:

```bash
python example_usage.py
```

Or import the autoencoder in your own scripts:

```python
from autoencoder_mlx import AutoencoderMLX

# Define your architecture
autoencoder = AutoencoderMLX(
    input_dim=100,  # Your data dimension
    hidden_dims=[128, 64],
    latent_dim=32
)

# Train the model
history = autoencoder.train(
    X_train=your_training_data,
    X_val=your_validation_data,
    batch_size=64,
    epochs=100
)

# Transform data to latent space
latent_representations = autoencoder.transform(your_data)
```

## Features

- Modular architecture with separate Encoder and Decoder components
- Efficient training with batching and validation
- Progress tracking during training
- Model saving and loading
- Fast data loading with Polars
- GPU support (if available via MLX)

## Customization

You can customize the architecture by modifying the `hidden_dims` parameter:

```python
# Deep architecture
autoencoder = AutoencoderMLX(
    input_dim=100,
    hidden_dims=[512, 256, 128, 64],
    latent_dim=32
)

# Shallow architecture
autoencoder = AutoencoderMLX(
    input_dim=100,
    hidden_dims=[64],
    latent_dim=16
)
```