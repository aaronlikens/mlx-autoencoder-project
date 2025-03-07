import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import time
import os
import polars as pl

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build encoder layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def __call__(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        
        # Build decoder layers (reverse of encoder)
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def __call__(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        # Reverse hidden_dims for decoder
        self.decoder = Decoder(latent_dim, hidden_dims[::-1], input_dim)
        
    def __call__(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)


class AutoencoderMLX:
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int] = [128, 64], 
        latent_dim: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "gpu"  # Default to GPU for Apple Silicon
    ):
        """
        Initialize an Autoencoder for dimensionality reduction using MLX.
        
        Args:
            input_dim (int): Dimension of input data
            hidden_dims (List[int]): List of hidden layer dimensions
            latent_dim (int): Dimension of the latent space (reduced dimension)
            learning_rate (float): Learning rate for optimizer
            weight_decay (float): Weight decay for regularization
            device (str): Device to use ('cpu' or 'gpu')
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        
        # Set default device to GPU if requested
        # MLX automatically uses the GPU when available
        if device.lower() == "gpu":
            try:
                # In MLX, we don't need to explicitly check for GPU availability
                # MLX will automatically use the GPU if available
                print("Using GPU for computation (MLX automatically uses GPU if available)")
            except Exception as e:
                print(f"Warning: Could not set GPU as default device: {e}")
                print("Falling back to CPU")
                self.device = "cpu"
        
        # Create model, loss fn, and optimizer
        self.model = Autoencoder(input_dim, hidden_dims, latent_dim)
        
        # Fixed: Create Adam optimizer without weight_decay parameter
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        
        # Training history
        self.history = {
            'loss': [],
            'val_loss': []
        }
        
        # Initialize model parameters
        mx.eval(self.model.parameters())
    
    def loss_fn(self, model, x):
        """Loss function for the autoencoder."""
        preds = model(x)
        loss = mx.mean((preds - x) ** 2)
        
        # Weight decay is handled by the optimizer in MLX
        return loss
    
    def train(
        self, 
        X_train, 
        X_val=None, 
        batch_size: int = 32, 
        epochs: int = 100, 
        verbose: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Train the autoencoder.
        
        Args:
            X_train: Training data as MLX array
            X_val: Validation data as MLX array (optional)
            batch_size (int): Batch size for training
            epochs (int): Number of epochs to train
            verbose (bool): Whether to print progress
            save_path (str): Path to save model weights (optional)
            
        Returns:
            Dict: Training history
        """
        # Get the loss and gradient function
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
        
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            start_time = time.time()
            epoch_loss = 0.0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            
            # Train on batches
            for i in range(n_batches):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, n_samples)
                X_batch = X_shuffled[batch_start:batch_end]
                
                # Compute loss and gradients
                loss, grads = loss_and_grad_fn(self.model, X_batch)
                
                # Update the model parameters
                self.optimizer.update(self.model, grads)
                
                # Force evaluation of model parameters and optimizer state
                mx.eval(self.model.parameters(), self.optimizer.state)
                
                epoch_loss += loss.item() * (batch_end - batch_start)
            
            # Calculate average loss
            epoch_loss /= n_samples
            self.history['loss'].append(epoch_loss)
            
            # Validation if provided
            val_loss = None
            if X_val is not None:
                with mx.defer_compute():
                    val_preds = self.model(X_val)
                    val_loss = mx.mean((val_preds - X_val) ** 2).item()
                    self.history['val_loss'].append(val_loss)
            
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                elapsed = time.time() - start_time
                val_info = f"- val_loss: {val_loss:.6f}" if val_loss is not None else ""
                print(f"Epoch {epoch+1}/{epochs} - {elapsed:.2f}s - loss: {epoch_loss:.6f} {val_info}")
        
        # Save model if path provided
        if save_path:
            self.save(save_path)
            
        return self.history
    
    def encode(self, X):
        """Encode data to latent space."""
        return self.model.encode(X)
    
    def decode(self, Z):
        """Decode data from latent space."""
        return self.model.decode(Z)
    
    def transform(self, X):
        """Transform data to latent space (alias for encode)."""
        return self.encode(X)
    
    def fit_transform(self, X_train, X_val=None, **train_kwargs):
        """Fit the model and transform the data in one step."""
        self.train(X_train, X_val, **train_kwargs)
        return self.transform(X_train)
    
    def save(self, path: str):
        """Save model parameters."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        params_dict = dict(self.model.parameters().items())
        
        # Convert each parameter to an MLX array if it's not already
        params_arrays = {}
        for k, v in params_dict.items():
            params_arrays[k] = v
        
        # Use savez to save the dictionary of arrays
        mx.savez(path, **params_arrays)
        
    def load(self, path: str):
        """Load model parameters."""
        loaded = mx.load(path)
        self.model.update(loaded)
        mx.eval(self.model.parameters())
        
    def save_csv(self, data, output_folder: str, filename: str = "latent_representation.csv"):
        """
        Save the reduced dimension latent matrix to a CSV file using polars.
        
        Args:
            data: Input data as MLX array or numpy array
            output_folder (str): Folder to save the CSV file
            filename (str): Name of the CSV file
            
        Returns:
            str: Path to the saved CSV file
        """
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # If input is an MLX array, convert to numpy
        if isinstance(data, mx.array):
            data_np = data.tolist()
        elif isinstance(data, np.ndarray):
            data_np = data
        else:
            data_np = np.array(data)
        
        # Generate latent representation
        latent_data = self.transform(mx.array(data_np))
        
        # Convert to numpy array
        latent_np = latent_data.tolist()
        
        # Create column names for latent dimensions
        cols = [f"dim_{i}" for i in range(1, self.latent_dim + 1)]
        
        # Create Polars DataFrame
        df = pl.DataFrame(latent_np, schema=cols)
        
        # Save to CSV
        output_path = os.path.join(output_folder, filename)
        df.write_csv(output_path)
        
        print(f"Saved latent representation to {output_path}")
        print(f"Dimensions: {df.shape}")
        
        return output_path