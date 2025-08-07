"""
Enhanced AI Models for Quantum Galton Board Analysis

This module contains robust neural network models for analyzing and optimizing 
quantum Galton board simulations with comprehensive error handling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import torchdiffeq
try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    warnings.warn(
        "torchdiffeq not installed. Neural ODE functionality will be limited. "
        "Install with: pip install torchdiffeq"
    )
    TORCHDIFFEQ_AVAILABLE = False
    odeint = None


class LatentODEFunc(nn.Module):
    """Neural ODE function for modeling quantum trajectory dynamics."""
    
    def __init__(self, latent_dim: int = 4, nhidden: int = 20):
        """
        Initialize the ODE function.
        
        Args:
            latent_dim: Dimension of latent space
            nhidden: Number of hidden units
        """
        super(LatentODEFunc, self).__init__()
        if latent_dim <= 0 or nhidden <= 0:
            raise ValueError("latent_dim and nhidden must be positive")
            
        self.latent_dim = latent_dim
        self.nhidden = nhidden
        
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, nhidden)
        self.fc4 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, t, x):
        """Forward pass through the ODE function."""
        if x.isnan().any():
            logger.warning("NaN detected in ODE input")
        
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        out = self.elu(out)
        out = self.fc4(out)
        
        return out
    
    def _initialize_weights(self):
        """Initialize network weights with Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


class RecognitionRNN(nn.Module):
    """RNN for encoding quantum trajectories into latent space."""
    
    def __init__(self, latent_dim: int = 4, obs_dim: int = 3, 
                 nhidden: int = 25, nbatch: int = 1):
        """
        Initialize the recognition RNN.
        
        Args:
            latent_dim: Dimension of latent space
            obs_dim: Dimension of observations
            nhidden: Number of hidden units
            nbatch: Batch size
        """
        super(RecognitionRNN, self).__init__()
        
        if any(x <= 0 for x in [latent_dim, obs_dim, nhidden, nbatch]):
            raise ValueError("All dimensions must be positive")
            
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.nhidden = nhidden
        self.nbatch = nbatch
        
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x, h):
        """Forward pass through the RNN."""
        if x.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected observation dim {self.obs_dim}, got {x.shape[-1]}")
            
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self, batch: int = 0):
        """Initialize hidden state."""
        batch_size = batch if batch > 0 else self.nbatch
        return torch.zeros(batch_size, self.nhidden)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


class Decoder(nn.Module):
    """Decoder for reconstructing quantum distributions from latent space."""
    
    def __init__(self, latent_dim: int = 4, obs_dim: int = 2, 
                 nhidden: int = 20, extra_layer: bool = False):
        """
        Initialize the decoder.
        
        Args:
            latent_dim: Dimension of latent space
            obs_dim: Dimension of observations
            nhidden: Number of hidden units
            extra_layer: Whether to add an extra hidden layer
        """
        super(Decoder, self).__init__()
        
        if any(x <= 0 for x in [latent_dim, obs_dim, nhidden]):
            raise ValueError("All dimensions must be positive")
            
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.extra_layer = extra_layer
        
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(latent_dim, nhidden)
        if extra_layer:
            self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, obs_dim)
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, z):
        """Forward pass through the decoder."""
        out = self.fc1(z)
        out = self.tanh(out)
        if self.extra_layer:
            out = self.fc2(out)
            out = self.tanh(out)
        out = self.fc3(out)
        return out
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


def log_normal_pdf(x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute log probability density function of normal distribution."""
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1: torch.Tensor, lv1: torch.Tensor, 
              mu2: torch.Tensor, lv2: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two normal distributions."""
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.
    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


class QuantumGaltonAI:
    """
    Enhanced AI model for learning quantum Galton board dynamics.
    
    This class provides a complete neural network framework for analyzing
    quantum trajectories with robust error handling and validation.
    """
    
    def __init__(self, 
                 obs_dim: int = 7,
                 latent_dim: int = 6, 
                 nhidden: int = 32, 
                 rnn_nhidden: int = 25, 
                 lr: float = 1e-3, 
                 batch_size: int = 100, 
                 beta: float = 1.0, 
                 extra_decode: bool = True,
                 device: str = 'auto'):
        """
        Initialize the AI model.
        
        Args:
            obs_dim: Maximum bins for distributions
            latent_dim: Dimension of latent space
            nhidden: Hidden units in decoder/ODE
            rnn_nhidden: Hidden units in RNN
            lr: Learning rate
            batch_size: Batch size for training
            beta: Beta parameter for KL weighting
            extra_decode: Whether to use extra decoder layer
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        # Validate inputs
        if any(x <= 0 for x in [obs_dim, latent_dim, nhidden, rnn_nhidden, batch_size]):
            raise ValueError("All dimension and batch parameters must be positive")
        if not (0 < lr < 1):
            raise ValueError("Learning rate must be between 0 and 1")
        if beta < 0:
            raise ValueError("Beta must be non-negative")
        
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.nhidden = nhidden
        self.rnn_nhidden = rnn_nhidden
        self.lr = lr
        self.batch_size = batch_size
        self.beta = beta
        self.epsilon = None
        self.training_history = []
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize networks
        try:
            if TORCHDIFFEQ_AVAILABLE:
                self.func = LatentODEFunc(latent_dim, nhidden).to(self.device)
            else:
                self.func = None
                logger.warning("Neural ODE functionality disabled without torchdiffeq")
                
            self.rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, batch_size).to(self.device)
            self.dec = Decoder(latent_dim, obs_dim, nhidden, extra_layer=extra_decode).to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to initialize neural networks: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
        
        # Setup optimizer
        params = list(self.dec.parameters()) + list(self.rec.parameters())
        if self.func is not None:
            params += list(self.func.parameters())
        
        self.optimizer = optim.Adam(params, lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        logger.info(f"Initialized QuantumGaltonAI with {sum(p.numel() for p in params)} parameters")
    
    def train(self, trajectories: torch.Tensor, time_steps: torch.Tensor, 
              num_epochs: int = 50, validate_every: int = 10) -> Dict[str, Any]:
        """
        Train the AI model on quantum trajectory data.
        
        Args:
            trajectories: Training trajectory data
            time_steps: Time step array
            num_epochs: Number of training epochs
            validate_every: Validation frequency
            
        Returns:
            Training history dictionary
        """
        if self.func is None:
            raise RuntimeError("Cannot train without torchdiffeq. Install with: pip install torchdiffeq")
        
        if trajectories.dim() != 3:
            raise ValueError(f"Trajectories should have 3 dimensions, got {trajectories.dim()}")
        
        # Move to device
        trajectories = trajectories.to(self.device)
        time_steps = time_steps.to(self.device)
        
        num_ts = time_steps.size(0)
        beta = self.beta
        
        logger.info(f"Training AI model for {num_epochs} epochs on {self.device}")
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            try:
                self.optimizer.zero_grad()
                
                # Encode trajectories
                h = self.rec.initHidden(batch=trajectories.shape[0]).to(self.device)
                for t in reversed(range(num_ts)):
                    obs = trajectories[:, t, :]
                    out, h = self.rec.forward(obs, h)
                
                qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
                
                # Sample from latent distribution
                epsilon = torch.randn(qz0_mean.size(), device=self.device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                
                # Forward through ODE and decode
                pred_z = odeint(self.func, z0, time_steps).permute(1, 0, 2)
                pred_x = self.dec(pred_z)
                
                # Compute loss
                noise_std = torch.zeros(pred_x.size(), device=self.device) + 0.1
                noise_logvar = 2. * torch.log(noise_std)
                logpx = log_normal_pdf(trajectories, pred_x, noise_logvar).sum(-1).sum(-1)
                
                pz0_mean = torch.zeros(z0.size(), device=self.device)
                pz0_logvar = torch.zeros(z0.size(), device=self.device)
                analytic_kl = beta * normal_kl(qz0_mean, qz0_logvar, 
                                             pz0_mean, pz0_logvar).sum(-1)
                
                loss = torch.mean(-logpx + analytic_kl, dim=0)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    logger.warning(f"NaN loss detected at epoch {epoch}")
                    break
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.rec.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.dec.parameters(), max_norm=1.0)
                if self.func is not None:
                    torch.nn.utils.clip_grad_norm_(self.func.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step(loss)
                
                # Store training history
                self.training_history.append({
                    'epoch': epoch,
                    'loss': loss.item(),
                    'kl_divergence': analytic_kl.mean().item(),
                    'reconstruction_loss': -logpx.mean().item()
                })
                
                if epoch == num_epochs:
                    self.epsilon = epsilon
                
                # Logging
                if epoch % validate_every == 0 or epoch == 1:
                    logger.info(f'Epoch: {epoch}/{num_epochs}, Loss: {loss.item():.6f}, '
                               f'KL: {analytic_kl.mean().item():.6f}, '
                               f'Recon: {-logpx.mean().item():.6f}')
                    
            except Exception as e:
                logger.error(f"Error during training at epoch {epoch}: {e}")
                break
        
        return {
            'training_history': self.training_history,
            'final_loss': self.training_history[-1]['loss'] if self.training_history else None
        }
    
    def encode(self, trajectories: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Encode trajectories into latent space."""
        trajectories = trajectories.to(self.device)
        time_steps = time_steps.to(self.device)
        
        with torch.no_grad():
            num_ts = time_steps.size(0)
            h = self.rec.initHidden(batch=trajectories.shape[0]).to(self.device)
            for t in reversed(range(num_ts)):
                obs = trajectories[:, t, :]
                out, h = self.rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
            z0 = qz0_mean  # Use mean for deterministic encoding
        return z0.cpu()
    
    def decode(self, z0: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Decode latent states back to trajectory space."""
        z0 = z0.to(self.device)
        time_steps = time_steps.to(self.device)
        
        if self.func is None:
            # Simple linear decode without ODE
            with torch.no_grad():
                if len(z0.shape) == 1:
                    z0 = z0.unsqueeze(0)
                pred_x = self.dec(z0.unsqueeze(1).repeat(1, time_steps.size(0), 1))
            return pred_x.cpu()
        
        with torch.no_grad():
            try:
                if len(z0.shape) == 1:
                    pred_z = odeint(self.func, z0, time_steps)
                else:
                    pred_z = odeint(self.func, z0, time_steps).permute(1, 0, 2)
                pred_x = self.dec(pred_z)
            except Exception as e:
                logger.error(f"Error during decoding: {e}")
                # Fallback to simple decoding
                if len(z0.shape) == 1:
                    z0 = z0.unsqueeze(0)
                pred_x = self.dec(z0.unsqueeze(1).repeat(1, time_steps.size(0), 1))
                
        return pred_x.cpu()
    
    def predict_distribution(self, parameters: torch.Tensor, 
                           time_steps: torch.Tensor) -> torch.Tensor:
        """Predict quantum distribution based on circuit parameters."""
        with torch.no_grad():
            # Create a simple mapping from parameters to latent space
            # This is a placeholder - could be enhanced with parameter encoding
            z0 = torch.randn(1, self.latent_dim, device=self.device) * 0.1
            pred_x = self.decode(z0, time_steps)
        return pred_x
    
    def save_model(self, path_prefix: str = "quantum_galton_ai"):
        """Save the trained model."""
        try:
            torch.save({
                'rec_state_dict': self.rec.state_dict(),
                'dec_state_dict': self.dec.state_dict(),
                'func_state_dict': self.func.state_dict() if self.func else None,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_history': self.training_history,
                'model_config': {
                    'obs_dim': self.obs_dim,
                    'latent_dim': self.latent_dim,
                    'nhidden': self.nhidden,
                    'rnn_nhidden': self.rnn_nhidden,
                    'lr': self.lr,
                    'batch_size': self.batch_size,
                    'beta': self.beta
                }
            }, f'{path_prefix}_full_model.pt')
            
            # Save individual components for compatibility
            torch.save(self.rec.state_dict(), f'{path_prefix}_rec.pt')
            torch.save(self.dec.state_dict(), f'{path_prefix}_dec.pt')
            if self.func is not None:
                torch.save(self.func.state_dict(), f'{path_prefix}_func.pt')
            if self.epsilon is not None:
                torch.save(self.epsilon, f'{path_prefix}_epsilon.pt')
                
            logger.info(f"Model saved with prefix: {path_prefix}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, path_prefix: str = "quantum_galton_ai"):
        """Load a previously trained model."""
        try:
            # Try to load full model first
            try:
                checkpoint = torch.load(f'{path_prefix}_full_model.pt', map_location=self.device)
                self.rec.load_state_dict(checkpoint['rec_state_dict'])
                self.dec.load_state_dict(checkpoint['dec_state_dict'])
                if self.func is not None and checkpoint['func_state_dict'] is not None:
                    self.func.load_state_dict(checkpoint['func_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.training_history = checkpoint.get('training_history', [])
                logger.info(f"Loaded full model from {path_prefix}_full_model.pt")
                
            except FileNotFoundError:
                # Fall back to individual files
                self.rec.load_state_dict(torch.load(f'{path_prefix}_rec.pt', map_location=self.device))
                self.dec.load_state_dict(torch.load(f'{path_prefix}_dec.pt', map_location=self.device))
                if self.func is not None:
                    try:
                        self.func.load_state_dict(torch.load(f'{path_prefix}_func.pt', map_location=self.device))
                    except FileNotFoundError:
                        logger.warning("ODE function weights not found")
                
                try:
                    self.epsilon = torch.load(f'{path_prefix}_epsilon.pt', map_location=self.device)
                except FileNotFoundError:
                    pass
                
                logger.info(f"Loaded individual model components from {path_prefix}_*.pt")
            
            # Set to eval mode
            self.rec.eval()
            self.dec.eval()
            if self.func is not None:
                self.func.eval()
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        param_count = sum(p.numel() for p in self.rec.parameters())
        param_count += sum(p.numel() for p in self.dec.parameters())
        if self.func is not None:
            param_count += sum(p.numel() for p in self.func.parameters())
        
        return {
            'obs_dim': self.obs_dim,
            'latent_dim': self.latent_dim,
            'nhidden': self.nhidden,
            'rnn_nhidden': self.rnn_nhidden,
            'total_parameters': param_count,
            'device': str(self.device),
            'ode_available': self.func is not None,
            'training_epochs': len(self.training_history),
            'final_loss': self.training_history[-1]['loss'] if self.training_history else None
        }
