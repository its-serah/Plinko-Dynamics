"""Enhanced AI Models for Quantum Galton Board Analysis

This module contains neural network models adapted from the Qnode project
for analyzing and optimizing quantum Galton board simulations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional

try:
    from torchdiffeq import odeint
except ImportError:
    print("Warning: torchdiffeq not installed. Install with: pip install torchdiffeq")
    odeint = None

class LatentODEfunc(nn.Module):
    """Neural ODE function for modeling quantum trajectory dynamics."""
    
    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, nhidden)
        self.fc4 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        out = self.elu(out)
        out = self.fc4(out)
        return out

class RecognitionRNN(nn.Module):
    """RNN for encoding quantum trajectories into latent space."""
    
    def __init__(self, latent_dim=4, obs_dim=3, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self, batch=0):
        if batch == 0:
            return torch.zeros(self.nbatch, self.nhidden)
        else:
            return torch.zeros(batch, self.nhidden)

class Decoder(nn.Module):
    """Decoder for reconstructing quantum distributions from latent space."""
    
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20, extra=False):
        super(Decoder, self).__init__()
        self.extra = extra
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.tanh(out)
        if self.extra:
            out = self.fc2(out)
            out = self.tanh(out)
        out = self.fc3(out)
        return out

def log_normal_pdf(x, mean, logvar):
    """Compute log probability density function of normal distribution."""
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))

def normal_kl(mu1, lv1, mu2, lv2):
    """Compute KL divergence between two normal distributions."""
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.
    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

class QuantumGaltonAI:
    """AI model for learning quantum Galton board dynamics."""
    
    def __init__(self, 
                 obs_dim=7,  # Max bins for distributions
                 latent_dim=6, 
                 nhidden=32, 
                 rnn_nhidden=25, 
                 lr=1e-3, 
                 batch=100, 
                 beta=1, 
                 extra_decode=True):
        
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.nhidden = nhidden
        self.rnn_nhidden = rnn_nhidden
        self.beta = beta
        self.epsilon = None
        
        # Initialize networks
        if odeint is not None:
            self.func = LatentODEfunc(latent_dim, nhidden)
        else:
            self.func = None
            print("Warning: ODE functionality disabled without torchdiffeq")
            
        self.rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, batch)
        self.dec = Decoder(latent_dim, obs_dim, nhidden, extra=extra_decode)
        
        # Optimizer
        params = list(self.dec.parameters()) + list(self.rec.parameters())
        if self.func is not None:
            params += list(self.func.parameters())
        self.params = params
        self.optimizer = optim.Adam(self.params, lr=lr)
    
    def train(self, trajs, ts, num_epochs=50):
        """Train the AI model on quantum trajectory data."""
        if self.func is None:
            print("Cannot train without torchdiffeq. Install with: pip install torchdiffeq")
            return
            
        num_ts = ts.size(0)
        beta = self.beta
        
        print(f"Training AI model for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            self.optimizer.zero_grad()
            
            # Encode trajectories
            h = self.rec.initHidden(batch=trajs.shape[0])
            for t in reversed(range(num_ts)):
                obs = trajs[:, t, :]
                out, h = self.rec.forward(obs, h)
            
            qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
            epsilon = torch.randn(qz0_mean.size())
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            
            # Forward through ODE and decode
            pred_z = odeint(self.func, z0, ts).permute(1, 0, 2)
            pred_x = self.dec(pred_z)
            
            # Compute loss
            noise_std_ = torch.zeros(pred_x.size()) + 0.1
            noise_logvar = 2. * torch.log(noise_std_)
            logpx = log_normal_pdf(trajs, pred_x, noise_logvar).sum(-1).sum(-1)
            
            pz0_mean = pz0_logvar = torch.zeros(z0.size())
            analytic_kl = beta * normal_kl(qz0_mean, qz0_logvar, 
                                         pz0_mean, pz0_logvar).sum(-1)
            
            loss = torch.mean(-logpx + analytic_kl, dim=0)
            loss.backward()
            self.optimizer.step()
            
            if epoch == num_epochs:
                self.epsilon = epsilon
            
            if epoch % 10 == 0 or epoch == 1:
                print(f'Epoch: {epoch}, Loss: {loss.item():.6f}')
    
    def encode(self, trajs, ts):
        """Encode trajectories into latent space."""
        with torch.no_grad():
            num_ts = ts.size(0)
            h = self.rec.initHidden(batch=trajs.shape[0])
            for t in reversed(range(num_ts)):
                obs = trajs[:, t, :]
                out, h = self.rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
            z0 = qz0_mean  # Use mean for deterministic encoding
        return z0
    
    def decode(self, z0, ts):
        """Decode latent states back to trajectory space."""
        if self.func is None:
            # Simple linear decode without ODE
            with torch.no_grad():
                pred_x = self.dec(z0.unsqueeze(1).repeat(1, ts.size(0), 1))
            return pred_x
        
        with torch.no_grad():
            if len(z0.shape) == 1:
                pred_z = odeint(self.func, z0, ts)
            else:
                pred_z = odeint(self.func, z0, ts).permute(1, 0, 2)
            pred_x = self.dec(pred_z)
        return pred_x
    
    def predict_distribution(self, parameters, ts):
        """Predict quantum distribution based on circuit parameters."""
        # Simple prediction without full trajectory
        # This could be enhanced based on specific needs
        with torch.no_grad():
            # Create a simple mapping from parameters to latent space
            z0 = torch.randn(1, self.latent_dim) * 0.1
            pred_x = self.decode(z0, ts)
        return pred_x
    
    def save_model(self, path_prefix="quantum_galton_ai"):
        """Save the trained model."""
        torch.save(self.rec.state_dict(), f'{path_prefix}_rec.pt')
        torch.save(self.dec.state_dict(), f'{path_prefix}_dec.pt')
        if self.func is not None:
            torch.save(self.func.state_dict(), f'{path_prefix}_func.pt')
        if self.epsilon is not None:
            torch.save(self.epsilon, f'{path_prefix}_epsilon.pt')
        print(f"Model saved with prefix: {path_prefix}")
    
    def load_model(self, path_prefix="quantum_galton_ai"):
        """Load a previously trained model."""
        self.rec.load_state_dict(torch.load(f'{path_prefix}_rec.pt'))
        self.dec.load_state_dict(torch.load(f'{path_prefix}_dec.pt'))
        if self.func is not None:
            self.func.load_state_dict(torch.load(f'{path_prefix}_func.pt'))
        try:
            self.epsilon = torch.load(f'{path_prefix}_epsilon.pt')
        except FileNotFoundError:
            pass
        
        self.rec.eval()
        self.dec.eval()
        if self.func is not None:
            self.func.eval()
        print(f"Model loaded from prefix: {path_prefix}")

# Convenience functions
def train_ai_on_galton_data(dataset, num_epochs=50, save_path="quantum_galton_ai"):
    """Train AI model on quantum Galton board dataset."""
    trajectories = dataset.trajectories
    time_steps = torch.linspace(0, 1, trajectories.shape[1])
    
    # Initialize model
    model = QuantumGaltonAI(
        obs_dim=trajectories.shape[-1],
        latent_dim=6,
        nhidden=32,
        rnn_nhidden=25,
        lr=1e-3
    )
    
    # Train
    model.train(trajectories, time_steps, num_epochs)
    
    # Save
    model.save_model(save_path)
    
    return model
