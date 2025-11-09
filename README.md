# Multivariate Probabilistic Time Series forecasting for Neural Dynamics

A minimal implementation of TimeGrad, a diffusion-based probabilistic forecasting model for time series data.

## Overview

This implementation uses denoising diffusion probabilistic models (DDPM) to learn the dynamics of 2D Ornstein-Uhlenbeck processes. The model learns to predict future states by combining a linear recurrent neural network with a diffusion-based generative process. The goal is to extend this to real-world neural datatsets. this repo exists as a proof-of-concept for mutlivariate probabilistic trajectory generation. 

## Architecture

### Components

1. **DiffusionModel**: Implements DDPM sampling with 50 diffusion steps
   - Forward process: Adds Gaussian noise according to a linear schedule
   - Reverse process: Denoises using a learned neural network
   - Training: MSE loss between predicted and true noise

2. **SimpleDenoiser**: MLP-based denoiser network
   - Inputs: Noisy data, timestep, hidden state context
   - Architecture: 3-layer MLP with SiLU activations
   - Outputs: Predicted noise at each timestep

3. **LinearRNN**: Basic recurrent network
   - Implements: h_t = W @ h_{t-1}
   - Learns transition matrix W from data

4. **TimeGrad**: Combined model
   - Autoregressive forecasting using learned dynamics
   - Generates probabilistic forecasts via diffusion sampling

## Data

The model is trained on synthetic 2D Ornstein-Uhlenbeck processes:

- Mean-reverting stochastic process
- Parameters: Θ (mean reversion), μ (long-term mean), σ (volatility)
- Training data: 500 trajectories of length 25
- Time discretization: dt = 0.1

Theoretical transition matrix (discrete-time):
```
W = I - Θ × dt = [[0.8, 0.0], [0.0, 0.85]]
```

## Training

- 10,000 epochs with batch size 64
- AdamW optimizer (lr=1e-3)
- Gradient clipping (max_norm=5.0)
- Multi-step loss weighting: 1.0 × L₁ + 0.5 × L₂ + 0.25 × L₃

The training tracks both forecasting loss and the Frobenius norm error between the learned and theoretical transition matrices.

## Results

The model successfully learns the underlying dynamics, with the learned transition matrix W converging toward the theoretical values. The implementation demonstrates that diffusion models can effectively capture temporal dependencies in stochastic processes.

## Key Features

- Pure PyTorch implementation
- DDPM-based probabilistic forecasting
- Multi-step training objective
- Autoregressive sampling with running hidden state
- Matrix recovery validation


## Usage

See the Jupyter notebook for complete training and evaluation code.

## References

Based on the TimeGrad architecture for probabilistic time series forecasting using denoising diffusion models.
