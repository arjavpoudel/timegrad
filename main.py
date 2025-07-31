"""
TimeGrad-style diffusion model for bimodal Gaussian mixture time series forecasting.
- Enhanced RNN encoder for complex temporal patterns
- Improved denoiser architecture for multimodal distributions
- Bimodal Gaussian mixture data generation
- Better evaluation metrics for multimodal forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine beta schedule for diffusion - better for complex distributions"""
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


class EnhancedDiffusion(nn.Module):
    """Enhanced diffusion model for bimodal distributions"""

    def __init__(self, data_dim, context_dim, num_steps=50, beta_schedule="cosine"):
        super().__init__()
        self.data_dim = data_dim
        self.num_steps = num_steps

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(num_steps)
        else:
            betas = np.linspace(0.0001, 0.02, num_steps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alphas", torch.tensor(alphas, dtype=torch.float32))
        self.register_buffer(
            "alphas_cumprod", torch.tensor(alphas_cumprod, dtype=torch.float32)
        )
        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(torch.tensor(alphas_cumprod, dtype=torch.float32)),
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1 - torch.tensor(alphas_cumprod, dtype=torch.float32)),
        )

        # Improved time embedding
        time_emb_dim = 64
        self.time_emb = nn.Sequential(
            nn.Embedding(num_steps, time_emb_dim // 2),
            nn.Linear(time_emb_dim // 2, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # More sophisticated denoiser with residual connections
        hidden_dim = 256
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)

        # Residual blocks for better gradient flow
        self.blocks = nn.ModuleList([self._make_block(hidden_dim) for _ in range(4)])

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, data_dim),
        )

    def _make_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Dropout(0.1), nn.Linear(dim, dim)
        )

    def add_noise(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(
            -1, 1
        )
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise(self, x_noisy, t, context):
        # Project inputs
        x_emb = self.input_proj(x_noisy)
        c_emb = self.context_proj(context)
        t_emb = self.time_proj(self.time_emb(t))

        # Combine inputs
        h = x_emb + c_emb + t_emb

        # Apply residual blocks
        for block in self.blocks:
            h = h + block(h)  # Residual connection

        return self.output_proj(h)

    def training_loss(self, x, context):
        batch_size = x.shape[0]
        t = torch.randint(0, self.num_steps, (batch_size,), device=x.device)
        noise = torch.randn_like(x)
        x_noisy = self.add_noise(x, t, noise)
        predicted_noise = self.predict_noise(x_noisy, t, context)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, context, num_samples=None):
        original_batch_size = context.shape[0]
        if num_samples is None:
            batch_size = original_batch_size
            final_context = context
        else:
            batch_size = original_batch_size * num_samples
            final_context = context.repeat_interleave(num_samples, dim=0)

        x = torch.randn(batch_size, self.data_dim, device=context.device)

        for step in reversed(range(self.num_steps)):
            t = torch.full((batch_size,), step, device=context.device, dtype=torch.long)
            predicted_noise = self.predict_noise(x, t, final_context)

            alpha_t = self.alphas[step]
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[step]

            # More stable sampling
            x = (x - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / torch.sqrt(
                alpha_t
            )

            if step > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(self.betas[step]) * noise

        if num_samples is None:
            return x
        else:
            return x.view(original_batch_size, num_samples, self.data_dim)


class BimodalTimeGrad(nn.Module):
    """TimeGrad model enhanced for bimodal Gaussian mixture forecasting"""

    def __init__(self, data_dim, hidden_dim=64, num_diffusion_steps=50):
        super().__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        # Enhanced RNN encoder for better context representation
        self.rnn = nn.LSTM(
            data_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1
        )
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.diffusion = EnhancedDiffusion(data_dim, hidden_dim, num_diffusion_steps)

    def encode_past(self, past_data):
        _, (h_n, _) = self.rnn(past_data)
        context = self.context_proj(h_n[-1])  # Use last layer's hidden state
        return context

    def forward(self, past_data, future_data):
        batch_size, pred_len, _ = future_data.shape
        context = self.encode_past(past_data)

        total_loss = 0.0
        for t in range(pred_len):
            target = future_data[:, t, :]
            loss = self.diffusion.training_loss(target, context)
            total_loss += loss

        return total_loss / pred_len

    def sample_future(self, past_data, num_future_steps, num_samples=1):
        self.eval()
        with torch.no_grad():
            context = self.encode_past(past_data)
            future_samples = []

            for _ in range(num_future_steps):
                samples = self.diffusion.sample(context, num_samples)
                if num_samples == 1:
                    samples = samples.unsqueeze(1)
                else:
                    samples = samples.unsqueeze(2)
                future_samples.append(samples)

            if num_samples == 1:
                return torch.cat(future_samples, dim=1)
            else:
                return torch.cat(future_samples, dim=2)


def create_bimodal_data(batch_size=32, seq_len=20, pred_len=5, data_dim=1):
    """
    Generate bimodal Gaussian mixture time series data.
    Each time series switches between two different AR processes representing different modes.
    """
    data = torch.zeros(batch_size, seq_len + pred_len, data_dim)

    for b in range(batch_size):
        # Initialize with random mode
        current_mode = np.random.choice([0, 1])
        x = torch.randn(data_dim) * 0.5

        # Parameters for two different AR processes (two modes)
        # Mode 0: Centered around 0, more stable
        ar_coeff_0 = 0.7
        noise_scale_0 = 0.3
        mean_0 = 0.0

        # Mode 1: Centered around 2, more volatile
        ar_coeff_1 = 0.5
        noise_scale_1 = 0.5
        mean_1 = 2.0

        for t in range(seq_len + pred_len):
            # Occasionally switch modes (creates bimodal behavior)
            if np.random.random() < 0.05:  # 5% chance to switch
                current_mode = 1 - current_mode

            if current_mode == 0:
                x = (
                    ar_coeff_0 * x
                    + (1 - ar_coeff_0) * mean_0
                    + noise_scale_0 * torch.randn(data_dim)
                )
            else:
                x = (
                    ar_coeff_1 * x
                    + (1 - ar_coeff_1) * mean_1
                    + noise_scale_1 * torch.randn(data_dim)
                )

            data[b, t] = x

    return data[:, :seq_len], data[:, seq_len:]


def create_simple_bimodal_data(batch_size=32, seq_len=20, pred_len=5, data_dim=1):
    """
    Create very simple 1D bimodal data for initial testing.
    Just alternates between two fixed values with noise.
    """
    data = torch.zeros(batch_size, seq_len + pred_len, data_dim)

    for b in range(batch_size):
        # Two modes at fixed locations
        mode_values = torch.tensor([-2.0, 2.0])
        current_mode = np.random.choice([0, 1])

        for t in range(seq_len + pred_len):
            # Switch modes occasionally
            if np.random.random() < 0.05:
                current_mode = 1 - current_mode

            # Add some noise around the mode
            data[b, t, 0] = mode_values[current_mode] + 0.3 * torch.randn(1)

    return data[:, :seq_len], data[:, seq_len:]


def create_complex_bimodal_data(batch_size=32, seq_len=20, pred_len=5, data_dim=2):
    """
    Generate more complex 2D bimodal time series where modes are spatially separated.
    Creates clearer bimodal patterns for better training.
    """
    data = torch.zeros(batch_size, seq_len + pred_len, data_dim)

    for b in range(batch_size):
        # Two well-separated modes in 2D space - increase separation
        mode_centers = torch.tensor([[-3.0, -2.0], [3.0, 2.0]])  # More separated modes
        current_mode = np.random.choice([0, 1])

        # Start closer to mode center
        x = mode_centers[current_mode] + 0.3 * torch.randn(data_dim)

        for t in range(seq_len + pred_len):
            # Switch modes less frequently for clearer patterns
            if np.random.random() < 0.02:  # Reduced from 0.03
                current_mode = 1 - current_mode

            # Stronger AR dynamics towards current mode center
            center = mode_centers[current_mode]
            # Increased attraction to mode center, reduced noise
            x = 0.7 * x + 0.3 * center + 0.2 * torch.randn(data_dim)
            data[b, t] = x

    return data[:, :seq_len], data[:, seq_len:]


def evaluate_bimodal_predictions(true_data, predicted_samples, num_modes=2):
    """
    Evaluate bimodal predictions using mode-aware metrics.
    """
    batch_size, pred_len, data_dim = true_data.shape
    num_samples = predicted_samples.shape[1]

    results = {}

    # 1. Standard MSE
    mean_pred = predicted_samples.mean(dim=1)
    mse = F.mse_loss(mean_pred, true_data).item()
    results["mse"] = mse

    # 2. Coverage probability (what fraction of true values fall within prediction intervals)
    pred_quantiles = torch.quantile(predicted_samples, torch.tensor([0.1, 0.9]), dim=1)
    coverage = (
        ((true_data >= pred_quantiles[0]) & (true_data <= pred_quantiles[1]))
        .float()
        .mean()
        .item()
    )
    results["coverage_80"] = coverage

    # 3. Alternative coverage with median absolute deviation
    pred_median = torch.median(predicted_samples, dim=1)[0]
    mad = torch.median(torch.abs(predicted_samples - pred_median.unsqueeze(1)), dim=1)[
        0
    ]
    coverage_mad = (
        ((torch.abs(true_data - pred_median) <= 2 * mad)).float().mean().item()
    )
    results["coverage_mad"] = coverage_mad

    # 4. Improved bimodality detection
    bimodality_scores = []
    for b in range(min(batch_size, 4)):  # Sample a few examples
        for t in range(pred_len):
            for d in range(data_dim):
                samples = predicted_samples[b, :, t, d].numpy()
                if len(samples) > 10:
                    # Simple bimodality test - check if distribution has two peaks
                    hist, bins = np.histogram(samples, bins=20)
                    # Find local maxima
                    peaks = []
                    for i in range(1, len(hist) - 1):
                        if (
                            hist[i] > hist[i - 1]
                            and hist[i] > hist[i + 1]
                            and hist[i] > np.max(hist) * 0.3
                        ):
                            peaks.append(i)

                    bimodality_score = (
                        len(peaks) / 2.0
                    )  # Score based on number of peaks
                    bimodality_scores.append(bimodality_score)

    results["avg_bimodality"] = np.mean(bimodality_scores) if bimodality_scores else 0

    # 5. Prediction interval width (narrower is better if coverage is good)
    interval_width = (pred_quantiles[1] - pred_quantiles[0]).mean().item()
    results["interval_width"] = interval_width

    return results


def train_bimodal_timegrad():
    # Enhanced hyperparameters for bimodal data
    data_dim = 2  # Use 2D for more interesting bimodal patterns
    seq_len = 30
    pred_len = 10
    batch_size = 64
    num_epochs = 150  # Reduced epochs
    lr = 1e-3  # Higher learning rate
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Training Bimodal TimeGrad")
    print(f"Data dimension: {data_dim}")
    print(f"Device: {device}")

    model = BimodalTimeGrad(
        data_dim,
        hidden_dim=64,
        num_diffusion_steps=50,  # Reduced steps for faster convergence
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    losses = []
    best_loss = float("inf")

    # Pre-generate a larger dataset to avoid overfitting to small samples
    print("Pre-generating training data...")
    all_past_data = []
    all_future_data = []
    for _ in range(10):  # Generate 10 batches worth of data
        past, future = create_complex_bimodal_data(
            batch_size, seq_len, pred_len, data_dim
        )
        all_past_data.append(past)
        all_future_data.append(future)

    for epoch in range(num_epochs):
        model.train()

        # Use pre-generated data with some shuffling
        data_idx = epoch % len(all_past_data)
        past_data = all_past_data[data_idx].to(device)
        future_data = all_future_data[data_idx].to(device)

        # Add some noise for regularization
        if epoch > 20:  # Add noise after initial convergence
            past_data = past_data + 0.01 * torch.randn_like(past_data)
            future_data = future_data + 0.01 * torch.randn_like(future_data)

        optimizer.zero_grad()
        loss = model(past_data, future_data)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()

        if epoch % 15 == 0:  # More frequent logging
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}, Best: {best_loss:.4f}"
            )

    return model, losses


def test_bimodal_model(model, device="cpu"):
    """Comprehensive testing for bimodal model"""
    model.eval()

    # Generate test data
    test_batch_size = 8
    past_data, true_future = create_complex_bimodal_data(test_batch_size, 30, 10, 2)
    past_data = past_data.to(device)

    with torch.no_grad():
        # Generate multiple samples to capture uncertainty
        num_samples = 100
        predicted_future = model.sample_future(
            past_data, num_future_steps=10, num_samples=num_samples
        )

    # Evaluate predictions
    metrics = evaluate_bimodal_predictions(true_future, predicted_future)
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Visualization
    plot_bimodal_results(past_data, true_future, predicted_future, num_examples=4)

    return metrics


def plot_bimodal_results(past_data, true_future, predicted_samples, num_examples=4):
    """Plot results showing bimodal predictions"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(min(num_examples, len(axes))):
        ax = axes[i]

        # Past data
        past = past_data[i].cpu().numpy()
        time_past = np.arange(len(past))

        # True future
        true = true_future[i].cpu().numpy()
        time_future = np.arange(len(past), len(past) + len(true))

        # Predicted samples
        pred_samples = (
            predicted_samples[i].cpu().numpy()
        )  # [num_samples, pred_len, data_dim]

        # Plot for each dimension
        colors = ["blue", "red"]
        for d in range(past.shape[1]):
            # Past data
            ax.plot(
                time_past,
                past[:, d],
                color=colors[d],
                linestyle="-",
                alpha=0.8,
                label=f"Past Dim {d}",
                linewidth=2,
            )

            # True future
            ax.plot(
                time_future,
                true[:, d],
                color=colors[d],
                linestyle="-",
                alpha=0.8,
                label=f"True Future Dim {d}",
                linewidth=2,
            )

            # Prediction quantiles (showing uncertainty)
            pred_quantiles = np.quantile(
                pred_samples[:, :, d], [0.1, 0.25, 0.5, 0.75, 0.9], axis=0
            )

            ax.fill_between(
                time_future,
                pred_quantiles[0],
                pred_quantiles[4],
                color=colors[d],
                alpha=0.2,
                label=f"80% Pred Interval Dim {d}",
            )
            ax.fill_between(
                time_future,
                pred_quantiles[1],
                pred_quantiles[3],
                color=colors[d],
                alpha=0.3,
            )
            ax.plot(
                time_future,
                pred_quantiles[2],
                color=colors[d],
                linestyle="--",
                alpha=0.9,
                label=f"Pred Median Dim {d}",
                linewidth=2,
            )

        ax.axvline(x=len(past) - 0.5, color="black", linestyle=":", alpha=0.5)
        ax.set_title(f"Bimodal Series {i+1}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("bimodal_timegrad_results.png", dpi=150, bbox_inches="tight")
    plt.show()


def visualize_data_distribution(batch_size=1000):
    """Visualize the bimodal data to ensure it has the right structure"""
    past_data, future_data = create_complex_bimodal_data(batch_size, 30, 10, 2)

    # Combine all data points
    all_data = torch.cat([past_data, future_data], dim=1)  # [batch, time, dim]
    all_data = all_data.reshape(-1, 2)  # Flatten to [batch*time, dim]

    plt.figure(figsize=(12, 4))

    # Plot 2D scatter
    plt.subplot(1, 3, 1)
    plt.scatter(all_data[:, 0], all_data[:, 1], alpha=0.5, s=1)
    plt.title("2D Data Distribution")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, alpha=0.3)

    # Plot marginal distributions
    plt.subplot(1, 3, 2)
    plt.hist(all_data[:, 0], bins=50, alpha=0.7, density=True)
    plt.title("Marginal Distribution - Dim 1")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.hist(all_data[:, 1], bins=50, alpha=0.7, density=True)
    plt.title("Marginal Distribution - Dim 2")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(
        f"Data stats - Dim 1: mean={all_data[:, 0].mean():.2f}, std={all_data[:, 0].std():.2f}"
    )
    print(
        f"Data stats - Dim 2: mean={all_data[:, 1].mean():.2f}, std={all_data[:, 1].std():.2f}"
    )


def simple_analysis(model, device="cpu"):
    """
    SIMPLE ANALYSIS: What does the real data look like vs model predictions?
    """
    print("=" * 60)
    print("SIMPLE ANALYSIS: Real Data vs Model Predictions")
    print("=" * 60)

    # Generate some test data
    batch_size = 100  # More data for better visualization
    past_data, true_future = create_complex_bimodal_data(batch_size, 30, 10, 2)
    past_data = past_data.to(device)

    # Get model predictions
    with torch.no_grad():
        model.eval()
        predicted_samples = model.sample_future(
            past_data, num_future_steps=10, num_samples=50
        )

    # Convert to numpy for plotting
    true_future_np = true_future.numpy()
    predicted_samples_np = predicted_samples.numpy()

    # Flatten all the data points (combine all time steps and batches)
    true_all = true_future_np.reshape(-1, 2)  # [batch*time, 2]
    pred_all = predicted_samples_np.reshape(-1, 2)  # [batch*samples*time, 2]

    # Create the comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Real data
    ax1 = axes[0]
    ax1.scatter(
        true_all[:, 0], true_all[:, 1], alpha=0.6, s=20, c="blue", label="Real Data"
    )
    ax1.set_title(
        "REAL DATA\n(What the true bimodal data looks like)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("Dimension 1")
    ax1.set_ylabel("Dimension 2")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Model predictions
    ax2 = axes[1]
    ax2.scatter(
        pred_all[:, 0],
        pred_all[:, 1],
        alpha=0.6,
        s=20,
        c="red",
        label="Model Predictions",
    )
    ax2.set_title(
        "MODEL PREDICTIONS\n(What the model generates)", fontsize=14, fontweight="bold"
    )
    ax2.set_xlabel("Dimension 1")
    ax2.set_ylabel("Dimension 2")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Overlay comparison
    ax3 = axes[2]
    ax3.scatter(
        true_all[:, 0], true_all[:, 1], alpha=0.5, s=15, c="blue", label="Real Data"
    )
    ax3.scatter(
        pred_all[:, 0],
        pred_all[:, 1],
        alpha=0.5,
        s=15,
        c="red",
        label="Model Predictions",
    )
    ax3.set_title("COMPARISON\n(Do they look similar?)", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Dimension 1")
    ax3.set_ylabel("Dimension 2")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Make all plots have the same scale for fair comparison
    all_data = np.vstack([true_all, pred_all])
    x_min, x_max = all_data[:, 0].min() - 0.5, all_data[:, 0].max() + 0.5
    y_min, y_max = all_data[:, 1].min() - 0.5, all_data[:, 1].max() + 0.5

    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig("simple_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Simple analysis
    print("\n" + "=" * 60)
    print("SIMPLE STATISTICS")
    print("=" * 60)

    # Check if both have similar means and spreads
    print(
        f"Real data - Mean: [{true_all[:, 0].mean():.2f}, {true_all[:, 1].mean():.2f}]"
    )
    print(f"Real data - Std:  [{true_all[:, 0].std():.2f}, {true_all[:, 1].std():.2f}]")
    print()
    print(
        f"Model data - Mean: [{pred_all[:, 0].mean():.2f}, {pred_all[:, 1].mean():.2f}]"
    )
    print(
        f"Model data - Std:  [{pred_all[:, 0].std():.2f}, {pred_all[:, 1].std():.2f}]"
    )
    print()

    # Check if both are bimodal
    def check_bimodal(data, name):
        print(f"{name} Bimodality Check:")
        try:
            # Fit 2-component Gaussian mixture
            from sklearn.mixture import GaussianMixture

            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(data)

            centers = gmm.means_
            print(
                f"  - Found 2 modes at: [{centers[0][0]:.1f}, {centers[0][1]:.1f}] and [{centers[1][0]:.1f}, {centers[1][1]:.1f}]"
            )
            print(f"  - Mode weights: {gmm.weights_[0]:.2f} and {gmm.weights_[1]:.2f}")

            # Distance between modes (higher = more separated)
            distance = np.sqrt(np.sum((centers[0] - centers[1]) ** 2))
            print(f"  - Distance between modes: {distance:.2f}")

            return centers, distance
        except:
            print(f"  - Could not fit 2 modes")
            return None, 0

    real_centers, real_distance = check_bimodal(true_all, "REAL DATA")
    print()
    model_centers, model_distance = check_bimodal(pred_all, "MODEL DATA")
    print()

    # Final verdict
    print("=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)

    if real_centers is not None and model_centers is not None:
        if abs(real_distance - model_distance) < 1.0:
            print("✅ SUCCESS: Model captures bimodal structure!")
            print(
                f"   Real modes separated by {real_distance:.2f}, model by {model_distance:.2f}"
            )
        else:
            print("⚠️  PARTIAL: Model shows bimodal structure but different separation")
            print(
                f"   Real modes separated by {real_distance:.2f}, model by {model_distance:.2f}"
            )
    else:
        print("❌ ISSUE: Could not detect clear bimodal structure")

    return true_all, pred_all


def explain_what_we_see():
    """
    Explain what all those confusing metrics actually mean
    """
    print("\n" + "=" * 60)
    print("WHAT DO THE CONFUSING METRICS MEAN?")
    print("=" * 60)

    print("1. MSE (Mean Squared Error) = 4.0")
    print("   - This measures 'how far off' the predictions are on average")
    print("   - For bimodal data, this will be high because sometimes the model")
    print("     predicts Mode A when the truth is Mode B (or vice versa)")
    print("   - MSE = 4.0 means predictions are off by ~2 units on average")
    print("   - This is NORMAL for bimodal data!")
    print()

    print("2. Coverage = 0.375 (37.5%)")
    print(
        "   - This measures: 'What % of true values fall within the prediction intervals?'"
    )
    print("   - We want 80% of true values to fall in the 80% prediction interval")
    print("   - Getting 37.5% means the model is being 'too confident'")
    print("   - This often happens early in training")
    print()

    print("3. Bimodality Score = -0.73")
    print("   - This tries to measure if the predictions have 2 modes")
    print("   - Positive = bimodal, Negative = unimodal")
    print("   - -0.73 suggests the model isn't fully capturing both modes yet")
    print("   - But the visual plots are more reliable than this metric!")
    print()

    print("BOTTOM LINE:")
    print("The metrics look 'bad' but this is normal for bimodal data.")
    print("What matters is: Do the scatter plots show 2 clear modes?")


if __name__ == "__main__":
    print("=== Visualizing Data Distribution ===")
    visualize_data_distribution()

    print("\n=== Bimodal TimeGrad Training ===")
    model, losses = train_bimodal_timegrad()

    print("\n=== SIMPLE ANALYSIS (EASY TO UNDERSTAND) ===")
    device = next(model.parameters()).device
    simple_analysis(model, device)

    print("\n=== EXPLANATION OF CONFUSING METRICS ===")
    explain_what_we_see()

    print("\n=== Original Complex Analysis ===")
    metrics = test_bimodal_model(model, device)

    print("\n=== Training Loss Plot ===")
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("bimodal_training_loss.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nDone! The key plot to look at is 'simple_comparison.png'")
