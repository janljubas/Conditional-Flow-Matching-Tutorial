"""
Noise source abstraction for CFM training and sampling.

Supports:
  - "gaussian": standard N(0, I) noise (default)
  - "quantum":  pre-computed boson sampling data from quantum_data/ folder,
                affine-rescaled to zero mean / unit variance per dimension,
                then projected to the target image shape via a fixed random matrix.

The projection is necessary because quantum samples live in R^8 while images
live in R^(C*H*W).  A fixed random matrix preserves the correlation structure
of the quantum distribution while expanding it to the required dimensionality.
The projection seed is fixed so the mapping is deterministic across runs.
"""

import torch
from pathlib import Path


class GaussianNoiseSampler:
    """Standard Gaussian noise -- just wraps torch.randn_like."""

    def __init__(self, **kwargs):
        pass

    def sample_like(self, x):
        """Return Gaussian noise with the same shape/device/dtype as x."""
        return torch.randn_like(x)

    def sample(self, shape, device):
        """Return Gaussian noise of the given shape on the given device."""
        return torch.randn(size=shape, device=device)


class QuantumNoiseSampler:
    """
    Loads pre-computed quantum boson sampling vectors, rescales them,
    and projects to arbitrary target shapes on demand.

    Steps:
      1. Load all .pt files from quantum_data_path, concatenate into [N, D] pool
      2. Affine-rescale per dimension to zero mean, unit variance
      3. On each call, draw B random rows and project R^D -> R^(C*H*W) via
         a fixed random matrix, then reshape to [B, C, H, W]
    """

    def __init__(self, quantum_data_path, projection_seed=42, **kwargs):
        pool = self._load_pool(quantum_data_path)
        self.pool = self._rescale(pool)
        self.qdim = self.pool.shape[1]
        self._proj_cache = {}
        self._projection_seed = projection_seed

    @staticmethod
    def _load_pool(data_path):
        """
        Loads the quantum data from the given path and returns a tensor of shape [N, D].

        Note that each .pt file contains a tensor of ~50 readouts(measurements) of the 8-mode quantum state,
        which represent the measurement statistics of that particular 8-mode quantum state.
        """
        # simple file loading, with expansion of the path to the home directory
        p = Path(data_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Quantum data path not found: {p}")
        files = sorted(p.glob("*.pt"))  # we sort the files to ensure reproducibility
        if not files:
            raise FileNotFoundError(f"No .pt files found in {p}")

        # now we load the .pt files into a tensor, we use cpu to avoid memory issues
        chunks = [torch.load(f, map_location="cpu", weights_only=True).float() for f in files]
        # `chunks` variable contains a list of tensors, on average each tensor is of shape [M=50, D=8]

        pool = torch.cat(chunks, dim=0)  # pooling (concatenation) of the chunks into a single tensor
        # this is fine because we assume that all chunks are i.i.d. measurement statistics of the same 8-mode quantum state (same interferometer configuration)
        # i.e. this is the same as running the experiment for longer, or just having all the readouts in one go
        print(f"[QuantumNoiseSampler] Loaded {pool.shape[0]} samples of dim {pool.shape[1]} from {p}")
        
        return pool  # this is a tensor of shape [~N*M, D] of type float32

    @staticmethod
    def _rescale(pool):
        """
        Affine rescale to zero mean, unit variance per dimension.
        Concretely, for each dimension (mode of the quantum state) independently, we compute the mean and standard deviation of the pool,
        and then rescale the pool to have zero mean and unit variance.

        This is completely fine, as basically all relevant information is preserved in the correlation structure of the data.

        Returns a tensor of the same shape as the input tensor `pool`.
        """
        mu = pool.mean(dim=0, keepdim=True)     # prepares the mean in the shape ideal for broadcasting subtraction (i.e. [1, D], because of the `keepdim=True` flag)
        std = pool.std(dim=0, keepdim=True).clamp(min=1e-8)     # we clamp the standard deviation to avoid division by zero and possible numerical instability
        return (pool - mu) / std  # still of shape [~N*M, D] of type float32

    def _get_projection(self, target_flat_dim, device, dtype):
        """
        The purpose of this function is to project the quantum data from the D-dimensional space into the target image space.
        Concretely, we draw a random matrix W from the normal distribution N(0, I) and normalize it.

        We then cache the projection matrix for future use to avoid re-generating it every time.

        Returns a fixed random projection matrix [qdim, target_flat_dim].
        """
        key = (target_flat_dim, device, dtype)
        # defined by: (a) target flattened dimension (i.e. the number of pixels in the target image), 
                    # (b) device
                    # (c) data type of the projected noise
        if key not in self._proj_cache:
            gen = torch.Generator().manual_seed(self._projection_seed)
            W = torch.randn(self.qdim, target_flat_dim, generator=gen, dtype=dtype)
            W = W / (self.qdim ** 0.5)  # we divide by the square root of the number of dimensions to ensure that the projected noise has the same variance as the original noise
            self._proj_cache[key] = W.to(device)  # we cache the projection matrix for future use
        return self._proj_cache[key]

    def _draw_and_project(self, batch_size, shape_tail, device, dtype):
        """
        Draws `batch_size` number of quantum samples and project them to the target image space of shape `shape_tail`.

        Returns a tensor of shape [batch_size, *shape_tail].
        """
        idx = torch.randint(0, self.pool.shape[0], (batch_size,))  # draws `batch_size` number of random indices from the [~N*M] pool
        z = self.pool[idx].to(device=device, dtype=dtype)
        flat_dim = 1
        for s in shape_tail:
            flat_dim *= s  # computes the flattened dimension (total number of pixels) in the target image (e.g. 28*28=784 for MNIST, 32*32*3=3072 for CIFAR10)
        W = self._get_projection(flat_dim, device, dtype)  # [D, flat_dim] (so e.g. [8, 784] for MNIST)

        projected = z @ W   # [B, D] @ [D, flat_dim] = [B, flat_dim]
        
        return projected.view(batch_size, *shape_tail)  # .view() reshapes it back from (784,) into (1, 28, 28) for example

    def sample_like(self, x):
        """
        Returns quantum noise with the same shape/device/dtype as the input tensor `x`.
        """
        return self._draw_and_project(x.shape[0], x.shape[1:], x.device, x.dtype)

    def sample(self, shape, device):
        """
        Returns quantum noise of the given shape on the given device (i.e. [B, C, H, W] for example).
        """
        return self._draw_and_project(shape[0], shape[1:], device, torch.float32)


def build_noise_sampler(settings):
    """
    Factory: build a noise sampler from settings dict.
    """
    source = settings.get("noise_source", "gaussian")
    if source == "gaussian":
        return GaussianNoiseSampler()
    elif source == "quantum":
        qpath = settings.get("quantum_data_path")
        if not qpath:
            raise ValueError("noise_source='quantum' requires 'quantum_data_path' in config.")
        return QuantumNoiseSampler(quantum_data_path=qpath)
    else:
        raise ValueError(f"Unknown noise_source '{source}'. Choose 'gaussian' or 'quantum'.")
