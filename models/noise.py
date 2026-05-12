"""
Noise source abstraction for CFM training and sampling.

Available samplers
--------------------------------------------
  (Phase 1.A)
  GaussianNoiseSampler          - standard N(0, I)
  QuantumSamplerLinear          - rank-8 single linear projection (baseline)

  (Phase 1.B)
  QuantumSamplerMultiSample     - Strategy 1: k samples tiled spatially
  QuantumSamplerMultiSamplePE   - Strategy 2: Strategy 1 + sinusoidal PE
  QuantumSamplerHybrid          - Strategy 3: alpha * q + sqrt(1 - alpha^2) * eps
  QuantumSamplerFourier         - Strategy 4: sin/cos basis expansion, then projection

  (Phase 1.D)
  QuantumSamplerMLP             - Strategy 5: trainable MLP from k stacked
                                  R^D quantum vectors to flat image. The ONLY
                                  sampler whose parameters are learned end-to-end
                                  with the velocity network (it is an nn.Module
                                  and the train scripts add its params to the
                                  optimizer).

Quantum-pool samplers (all except Gaussian) share:
  - Pool loading from .pt files in `quantum_data_path`
  - Per-dimension affine rescaling to zero mean / unit variance
  - A fixed `projection_seed` for reproducible projection matrices (Strategies 1-4)

Why these strategies?
The current quantum pool is 8-dim, MNIST is 784-dim. A single linear
projection R^8 -> R^784 has rank 8: only an 8-dimensional affine subspace
of pixel space is reachable, the other 776 dims are deterministic given
which 8-D point was drawn. With a finite pool (~64,720 samples) every
MNIST image effectively gets a near-deterministic low-dim "tag", and the
model overfits the (proj, x_1) coupling -- low train loss, terrible FID.
Strategies 1-4 try to overcome this failure mode by either raising the
effective rank of the projected noise (Multi-sample, Multi-sample with PE,
Fourier) or guaranteeing full rank by construction (Hybrid). Strategy 5
(MLP) takes a different tack: replace the fixed random projection with a
trainable MLP, so the projection from quantum space to image space is
data-adapted instead of random.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Gaussian baseline
# ---------------------------------------------------------------------------

class GaussianNoiseSampler:
    """
    Standard Gaussian noise; a thin wrapper over torch.randn.
    """

    def __init__(self, **kwargs):
        pass

    def sample_like(self, x):
        """Return Gaussian noise with the same shape/device/dtype as x."""
        return torch.randn_like(x)

    def sample(self, shape, device):
        """Return Gaussian noise of the given shape on the given device."""
        return torch.randn(size=shape, device=device)


# ---------------------------------------------------------------------------
# Shared quantum-pool base (loading, rescaling, projection caching)
# ---------------------------------------------------------------------------

class _QuantumPoolBase:
    """
    Shared utilities for quantum samplers:
      * loads all .pt files in `quantum_data_path` into a CPU pool [N, D]
      * affine-rescales per dimension to zero mean / unit variance
      * caches fixed random projection matrices (deterministic via seed)
      * provides batched `_draw()` of B*k vectors from the pool

    Subclasses must implement `_draw_and_shape(batch_size, shape_tail, ...)`
    which returns the final noise tensor of shape [B, *shape_tail].
    """

    def __init__(self, quantum_data_path, projection_seed=42, **kwargs):
        pool = self._load_pool(quantum_data_path)
        self.pool = self._rescale(pool)             # [N, D] float32, zero mean/unit var per dim
        self.qdim = self.pool.shape[1]              # D, typically 8
        self._projection_seed = projection_seed
        self._proj_cache = {}                       # cache for fixed random projections
        self._pe_cache = {}                         # cache for positional encodings
        self._freq_cache = {}                       # cache for Fourier frequencies

    # -- Pool I/O ----------------------------------------------------------

    @staticmethod
    def _load_pool(data_path):
        """
        Read every .pt file in `data_path` and concatenate into [N, D].

        Each .pt file holds ~50 i.i.d. measurements of the same 8-mode
        quantum state, so concatenation = pretending we ran the experiment
        for longer. Files are sorted for reproducibility.
        """
        p = Path(data_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Quantum data path not found: {p}")
        files = sorted(p.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No .pt files found in {p}")
        chunks = [torch.load(f, map_location="cpu", weights_only=True).float() for f in files]
        pool = torch.cat(chunks, dim=0)
        print(f"[QuantumPool] Loaded {pool.shape[0]} samples of dim {pool.shape[1]} from {p}")
        return pool

    @staticmethod
    def _rescale(pool):
        """
        Affine rescale each dim independently to zero mean / unit variance.

        Preserves the *correlation structure* between dims (which is what
        carries the quantum information) while removing per-mode scale.
        Clamping the std avoids division-by-zero on (degenerate) constant dims.
        """
        mu = pool.mean(dim=0, keepdim=True)
        std = pool.std(dim=0, keepdim=True).clamp(min=1e-8)
        return (pool - mu) / std

    # -- Random helpers ----------------------------------------------------

    def _draw(self, n, device, dtype):
        """
        Draw n random rows from the pool, ship to (device, dtype).
        """
        idx = torch.randint(0, self.pool.shape[0], (n,))
        return self.pool[idx].to(device=device, dtype=dtype)         # [n, D]

    def _get_projection(self, in_dim, out_dim, device, dtype, name="proj", unit_columns=False):
        """
        Cached fixed-random projection matrix of shape [in_dim, out_dim].

        Variance normalization keeps output coords ~unit-variance when input
        coords are unit-variance Gaussian, matching the marginal scale of an
        N(0, I) baseline:

          unit_columns=False (default, used by linear/hybrid/fourier):
            scale entries by 1/sqrt(in_dim) -> E[var(output)] = 1, but a
            single realization has var that fluctuates by O(1/sqrt(in_dim)).
            For the linear sampler (out_dim=784, single shared row across
            pixels) the per-pixel fluctuations average out to std ~1.0.

          unit_columns=True (used by multi_sample/multi_sample_pe):
            scale each column to unit Euclidean norm -> output var is
            exactly 1 per coord. This matters when the SAME small projection
            (e.g. D=8 -> region_flat=4) is reused across many regions: the
            column-norm bias is fixed across all regions, so without this
            normalization the marginal std can drift to ~0.85 for fine
            tilings (k=196 etc.).

        `name` namespaces the cache so different strategies (linear, multi,
        fourier) generate independent projection matrices instead of sharing.
        """
        key = (name, in_dim, out_dim, device, dtype, unit_columns)
        if key not in self._proj_cache:
            seed = self._projection_seed + (hash(name) % (2**31))
            gen = torch.Generator().manual_seed(seed)
            W = torch.randn(in_dim, out_dim, generator=gen, dtype=dtype)
            if unit_columns:
                W = W / W.norm(dim=0, keepdim=True).clamp(min=1e-8)
            else:
                W = W / (in_dim ** 0.5)
            self._proj_cache[key] = W.to(device)
        return self._proj_cache[key]

    # -- Public sampling API (shared) --------------------------------------

    def sample_like(self, x):
        """
        Return quantum noise with the same shape/device/dtype as x.
        """
        return self._draw_and_shape(x.shape[0], x.shape[1:], x.device, x.dtype)

    def sample(self, shape, device):
        """
        Return quantum noise of the given shape on the given device.
        """
        return self._draw_and_shape(shape[0], shape[1:], device, torch.float32)

    def _draw_and_shape(self, batch_size, shape_tail, device, dtype):
        """
        Subclasses must implement this method to return the final noise tensor of shape [B, *shape_tail].
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Baseline: rank-8 single linear projection
# ---------------------------------------------------------------------------

class QuantumSamplerLinear(_QuantumPoolBase):
    """
    Original behavior from Phase 0 (kept for backwards compatibility and use as the inner
    sampler of QuantumSamplerHybrid).

    One quantum draw per image, projected by a fixed `R^D -> R^N` matrix.
    Effective rank = D = 8 << 784. This is the failure mode the other
    strategies from Phase 1 are designed to fix.
    """

    def _draw_and_shape(self, batch_size, shape_tail, device, dtype):
        flat_dim = int(torch.tensor(shape_tail).prod().item())
        z = self._draw(batch_size, device, dtype)                         # [B, D]
        W = self._get_projection(self.qdim, flat_dim, device, dtype, name="linear")
        return (z @ W).view(batch_size, *shape_tail)                       # [B, C, H, W]


# ---------------------------------------------------------------------------
# Strategy 1 -- Multi-sample composition (row or patch tiling)
# ---------------------------------------------------------------------------

def _tiling_layout(tiling, shape_tail, patch_size):
    """
    Compute (k, region_flat, to_image) for a given tiling scheme.
    This is a helper function for the tiling schemes in Strategy 1.

    Returns:
      k           : number of tiles (= number of quantum draws per image)
      region_flat : flattened size of one tile (input dim of the projection)
      to_image    : function mapping [B, k, region_flat] -> [B, C, H, W]
    """
    C, H, W = shape_tail

    if tiling == "row":
        # Each row of the image gets one quantum draw; region = (C, 1, W).
        k = H
        region_flat = C * W

        def to_image(t):
            # t is [B, H, C*W]; un-flatten the last dim to (C, W) and permute
            # to put channels first so the result is [B, C, H, W].
            return t.view(t.shape[0], H, C, W).permute(0, 2, 1, 3).contiguous()

        return k, region_flat, to_image

    if tiling == "patch":
        # Normalize patch_size to (ph, pw). Allow rectangular patches so we
        # can hit k values that don't admit a square tiling -- e.g. k=98 needs
        # 2x4 patches on 28x28 (98 = 14*7), giving full theoretical rank.
        if isinstance(patch_size, (list, tuple)):
            if len(patch_size) != 2:
                raise ValueError(
                    f"patch_size as list/tuple must have length 2, got {patch_size}."
                )
            ph, pw = int(patch_size[0]), int(patch_size[1])
        else:
            ph = pw = int(patch_size)

        if H % ph or W % pw:
            raise ValueError(
                f"patch_size=({ph},{pw}) must divide H={H} and W={W}."
            )
        nh, nw = H // ph, W // pw
        k = nh * nw
        region_flat = C * ph * pw

        def to_image(t):
            B = t.shape[0]
            # [B, k, C*ph*pw] -> [B, nh, nw, C, ph, pw]
            #                 -> [B, C, nh, ph, nw, pw]  (interleave grid + patch dims)
            #                 -> [B, C, H, W]
            return (
                t.view(B, nh, nw, C, ph, pw)
                 .permute(0, 3, 1, 4, 2, 5)
                 .reshape(B, C, H, W)
            )

        return k, region_flat, to_image

    raise ValueError(f"Unknown tiling '{tiling}'. Choose 'row' or 'patch'.")


class QuantumSamplerMultiSample(_QuantumPoolBase):
    """
    Strategy 1 -- Multi-sample composition.

    For each image we draw k *independent* quantum samples and project each
    to the region's flattened size via a single shared fixed
    R^D -> R^region_flat projection. The resulting tiles are then arranged
    spatially (rows or patches).

    Tilings:
      'row'   : k = H (each row gets its own draw, region size C*W)
      'patch' : k = (H/ph)*(W/pw) -- patch_size may be int (square) or [ph,pw]
                (rectangular). Use rectangular to reach k values without a
                square divisor (e.g. patch_size=[2,4] on 28x28 -> k=98).

    Effective rank rises from D (=8) to min(D*k, C*H*W). Examples on 28x28:
      - row tiling           k=28,  D*k = 224  (partial rank)
      - patch [4,4]          k=49,  D*k = 392  (partial rank)
      - patch [2,4]          k=98,  D*k = 784  (full rank by construction)
      - patch [2,2]          k=196, region=4, min(D,4)*k = 784 (full rank)

    Note: we use a single shared projection (not k different ones) so that
    every region is statistically identical -- this means the rank gain is
    real (k *truly independent* draws from the same channel) and the model
    cannot trivially memorize k separate per-region decoders.
    """

    def __init__(self, quantum_data_path, tiling="row", patch_size=7, **kwargs):
        super().__init__(quantum_data_path=quantum_data_path, **kwargs)
        self.tiling = tiling
        self.patch_size = patch_size

    def _draw_and_shape(self, batch_size, shape_tail, device, dtype):
        k, region_flat, to_image = _tiling_layout(self.tiling, shape_tail, self.patch_size)

        # Draw B * k quantum vectors in one go, then reshape to [B, k, D].
        z = self._draw(batch_size * k, device, dtype).view(batch_size, k, self.qdim)

        # Hook for subclasses (e.g. PE in Strategy 2). Identity by default.
        z = self._pretransform(z, k, device, dtype)                       # [B, k, D]

        # Single shared projection: every region uses the same R^D -> R^region_flat.
        # `unit_columns=True` -> each output coord has variance exactly 1
        # (avoids the column-norm bias becoming a fixed scale offset across
        # all regions when D=8 and region_flat is small, e.g. k=196 -> 4).
        W = self._get_projection(
            self.qdim, region_flat, device, dtype, name="multi", unit_columns=True
        )
        proj = z @ W                                                      # [B, k, region_flat]
        return to_image(proj)                                             # [B, C, H, W]

    def _pretransform(self, z, k, device, dtype):
        """
        Hook: transform z BEFORE projection. Identity in Strategy 1.
        """
        return z


# ---------------------------------------------------------------------------
# Strategy 2 -- Multi-sample + sinusoidal positional encoding
# ---------------------------------------------------------------------------

def _sinusoidal_pe(num_positions, dim, device, dtype):
    """
    Standard transformer-style sinusoidal positional encoding.
    This is a helper function for the sinusoidal positional encoding in Strategy 2.
    The positional encoding is used to add position information to the quantum noise.

    Returns [num_positions, dim] with alternating sin/cos of geometrically
    increasing wavelengths (1 .. 10000^((dim-1)/dim)).
    """
    pe = torch.zeros(num_positions, dim, device=device, dtype=dtype)
    pos = torch.arange(num_positions, device=device, dtype=dtype).unsqueeze(1)
    half = max(dim // 2, 1)
    div = torch.exp(
        torch.arange(half, device=device, dtype=dtype) * (-math.log(10000.0) / half)
    )
    args = pos * div                                                       # [num_positions, half]
    pe[:, 0:2 * half:2] = torch.sin(args)
    pe[:, 1:2 * half:2] = torch.cos(args)
    return pe                                                              # last col stays 0 if dim is odd


class QuantumSamplerMultiSamplePE(QuantumSamplerMultiSample):
    """
    Strategy 2 -- Multi-sample with positional encoding.

    Same as Strategy 1, but BEFORE projection we add a sinusoidal positional
    encoding to each of the k quantum tokens:
        z'_i = (z_i + pe_scale * PE(i)) / sqrt(1 + pe_scale^2 * var_pe)
    The PE is computed in the qdim-D space (D=8), so even if two regions
    happen to draw the same quantum vector (possible with a finite pool of
    ~64,720 samples) they still produce *different* projected pixels.

    This breaks the spatial-permutation symmetry of Strategy 1: the model
    can attribute different velocity fields to different regions based on
    position alone, which is exactly the kind of structure MNIST has
    (digits live in the centre, edges are background, etc.).

    Variance handling:
      Sinusoidal PE has empirical variance ~0.5 per entry. Naively adding it
      to unit-variance z would inflate output std to sqrt(1 + pe_scale^2 *
      0.5). We renormalize by sqrt(1 + pe_scale^2 * var_pe) (computed from the
      cached PE) so the input to the projection has approximately unit variance
      regardless of pe_scale. This keeps every quantum sampler producing
      ~N(0, 1)-marginal noise, so all strategies are directly comparable to
      the Gaussian baseline.
    """

    def __init__(self, quantum_data_path, pe_scale=1.0, **kwargs):
        super().__init__(quantum_data_path=quantum_data_path, **kwargs)
        self.pe_scale = float(pe_scale)

    def _pretransform(self, z, k, device, dtype):
        key = (k, self.qdim, device, dtype)
        if key not in self._pe_cache:
            pe = _sinusoidal_pe(k, self.qdim, device, dtype)              # [k, D]
            # Center per-channel and renormalize so the additive perturbation
            # has approximately zero mean / unit variance per dim. This keeps
            # `z + scale*PE` as ~N(0, 1)-marginal regardless of k or D, so
            # this strategy is directly comparable to the Gaussian baseline.
            pe = pe - pe.mean(dim=0, keepdim=True)                        # zero-mean per dim
            var_pe = float(pe.var(unbiased=False).item())                 # empirical, ≈ 0.5
            denom = math.sqrt(max(1.0 + self.pe_scale ** 2 * var_pe, 1e-12))
            self._pe_cache[key] = (pe, denom)
        pe, denom = self._pe_cache[key]
        return (z + self.pe_scale * pe.unsqueeze(0)) / denom              # [B, k, D]


# ---------------------------------------------------------------------------
# Strategy 3 -- Gaussian + quantum hybrid
# ---------------------------------------------------------------------------

class QuantumSamplerHybrid:
    """
    Strategy 3 -- Gaussian + quantum hybrid.

        x_0 = alpha * q + sqrt(1 - alpha^2) * eps,
        eps ~ N(0, I),  q = QuantumSamplerLinear(...).sample(...)

    This guarantees full-rank source noise (the Gaussian component spans the
    whole space) while injecting quantum correlations as a lower-dim
    perturbation.

    Variance: with both q and eps approximately unit-variance and (mostly)
    uncorrelated -- a reasonable approximation since q lives on an 8-D
    subspace and eps fills the other ~776 dims -- the marginal variance of
    x_0 stays near 1 by Pythagoras: alpha^2 * 1 + (1 - alpha^2) * 1 = 1.

    Limits:
      alpha = 0  -> pure Gaussian (recovers the baseline control)
      alpha = 1  -> pure rank-8 quantum (recovers QuantumSamplerLinear)
      alpha = 0.5 -> 25% quantum variance + 75% Gaussian variance, mixed.
    """

    def __init__(self, quantum_data_path, alpha=0.5, projection_seed=42, **kwargs):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
        self.alpha = float(alpha)
        self.beta = math.sqrt(max(1.0 - self.alpha * self.alpha, 0.0))
        self._inner = QuantumSamplerLinear(
            quantum_data_path=quantum_data_path,
            projection_seed=projection_seed,
        )

    def sample_like(self, x):
        q = self._inner.sample_like(x)
        eps = torch.randn_like(x)
        return self.alpha * q + self.beta * eps

    def sample(self, shape, device):
        q = self._inner.sample(shape, device)
        eps = torch.randn(size=shape, device=device)
        return self.alpha * q + self.beta * eps


# ---------------------------------------------------------------------------
# Strategy 4 -- Fourier feature mapping
# ---------------------------------------------------------------------------

class QuantumSamplerFourier(_QuantumPoolBase):
    """
    Strategy 4 -- Fourier feature expansion.

    For each quantum dim z_j (j=1..D=8) we build a sin/cos basis at F
    different frequencies:

        FF(z_j) = [sin(2*pi*f_1 z_j), cos(2*pi*f_1 z_j),
                   sin(2*pi*f_2 z_j), cos(2*pi*f_2 z_j),
                   ...,
                   sin(2*pi*f_F z_j), cos(2*pi*f_F z_j)]

    Concatenating across j gives 2*D*F features per quantum vector. The
    expanded vector is then linearly projected (fixed random) to the target
    image dim N. For MNIST with D=8, F=49: 2*D*F = 784 = N exactly, so the
    projection becomes (effectively) a square random rotation.

    Why this raises rank:
      The original linear projection produces a rank-8 image. Fourier
      features lift each scalar nonlinearly into a 2F-dim sinusoidal
      manifold; concatenated across D dims this gives a 2DF-dim feature
      whose effective rank can reach min(2DF, N) = N for the typical setup.

    Frequencies:
      'log'    -> log-spaced in [1, max_freq] (multi-scale, recommended).
      'random' -> Gaussian-random with std max_freq (Tancik-style).
      Frequencies are fixed at init via `projection_seed`, so identical
      across runs that use the same seed.

    Caveat (rank on discrete inputs):
      Quantum boson samples take only ~3 distinct values per dim (photon
      counts in {0, 1, 2} after rescaling). The Fourier expansion of a
      scalar with V distinct values has linear span at most V, so the
      total feature rank is bounded by V * D, *not* 2 * D * F. Empirically
      this gives ≤ 24 effective rank for D=8, V=3, regardless of F.
      To exceed that ceiling you'd need to inject continuous noise into z
      (e.g. compose Fourier with hybrid -- not done here for clarity).

    Variance handling:
      The raw sin/cos features have empirical variance ~0.5 per element
      (E[sin^2(arg)] = 0.5 for arg distributed broadly). We rescale them
      by sqrt(2) before projection so feats have unit-variance entries,
      which keeps the projected output ~N(0, 1)-marginal -- consistent with
      the Gaussian baseline and with the other quantum strategies.
    """

    def __init__(self, quantum_data_path,
                 n_frequencies=49,
                 max_freq=32.0,
                 freq_init="log",
                 **kwargs):
        super().__init__(quantum_data_path=quantum_data_path, **kwargs)
        self.F = int(n_frequencies)
        self.max_freq = float(max_freq)
        self.freq_init = str(freq_init)

    def _get_freqs(self, device, dtype):
        key = ("fourier_freqs", self.F, self.max_freq, self.freq_init, device, dtype)
        if key not in self._freq_cache:
            if self.freq_init == "log":
                f = torch.logspace(
                    0.0, math.log10(max(self.max_freq, 1.0)),
                    self.F, device=device, dtype=dtype,
                )
            elif self.freq_init == "random":
                gen = torch.Generator().manual_seed(self._projection_seed + 7)
                f = (torch.randn(self.F, generator=gen, dtype=dtype) * self.max_freq).to(device)
            else:
                raise ValueError(f"Unknown freq_init '{self.freq_init}'. Choose 'log' or 'random'.")
            self._freq_cache[key] = f
        return self._freq_cache[key]                                       # [F]

    def _draw_and_shape(self, batch_size, shape_tail, device, dtype):
        flat_dim = int(torch.tensor(shape_tail).prod().item())

        z = self._draw(batch_size, device, dtype)                          # [B, D]
        f = self._get_freqs(device, dtype)                                 # [F]

        # args[b, j, i] = 2 * pi * z[b, j] * f[i] -> [B, D, F]
        args = (2.0 * math.pi) * z.unsqueeze(-1) * f.view(1, 1, -1)
        # Multiply by sqrt(2) so each entry has unit variance (E[2*sin^2] = 1).
        feats = math.sqrt(2.0) * torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, D, 2F]
        feats = feats.flatten(1)                                           # [B, 2*D*F]

        # Linear projection from 2*D*F to flat image dim. When 2*D*F == flat_dim
        # this is an (almost) square random rotation; otherwise it acts as a
        # fixed-random low-rank embedding.
        W = self._get_projection(2 * self.qdim * self.F, flat_dim, device, dtype, name="fourier")
        return (feats @ W).view(batch_size, *shape_tail)


# ---------------------------------------------------------------------------
# Strategy 5 -- Trainable MLP from k stacked quantum vectors to image
# ---------------------------------------------------------------------------

class QuantumSamplerMLP(_QuantumPoolBase, nn.Module):
    """
    Strategy 5 (Phase 1.D) -- Trainable MLP from k stacked quantum vectors to image.

    For each image we draw k independent quantum samples (each in R^D=R^8),
    concatenate them into a single [k*D]-dim vector, and pass that vector
    through a trainable multi-layer MLP with non-linear activations to
    produce a flat [H*W*C] image vector. The result is reshaped to
    [B, C, H, W] and used as x_0 in the CFM training loop.

    Unlike Strategies 1-4, this sampler is an nn.Module: its parameters
    are optimized end-to-end with the velocity network. The training
    scripts detect this via isinstance(sampler, nn.Module) and combine
    the parameter groups in the optimizer. Inference scripts likewise
    load the sampler's state_dict from the checkpoint.

    Choice of k (input_dim = k * D = k * 8):
      k=96  -> input_dim 768, slightly under image dim 784 (16-dim deficit)
      k=98  -> input_dim 784, exactly matches image dim
      k=128 -> input_dim 1024, ~30% oversubscribed (full theoretical rank)

    Why MLP, not learned linear?
      A learned linear (k*D)->784 has rank min(k*D, 784) -- equivalent to
      multi_sample for k*D >= 784. The MLP's nonlinearities (GELU) let the
      network compose data-adapted feature maps that can in principle
      exceed the linear rank of the input. Empirically this is the first
      strategy where the projection from quantum space to image space is
      data-aware rather than fixed random.

    Architecture (defaults):
      input -> Linear(k*D, hidden_dim) -> GELU
            -> Linear(hidden_dim, hidden_dim) -> GELU      (n_hidden_layers - 1 times)
            -> Linear(hidden_dim, output_dim)
            -> multiply by learnable scalar `output_scale`
      For k=96, hidden=512, n_hidden_layers=2: ~1.06M trainable params.

    Variance handling:
      Inputs are unit-variance per dim (rescaled by _QuantumPoolBase).
      Hidden layers use He init (good for GELU). Final layer uses Xavier
      uniform (no activation after) so output variance ~1.0 at init. A
      learnable scalar `output_scale` (init 1.0) lets the model adjust
      the marginal variance during training without forcing it to.

    Notes:
      - The k draws are sampled fresh per call (per training step), so
        each batch sees different quantum noise even though the MLP is
        deterministic.
      - The pool stays on CPU (managed by _QuantumPoolBase); only the
        MLP parameters move to GPU via .to(device).
      - This sampler does NOT use _get_projection (no fixed random matrix
        is needed; the MLP IS the projection).
    """

    _ALLOWED_ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}

    def __init__(
        self,
        quantum_data_path,
        output_dim,
        k=96,
        hidden_dim=512,
        n_hidden_layers=2,
        activation="gelu",
        projection_seed=42,
        **kwargs,
    ):
        nn.Module.__init__(self)
        _QuantumPoolBase.__init__(
            self,
            quantum_data_path=quantum_data_path,
            projection_seed=projection_seed,
            **kwargs,
        )

        self.k = int(k)
        self.input_dim = self.k * self.qdim
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_hidden_layers = int(n_hidden_layers)

        act_key = str(activation).lower()
        if act_key not in self._ALLOWED_ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Choose from "
                f"{sorted(self._ALLOWED_ACTIVATIONS)}."
            )
        act_cls = self._ALLOWED_ACTIVATIONS[act_key]

        # Build [in -> hidden -> ... -> hidden -> out] with activations between
        # every pair *except* after the last layer (output should be linear so
        # marginal variance is controllable).
        dims = (
            [self.input_dim]
            + [self.hidden_dim] * self.n_hidden_layers
            + [self.output_dim]
        )
        layers = []
        n_pairs = len(dims) - 1
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            lin = nn.Linear(d_in, d_out)
            self._init_linear(lin, is_final=(i == n_pairs - 1))
            layers.append(lin)
            if i < n_pairs - 1:
                layers.append(act_cls())
        self.mlp = nn.Sequential(*layers)

        # Learnable output scale (lets the model tune marginal std).
        # We calibrate this from actual quantum-pool samples at init so the
        # initial output has ~unit variance, matching the Gaussian baseline
        # and the other quantum samplers (which all produce ~N(0,1)-marginal
        # noise). Without calibration, He init + GELU + Xavier final layer
        # produces ~0.80 std for k=96 / ~0.84 for k=128 (empirical), which
        # would silently make the source noise smaller than the baselines.
        self.output_scale = nn.Parameter(torch.ones(1))
        self._calibrate_output_scale(n_samples=1024)

    @torch.no_grad()
    def _calibrate_output_scale(self, n_samples=1024):
        """Set output_scale = 1/std(MLP(z)) so initial output has ~unit var.

        Uses a deterministic RNG seeded off projection_seed so calibration is
        reproducible across runs that share the seed. Drawn from the actual
        rescaled quantum pool (not synthetic Gaussian) so calibration accounts
        for the pool's discrete distribution (photon counts in {0, 1, 2}).
        """
        if self.pool.shape[0] < self.k:
            return  # safety: can't calibrate
        g = torch.Generator().manual_seed(self._projection_seed + 17)
        idx = torch.randint(0, self.pool.shape[0], (n_samples * self.k,), generator=g)
        z = self.pool[idx].view(n_samples, self.input_dim).to(dtype=next(self.mlp.parameters()).dtype)
        out = self.mlp(z)
        measured_std = float(out.std().item())
        if measured_std > 1e-8:
            self.output_scale.data.fill_(1.0 / measured_std)

    @staticmethod
    def _init_linear(lin, is_final):
        if is_final:
            # Xavier (uniform) for final layer: maps unit-var input -> ~unit-var output
            nn.init.xavier_uniform_(lin.weight)
        else:
            # He / Kaiming for hidden layers (good for ReLU/GELU)
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
        nn.init.zeros_(lin.bias)

    def _draw_and_shape(self, batch_size, shape_tail, device, dtype):
        # Sanity: shape_tail must match the MLP's output_dim chosen at construction.
        flat_dim = int(torch.tensor(shape_tail).prod().item())
        if flat_dim != self.output_dim:
            raise ValueError(
                f"QuantumSamplerMLP was built with output_dim={self.output_dim} "
                f"but received shape_tail={tuple(shape_tail)} (flat={flat_dim}). "
                "img_size in the config must match the output_dim used at construction."
            )

        # Draw B*k quantum vectors and concat per image.
        z = self._draw(batch_size * self.k, device, dtype)        # [B*k, D]
        z = z.view(batch_size, self.input_dim)                    # [B, k*D]

        # MLP -> flat image -> reshape.
        out = self.mlp(z) * self.output_scale                     # [B, output_dim]
        return out.view(batch_size, *shape_tail)                  # [B, C, H, W]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_QUANTUM_REGISTRY = {
    "linear":          QuantumSamplerLinear,
    "multi_sample":    QuantumSamplerMultiSample,
    "multi_sample_pe": QuantumSamplerMultiSamplePE,
    "hybrid":          QuantumSamplerHybrid,
    "fourier":         QuantumSamplerFourier,
    "mlp":             QuantumSamplerMLP,
}


def build_noise_sampler(settings):
    """
    Factory: instantiate a noise sampler from a settings dictionary.

    Recognised keys (all read with `settings.get(...)`):
        noise_source                : "gaussian" | "quantum"
        quantum_data_path           : path to .pt files (required if quantum)
        quantum_projection          : strategy name (default "linear" for
                                      backwards compat with old configs)
        quantum_projection_seed     : int, seed for fixed projections (default 42)
        quantum_tiling              : "row" | "patch" (multi_sample[_pe], default "row")
        quantum_patch_size          : int (multi_sample[_pe] patch tiling, default 7)
        quantum_pe_scale            : float (multi_sample_pe only, default 1.0)
        quantum_alpha               : float in [0, 1] (hybrid only, default 0.5)
        quantum_fourier_freqs       : int F (fourier only, default 49)
        quantum_fourier_max_freq    : float (fourier only, default 32)
        quantum_fourier_init        : "log" | "random" (fourier only, default "log")
        quantum_mlp_k               : int (mlp only, default 96; input_dim = k*D)
        quantum_mlp_hidden_dim      : int (mlp only, default 512)
        quantum_mlp_n_hidden_layers : int (mlp only, default 2)
        quantum_mlp_activation      : "gelu"|"relu"|"silu" (mlp only, default "gelu")
        img_size                    : (H, W, C) tuple (required for mlp; sets output_dim = H*W*C)
    """
    source = settings.get("noise_source", "gaussian")
    if source == "gaussian":
        return GaussianNoiseSampler()
    if source != "quantum":
        raise ValueError(f"Unknown noise_source '{source}'. Choose 'gaussian' or 'quantum'.")

    qpath = settings.get("quantum_data_path")
    if not qpath:
        raise ValueError("noise_source='quantum' requires 'quantum_data_path' in config.")

    projection = settings.get("quantum_projection", "linear")
    cls = _QUANTUM_REGISTRY.get(projection)
    if cls is None:
        raise ValueError(
            f"Unknown quantum_projection '{projection}'. "
            f"Choose one of: {sorted(_QUANTUM_REGISTRY)}"
        )

    common = {
        "quantum_data_path": qpath,
        "projection_seed": int(settings.get("quantum_projection_seed", 42)),
    }

    if projection in ("multi_sample", "multi_sample_pe"):
        # patch_size: int (square) OR a 2-list/tuple (rectangular).
        # Rectangular is needed for k values without a square divisor
        # (e.g. patch_size=[2,4] -> k=98 on 28x28, full theoretical rank).
        raw_ps = settings.get("quantum_patch_size", 7)
        if isinstance(raw_ps, (list, tuple)):
            patch_size = [int(x) for x in raw_ps]
        else:
            patch_size = int(raw_ps)
        kwargs = dict(
            tiling=str(settings.get("quantum_tiling", "row")),
            patch_size=patch_size,
            **common,
        )
        if projection == "multi_sample_pe":
            kwargs["pe_scale"] = float(settings.get("quantum_pe_scale", 1.0))
        return cls(**kwargs)
    if projection == "hybrid":
        return cls(
            alpha=float(settings.get("quantum_alpha", 0.5)),
            **common,
        )
    if projection == "fourier":
        return cls(
            n_frequencies=int(settings.get("quantum_fourier_freqs", 49)),
            max_freq=float(settings.get("quantum_fourier_max_freq", 32.0)),
            freq_init=str(settings.get("quantum_fourier_init", "log")),
            **common,
        )
    if projection == "mlp":
        # output_dim is derived from img_size (H, W, C); resolve_settings
        # already populates settings["img_size"]. Failing loudly here is
        # better than letting a shape mismatch surface later in training.
        img_size = settings.get("img_size")
        if img_size is None or len(img_size) != 3:
            raise ValueError(
                "quantum_projection='mlp' requires settings['img_size'] = (H, W, C). "
                f"Got: {img_size!r}"
            )
        h, w, c = img_size
        output_dim = int(h) * int(w) * int(c)
        return cls(
            output_dim=output_dim,
            k=int(settings.get("quantum_mlp_k", 96)),
            hidden_dim=int(settings.get("quantum_mlp_hidden_dim", 512)),
            n_hidden_layers=int(settings.get("quantum_mlp_n_hidden_layers", 2)),
            activation=str(settings.get("quantum_mlp_activation", "gelu")),
            **common,
        )
    return cls(**common)
