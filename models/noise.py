"""
Noise source abstraction for CFM training and sampling.

Available samplers
--------------------------------------------
  (Phase 0)
  GaussianNoiseSampler          - standard N(0, I)
  QuantumSamplerLinear          - rank-8 single linear projection (baseline)

  (Phase 1)
  QuantumSamplerMultiSample     - Strategy 1: k samples tiled spatially
  QuantumSamplerMultiSamplePE   - Strategy 2: Strategy 1 + sinusoidal PE
  QuantumSamplerHybrid          - Strategy 3: alpha * q + sqrt(1 - alpha^2) * eps
  QuantumSamplerFourier         - Strategy 4: sin/cos basis expansion, then projection

All quantum samplers share:
  - Pool loading from .pt files in `quantum_data_path`
  - Per-dimension affine rescaling to zero mean / unit variance
  - A fixed `projection_seed` for reproducible projection matrices

Why these strategies?
The current quantum pool is 8-dim, MNIST is 784-dim. A single linear
projection R^8 -> R^784 has rank 8: only an 8-dimensional affine subspace
of pixel space is reachable, the other 776 dims are deterministic given
which 8-D point was drawn. With a finite pool (~64,720 samples) every
MNIST image effectively gets a near-deterministic low-dim "tag", and the
model overfits the (proj, x_1) coupling -- low train loss, terrible FID.
The strategies from Phase 1 try to overcome this failure mode by either raising the effective rank of the projected
noise (Multi-sample, Multi-sample with PE, Fourier) or guarantee full rank by construction (Hybrid).
"""

import math
from pathlib import Path

import torch


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

    def _get_projection(self, in_dim, out_dim, device, dtype, name="proj"):
        """
        Cached fixed-random projection matrix of shape [in_dim, out_dim].

        Variance normalization (1/sqrt(in_dim)) keeps output entries unit-
        variance when input entries are unit-variance Gaussian, matching
        the marginal scale of an N(0, I) baseline.

        `name` namespaces the cache so different strategies (linear, multi,
        fourier) generate independent projection matrices instead of sharing.
        """
        key = (name, in_dim, out_dim, device, dtype)
        if key not in self._proj_cache:
            seed = self._projection_seed + (hash(name) % (2**31))
            gen = torch.Generator().manual_seed(seed)
            W = torch.randn(in_dim, out_dim, generator=gen, dtype=dtype) / (in_dim ** 0.5)
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
        # Image is split into a grid of (H/p) x (W/p) patches of size p x p.
        if H % patch_size or W % patch_size:
            raise ValueError(
                f"patch_size={patch_size} must divide H={H} and W={W}."
            )
        nh, nw = H // patch_size, W // patch_size
        k = nh * nw
        region_flat = C * patch_size * patch_size

        def to_image(t):
            B = t.shape[0]
            # [B, k, C*p*p] -> [B, nh, nw, C, p, p]
            #               -> [B, C, nh, p, nw, p]   (interleave grid + patch dims)
            #               -> [B, C, H, W]
            return (
                t.view(B, nh, nw, C, patch_size, patch_size)
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
      'patch' : k = (H/p)*(W/p) (each p x p patch gets its own draw)

    Effective rank rises from D (=8) to min(D*k, C*H*W). For row-tiled MNIST
    with k=28 this is 8*28 = 224 (vs 8 in the baseline) -- still incomplete,
    but qualitatively a different regime.

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
        W = self._get_projection(self.qdim, region_flat, device, dtype, name="multi")
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
# Factory
# ---------------------------------------------------------------------------

_QUANTUM_REGISTRY = {
    "linear":          QuantumSamplerLinear,
    "multi_sample":    QuantumSamplerMultiSample,
    "multi_sample_pe": QuantumSamplerMultiSamplePE,
    "hybrid":          QuantumSamplerHybrid,
    "fourier":         QuantumSamplerFourier,
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
        kwargs = dict(
            tiling=str(settings.get("quantum_tiling", "row")),
            patch_size=int(settings.get("quantum_patch_size", 7)),
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
    return cls(**common)
