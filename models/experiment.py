"""
Shared experiment infrastructure: W&B logging, FID evaluation, control-run support.

Used by both models/CFM/train.py and models/CFM_torchcfm/train.py.
"""

import os
from pathlib import Path

import torch
from torchvision.utils import make_grid


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

_wandb_enabled = False


def init_wandb(settings, run_dir, pipeline_name):
    """Initialize W&B run. No-op if wandb is disabled or not installed."""
    global _wandb_enabled
    if not settings.get("use_wandb", True):
        print("[experiment] W&B disabled via config.")
        return

    try:
        import wandb
    except ImportError:
        print("[experiment] wandb not installed -- skipping W&B logging.")
        return

    if not os.environ.get("WANDB_API_KEY") and wandb.api.api_key is None:
        print("[experiment] W&B not logged in -- skipping. Run `wandb login` first.")
        return

    run_name = Path(run_dir).name
    noise = settings.get("noise_source", "gaussian")
    tags = [pipeline_name, settings.get("backbone", "unet"), noise]
    if settings.get("cfm_variant"):
        tags.append(str(settings["cfm_variant"]))
    tags.append("control" if noise != "quantum" else "quantum-run")

    wandb.init(
        project=settings.get("wandb_project", "msc-thesis"),
        group=pipeline_name,
        name=run_name,
        tags=tags,
        config=settings,
        dir=str(run_dir),
        reinit="finish_previous",
    )
    _wandb_enabled = True
    print(f"[experiment] W&B initialized: project={wandb.run.project}, run={wandb.run.name}")


def log_metrics(metrics_dict, step=None):
    """Log a dict of metrics to W&B. No-op if disabled."""
    if not _wandb_enabled:
        return
    import wandb
    wandb.log(metrics_dict, step=step)


def log_images(key, images_tensor, caption=None, step=None):
    """Log a grid of images to W&B. images_tensor: [N, C, H, W] float."""
    if not _wandb_enabled:
        return
    import wandb
    grid = make_grid(images_tensor.detach().cpu(), nrow=8, normalize=True).permute(1, 2, 0).numpy()
    wandb.log({key: wandb.Image(grid, caption=caption)}, step=step)


def log_summary(key, value):
    """Set a W&B summary metric (appears in the runs table)."""
    if not _wandb_enabled:
        return
    import wandb
    wandb.run.summary[key] = value


def finish_wandb():
    """Finalize the W&B run."""
    global _wandb_enabled
    if not _wandb_enabled:
        return
    import wandb
    wandb.finish()
    _wandb_enabled = False


def log_model_artifact(checkpoint_path, artifact_name: str, metadata=None):
    """Upload a checkpoint file as a W&B Model artifact (does not remove the local file).

    Artifacts count against your W&B storage quota; use sparingly for large checkpoints.
    """
    if not _wandb_enabled:
        return
    path = Path(checkpoint_path)
    if not path.is_file():
        print(f"[experiment] W&B artifact skip: not a file: {path}")
        return
    try:
        import wandb

        art = wandb.Artifact(artifact_name, type="model", metadata=metadata or {})
        art.add_file(str(path))
        wandb.log_artifact(art)
        print(f"[experiment] Logged W&B model artifact: {artifact_name} ({path.name})")
    except Exception as exc:
        print(f"[experiment] W&B model artifact failed: {exc}")


# ---------------------------------------------------------------------------
# FID evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_fid(generate_fn, test_loader, num_fid_samples, device):
    """Compute FID between generated samples and real test data.

    Args:
        generate_fn: callable(n) -> tensor [n, C, H, W] of generated images in [0, 1]
        test_loader: DataLoader yielding (images, labels) with images in [0, 1]
        num_fid_samples: how many samples to use for FID (both real and fake)
        device: torch device

    Returns:
        FID score (float), or None if computation fails.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("[experiment] torchmetrics[image] not installed -- skipping FID.")
        return None

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    real_count = 0
    for images, _ in test_loader:
        images = images.to(device)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        fid.update(images, real=True)
        real_count += images.shape[0]
        if real_count >= num_fid_samples:
            break

    generated_count = 0
    batch_size = min(64, num_fid_samples)
    while generated_count < num_fid_samples:
        n = min(batch_size, num_fid_samples - generated_count)
        fake = generate_fn(n)
        fake = fake.clamp(0, 1)
        if fake.shape[1] == 1:
            fake = fake.repeat(1, 3, 1, 1)
        fid.update(fake.to(device), real=False)
        generated_count += n

    try:
        score = fid.compute().item()
        print(f"[experiment] FID score: {score:.2f} ({real_count} real, {generated_count} fake)")
        return score
    except Exception as e:
        print(f"[experiment] FID computation failed: {e}")
        return None
