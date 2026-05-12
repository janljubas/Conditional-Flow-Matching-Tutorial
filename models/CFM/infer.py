#!/usr/bin/env python3
"""Inference script for trained CFM checkpoints (Hydra-driven).

Usage:
  python models/CFM/infer.py \\
    +checkpoint_path=runs/CFM/hybrid/<run>/checkpoints/last.pt \\
    run_dir=runs/CFM/hybrid/<run>/inference_<tag> \\
    num_sample_images=128 sample_steps=50

Notes:
- Settings stored inside the checkpoint take precedence (so a quantum-trained
  model is sampled with the matching quantum noise sampler). The CLI / config
  is consulted only for inference-time knobs (run_dir, num_sample_images,
  sample_steps, gpu_id, num_classes for sampling labels) and overrides.
- Saves samples to <run_dir>/samples/ and a small inference_metrics.json.
"""

import json
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import build_velocity_net  # noqa: E402
from noise import build_noise_sampler  # noqa: E402
from utils import (  # noqa: E402
    prepare_run_dir,
    resolve_device,
    resolve_settings,
    seed_everything,
)


# Inference-time overrides take precedence over checkpoint settings.
# Everything else (architecture, noise source/projection, dataset, etc.)
# comes from the checkpoint so the model loads correctly.
_INFER_OVERRIDE_KEYS = {
    "run_dir",
    "num_sample_images",
    "sample_steps",
    "gpu_id",
    "cuda",
    "num_workers",
    "seed",
    "use_wandb",
}


def _merge_settings(ckpt_settings: dict, cli_settings: dict) -> dict:
    """Use checkpoint settings as the base; allow CLI to override only a small
    whitelist of inference-time knobs. This prevents an inference-time CLI
    from accidentally building a different architecture than the trained one.
    """
    merged = dict(ckpt_settings)
    for k in _INFER_OVERRIDE_KEYS:
        if k in cli_settings and cli_settings[k] is not None:
            merged[k] = cli_settings[k]
    if "img_size" not in merged:
        merged["img_size"] = (32, 32, 3) if merged.get("dataset") == "cifar10" else (28, 28, 1)
    return merged


@torch.no_grad()
def run_inference(settings: dict, checkpoint_path: Path, run_dir: Path):
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(settings["seed"])
    device = resolve_device(settings)

    model = build_velocity_net(settings).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    noise_sampler = (
        build_noise_sampler(settings)
        if settings.get("noise_source") == "quantum"
        else None
    )

    # If the sampler is an nn.Module (e.g. Strategy 5: QuantumSamplerMLP),
    # load its trained weights from the checkpoint (saved by train.py under
    # "noise_sampler_state_dict") and put it in eval mode on the right device.
    if isinstance(noise_sampler, torch.nn.Module):
        noise_sampler.to(device)
        sampler_sd = ckpt.get("noise_sampler_state_dict")
        if sampler_sd is None:
            raise RuntimeError(
                "Trainable noise sampler requested but checkpoint has no "
                "'noise_sampler_state_dict' key. Was the model trained before "
                "the trainable-sampler refactor? Re-train with the latest train.py."
            )
        noise_sampler.load_state_dict(sampler_sd)
        noise_sampler.eval()

    h, w, c = settings["img_size"]
    n = int(settings.get("num_sample_images", 64))
    steps = int(settings.get("sample_steps", 25))
    shape = [n, c, h, w]

    # Class-conditional sampling: spread labels uniformly across the batch.
    num_classes = int(settings.get("num_classes", 0) or 0)
    class_labels = (
        torch.arange(n, device=device) % num_classes if num_classes > 0 else None
    )

    generated = model.sample(
        t_steps=steps,
        shape=shape,
        device=device,
        noise_sampler=noise_sampler,
        class_labels=class_labels,
    )
    out_path = samples_dir / f"samples_steps{steps}.png"
    save_image(generated, out_path, nrow=8, normalize=True)

    metrics = {
        "pipeline": "cfm_inference",
        "checkpoint": str(checkpoint_path),
        "output_image": str(out_path),
        "num_samples": n,
        "sample_steps": steps,
        "noise_source": settings.get("noise_source"),
        "quantum_projection": settings.get("quantum_projection") if settings.get("noise_source") == "quantum" else None,
        "settings": settings,
    }
    (run_dir / "inference_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps({k: v for k, v in metrics.items() if k != "settings"}, indent=2))


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    cli_settings = resolve_settings(cfg)

    ckpt_path_str = cli_settings.get("checkpoint_path", "")
    if not ckpt_path_str:
        raise ValueError(
            "Missing required override: pass +checkpoint_path=<path> on the CLI "
            "(e.g. +checkpoint_path=runs/CFM/hybrid/<run>/checkpoints/last.pt)."
        )
    checkpoint_path = Path(ckpt_path_str).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_settings = ckpt.get("settings", {}) or {}
    settings = _merge_settings(ckpt_settings, cli_settings)

    run_dir = prepare_run_dir(settings.get("run_dir") or str(checkpoint_path.parent.parent / "inference"), "CFM")
    run_inference(settings, checkpoint_path, run_dir)


if __name__ == "__main__":
    main()
