import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(settings):
    if settings["cuda"] and torch.cuda.is_available():
        torch.cuda.set_device(settings["gpu_id"])
        return torch.device(f"cuda:{settings['gpu_id']}")
    return torch.device("cpu")


def prepare_run_dir(run_dir_str, model_name):
    repo_root = Path(__file__).resolve().parents[2]
    allowed_root = (repo_root / "runs" / model_name).resolve()
    run_dir = Path(run_dir_str).expanduser()
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir
    run_dir = run_dir.resolve()
    if run_dir != allowed_root and allowed_root not in run_dir.parents:
        raise ValueError(f"run_dir must be inside '{allowed_root}'. Got '{run_dir}'.")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_settings(cfg) -> dict:
    """Convert Hydra DictConfig to a validated plain-dict settings object."""
    from omegaconf import OmegaConf
    s = OmegaConf.to_container(cfg, resolve=True)

    s["dataset"] = str(s.get("dataset", "mnist")).lower()
    s["backbone"] = str(s.get("backbone", "unet")).lower()
    s["noise_source"] = str(s.get("noise_source", "gaussian")).lower()

    s["dataset_path"] = str(Path(s.get("dataset_path", "~/datasets")).expanduser())
    qpath = s.get("quantum_data_path", "")
    s["quantum_data_path"] = str(Path(qpath).expanduser()) if qpath else ""

    if s["dataset"] not in {"mnist", "cifar10"}:
        raise ValueError(f"Unsupported dataset '{s['dataset']}'.")
    if s["backbone"] not in {"convstack", "unet"}:
        raise ValueError(f"Unsupported backbone '{s['backbone']}'. Choose 'convstack' or 'unet'.")
    if s["noise_source"] not in {"gaussian", "quantum"}:
        raise ValueError(f"Unsupported noise_source '{s['noise_source']}'. Choose 'gaussian' or 'quantum'.")
    if s["noise_source"] == "quantum" and not s["quantum_data_path"]:
        raise ValueError("noise_source='quantum' requires a non-empty 'quantum_data_path'.")

    # Phase 1: validate quantum_projection early so typos fail before data load.
    s["quantum_projection"] = str(s.get("quantum_projection", "linear")).lower()
    _allowed_proj = {"linear", "multi_sample", "multi_sample_pe", "hybrid", "fourier", "mlp"}
    if s["noise_source"] == "quantum" and s["quantum_projection"] not in _allowed_proj:
        raise ValueError(
            f"Unsupported quantum_projection '{s['quantum_projection']}'. "
            f"Choose one of: {sorted(_allowed_proj)}"
        )

    s["img_size"] = (32, 32, 3) if s["dataset"] == "cifar10" else (28, 28, 1)
    s["hidden_dims"] = [s.get("hidden_dim", 256)] * s.get("n_layers", 8)

    s.pop("hydra", None)
    return s
