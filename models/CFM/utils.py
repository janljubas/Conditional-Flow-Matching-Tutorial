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

    s["img_size"] = (32, 32, 3) if s["dataset"] == "cifar10" else (28, 28, 1)
    s["hidden_dims"] = [s.get("hidden_dim", 256)] * s.get("n_layers", 8)

    s.pop("hydra", None)
    return s


# ---------------------------------------------------------------------------
# Legacy argparse API — kept for infer.py (will be migrated separately)
# ---------------------------------------------------------------------------
import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_yaml_config(config_path: Path):
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyYAML required: pip install pyyaml") from exc
    config = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level mapping.")
    return config


def pick_value(cli_value, config, key, default):
    return cli_value if cli_value is not None else config.get(key, default)


def build_infer_parser():
    parser = argparse.ArgumentParser(description="Run CFM inference from checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--cuda", type=str2bool, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--num-sample-images", type=int, default=None)
    return parser


def resolve_infer_settings(args, config):
    settings = {
        "dataset_path": str(Path(pick_value(args.dataset_path, config, "dataset_path", "~/datasets")).expanduser()),
        "dataset": str(pick_value(args.dataset, config, "dataset", "mnist")).lower(),
        "gpu_id": int(pick_value(args.gpu_id, config, "gpu_id", 0)),
        "cuda": bool(pick_value(args.cuda, config, "cuda", True)),
        "hidden_dim": int(pick_value(args.hidden_dim, config, "hidden_dim", 256)),
        "n_layers": int(pick_value(args.n_layers, config, "n_layers", 8)),
        "sigma_min": float(pick_value(None, config, "sigma_min", 0.0)),
        "inference_batch_size": int(pick_value(args.inference_batch_size, config, "inference_batch_size", 64)),
        "sample_steps": int(pick_value(args.sample_steps, config, "sample_steps", 25)),
        "num_sample_images": int(pick_value(args.num_sample_images, config, "num_sample_images", 64)),
        "seed": int(pick_value(args.seed, config, "seed", 1234)),
        "num_workers": int(pick_value(args.num_workers, config, "num_workers", 1)),
    }
    if settings["dataset"] not in {"mnist", "cifar10"}:
        raise ValueError(f"Unsupported dataset '{settings['dataset']}'.")
    settings["img_size"] = (32, 32, 3) if settings["dataset"] == "cifar10" else (28, 28, 1)
    settings["hidden_dims"] = [settings["hidden_dim"]] * settings["n_layers"]
    return settings
