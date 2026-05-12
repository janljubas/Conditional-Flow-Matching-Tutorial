#!/usr/bin/env python3
"""Train CFM model using the torchcfm library (Tong et al.)."""

import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig
from torch.optim import AdamW
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid, save_image

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
from torchcfm.models.unet import UNetModel as UNetModelWrapper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment import (
    init_wandb,
    log_metrics,
    log_images,
    log_summary,
    finish_wandb,
    compute_fid,
    log_model_artifact,
)
from noise import build_noise_sampler
from utils import prepare_run_dir, resolve_device, resolve_settings, seed_everything


def load_dataset(dataset, dataset_path, train_batch_size, inference_batch_size, num_workers):
    from torch.utils.data import DataLoader

    kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    transform = ToTensor()
    if dataset == "cifar10":
        train_dataset = CIFAR10(dataset_path, transform=transform, train=True, download=True)
        test_dataset = CIFAR10(dataset_path, transform=transform, train=False, download=True)
    else:
        train_dataset = MNIST(dataset_path, transform=transform, train=True, download=True)
        test_dataset = MNIST(dataset_path, transform=transform, train=False, download=True)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=inference_batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


def save_training_curve(run_dir: Path, history):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["epoch"], history["train_loss"], label="train_loss")
    ax.set_title("CFM (torchcfm) Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "train_loss.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def sample_euler(model, t_steps, shape, device, noise_sampler=None):
    """Generate samples by Euler-integrating the learned ODE."""
    if noise_sampler is not None:
        x = noise_sampler.sample(shape, device)
    else:
        x = torch.randn(size=shape, device=device)
    delta = 1.0 / max(t_steps - 1, 1)
    t_vals = torch.linspace(0, 1, t_steps, device=device)

    for i in range(t_steps - 1):
        t_cur = torch.full((shape[0],), t_vals[i].item(), device=device)
        x = x + model(t_cur, x) * delta
    return x


@torch.no_grad()
def save_sample_grid(model, run_dir: Path, sample_steps, num_sample_images, img_size, device, noise_sampler=None):
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    h, w, c = img_size
    generated = sample_euler(model, t_steps=sample_steps, shape=[num_sample_images, c, h, w], device=device, noise_sampler=noise_sampler)
    save_image(generated, samples_dir / f"samples_steps{sample_steps}.png", nrow=8, normalize=True)

    fig = plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.title(f"torchcfm samples (steps={sample_steps})")
    plt.imshow(make_grid(generated.detach().cpu(), padding=2, normalize=True).permute(1, 2, 0))
    fig.tight_layout()
    fig.savefig(samples_dir / f"grid_steps{sample_steps}.png", dpi=150)
    plt.close(fig)


def build_cfm(settings):
    """Build the flow matcher object based on config."""
    sigma = settings["cfm_sigma"]
    if settings["cfm_variant"] == "otcfm":
        return ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    return ConditionalFlowMatcher(sigma=sigma)


def train(settings, run_dir: Path):
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(settings["seed"])
    device = resolve_device(settings)

    h, w, c = settings["img_size"]
    model = UNetModelWrapper(
        dim=(c, h, w),
        num_channels=settings["num_channels"],
        num_res_blocks=settings["num_res_blocks"],
    ).to(device)

    noise_sampler = build_noise_sampler(settings)

    # Some samplers (e.g. Strategy 5: QuantumSamplerMLP) are nn.Modules with
    # trainable parameters. Move to device, set train mode, and combine
    # their params with the velocity net's into a single optimizer.
    sampler_is_module = isinstance(noise_sampler, torch.nn.Module)
    if sampler_is_module:
        noise_sampler.to(device)
        noise_sampler.train()

    nparams = sum(p.numel() for p in model.parameters())
    sampler_nparams = (
        sum(p.numel() for p in noise_sampler.parameters()) if sampler_is_module else 0
    )
    print(
        f"torchcfm UNet | Noise: {settings['noise_source']} "
        f"| Velocity-net params: {nparams:,}"
        + (f" | Sampler params: {sampler_nparams:,}" if sampler_is_module else "")
        + f" | CFM variant: {settings['cfm_variant']}"
    )
    log_summary("num_params", nparams)
    if sampler_is_module:
        log_summary("num_sampler_params", sampler_nparams)

    trainable_params = list(model.parameters())
    if sampler_is_module:
        trainable_params += list(noise_sampler.parameters())
    optimizer = AdamW(trainable_params, lr=settings["lr"], betas=(0.9, 0.99))
    cfm = build_cfm(settings)

    train_loader, test_loader = load_dataset(
        settings["dataset"],
        settings["dataset_path"],
        settings["train_batch_size"],
        settings["inference_batch_size"],
        settings["num_workers"],
    )

    history = {"epoch": [], "train_loss": []}
    print("Start training (torchcfm)...")
    model.train()
    start = time.time()
    global_step = 0

    for epoch in range(settings["n_epochs"]):
        epoch_start = time.time()
        total_loss = 0.0
        for batch_idx, (x_1, _) in enumerate(train_loader):
            optimizer.zero_grad()
            x_1 = x_1.to(device)
            x_0 = noise_sampler.sample_like(x_1)

            t, x_t, u_t = cfm.sample_location_and_conditional_flow(x_0, x_1)
            t = t.to(device)
            x_t = x_t.to(device)
            u_t = u_t.to(device)

            v_pred = model(t, x_t)

            loss = ((v_pred - u_t) ** 2).mean()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            global_step += 1

            if batch_idx % 100 == 0:
                grad_norm = sum(p.grad.norm().item() for p in trainable_params if p.grad is not None)
                print(f"\t\tCFM loss: {loss.item():.6f}  grad_norm: {grad_norm:.4f}")
                log_metrics({"batch_loss": loss.item(), "grad_norm": grad_norm}, step=global_step)

        epoch_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(epoch_loss))
        print(f"\tEpoch {epoch + 1} complete!\tCFM loss: {epoch_loss:.6f}")
        log_metrics({"train_loss": epoch_loss, "epoch": epoch + 1, "epoch_time_sec": epoch_time}, step=global_step)

        epoch_num = epoch + 1
        ce = max(1, int(settings["checkpoint_every"]))
        save_epochs = settings.get("save_epoch_checkpoints", True)
        wb_every = int(settings.get("wandb_log_model_artifact_every", 0) or 0)
        disk_save = save_epochs and (epoch_num % ce == 0 or epoch_num == settings["n_epochs"])
        wb_mid = wb_every > 0 and (epoch_num % wb_every == 0 or epoch_num == settings["n_epochs"])

        ckpt_payload = {
            "epoch": epoch_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "settings": settings,
            "epoch_loss": float(epoch_loss),
        }
        if sampler_is_module:
            ckpt_payload["noise_sampler_state_dict"] = noise_sampler.state_dict()

        ckpt_path = None
        if disk_save:
            ckpt_path = checkpoints_dir / f"epoch_{epoch_num:04d}.pt"
            torch.save(ckpt_payload, ckpt_path)

        if wb_mid:
            if ckpt_path is not None and ckpt_path.is_file():
                log_model_artifact(
                    ckpt_path,
                    f"checkpoint-epoch-{epoch_num}",
                    metadata={"epoch": epoch_num, "epoch_loss": float(epoch_loss)},
                )
            else:
                fd, tmp_name = tempfile.mkstemp(suffix=".pt", prefix=f"ckpt_e{epoch_num}_", dir=str(run_dir))
                os.close(fd)
                tmp_path = Path(tmp_name)
                try:
                    torch.save(ckpt_payload, tmp_path)
                    log_model_artifact(
                        tmp_path,
                        f"checkpoint-epoch-{epoch_num}",
                        metadata={"epoch": epoch_num, "epoch_loss": float(epoch_loss)},
                    )
                finally:
                    tmp_path.unlink(missing_ok=True)

    elapsed_sec = round(time.time() - start, 4)
    save_last = settings.get("save_last_checkpoint", True)
    last_path = checkpoints_dir / "last.pt"
    if save_last:
        last_payload = {"model_state_dict": model.state_dict(), "settings": settings}
        if sampler_is_module:
            last_payload["noise_sampler_state_dict"] = noise_sampler.state_dict()
        torch.save(last_payload, last_path)
    if settings.get("wandb_log_model_artifact", False) and save_last and last_path.is_file():
        log_model_artifact(last_path, "last-model", metadata={"epoch": settings["n_epochs"]})
    save_training_curve(run_dir, history)

    sample_steps = settings["sample_steps"]
    num_sample_images = settings["num_sample_images"]

    model.eval()
    if sampler_is_module:
        noise_sampler.eval()
    save_sample_grid(model, run_dir, sample_steps, num_sample_images, settings["img_size"], device, noise_sampler)

    generated_for_log = sample_euler(model, t_steps=sample_steps, shape=[min(64, num_sample_images), c, h, w], device=device, noise_sampler=noise_sampler)
    log_images("samples", generated_for_log, caption=f"steps={sample_steps}")

    num_fid = settings.get("num_fid_samples", 1024)
    if num_fid > 0:
        def gen_fn(n):
            return sample_euler(model, t_steps=sample_steps, shape=[n, c, h, w], device=device, noise_sampler=noise_sampler)
        fid_score = compute_fid(gen_fn, test_loader, num_fid, device)
        if fid_score is not None:
            log_summary("fid", fid_score)
            history["fid"] = fid_score
    else:
        fid_score = None

    log_summary("final_train_loss", history["train_loss"][-1])
    log_summary("elapsed_sec", elapsed_sec)

    return {"history": history, "elapsed_sec": elapsed_sec, "fid": fid_score}


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    settings = resolve_settings(cfg)

    if settings.get("run_dir") is None:
        repo_root = Path(__file__).resolve().parents[2]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        settings["run_dir"] = str(repo_root / "runs" / "CFM_torchcfm" / timestamp)

    run_dir = prepare_run_dir(settings["run_dir"], "CFM_torchcfm")

    init_wandb(settings, run_dir, pipeline_name="cfm-torchcfm")
    out = train(settings, run_dir)

    metrics = {
        "pipeline": "cfm-torchcfm",
        "elapsed_sec": out["elapsed_sec"],
        "fid": out.get("fid"),
        "settings": settings,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (run_dir / "history.json").write_text(json.dumps(out["history"], indent=2))

    finish_wandb()
    print("Finish!!")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
