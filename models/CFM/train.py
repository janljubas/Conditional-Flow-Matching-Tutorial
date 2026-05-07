#!/usr/bin/env python3
"""Train Conditional Flow Matching model (custom pipeline)."""

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
from model import build_velocity_net
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
    ax.set_title("CFM Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "train_loss.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def save_sample_grid(model, run_dir: Path, sample_steps, num_sample_images, img_size, device,
                     noise_sampler=None, class_labels=None):
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    h, w, c = img_size
    generated = model.sample(t_steps=sample_steps, shape=[num_sample_images, c, h, w], device=device,
                             noise_sampler=noise_sampler, class_labels=class_labels)
    save_image(generated, samples_dir / f"samples_steps{sample_steps}.png", nrow=8, normalize=True)

    fig = plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.title(f"CFM samples (steps={sample_steps})")
    plt.imshow(make_grid(generated.detach().cpu(), padding=2, normalize=True).permute(1, 2, 0))
    fig.tight_layout()
    fig.savefig(samples_dir / f"grid_steps{sample_steps}.png", dpi=150)
    plt.close(fig)


def train(settings, run_dir: Path):
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(settings["seed"])
    device = resolve_device(settings)
    model = build_velocity_net(settings).to(device)
    noise_sampler = build_noise_sampler(settings)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Backbone: {settings['backbone']} | Noise: {settings['noise_source']} | Params: {nparams:,}")
    log_summary("num_params", nparams)
    optimizer = AdamW(model.parameters(), lr=settings["lr"], betas=(0.9, 0.99))

    train_loader, test_loader = load_dataset(
        settings["dataset"],
        settings["dataset_path"],
        settings["train_batch_size"],
        settings["inference_batch_size"],
        settings["num_workers"],
    )

    use_class_cond = settings.get("num_classes", 0) > 0

    history = {"epoch": [], "train_loss": []}
    print("Start training CFM...")
    model.train()
    start = time.time()
    global_step = 0

    for epoch in range(settings["n_epochs"]):
        epoch_start = time.time()
        total_loss = 0.0
        for batch_idx, (x_1, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            x_1 = x_1.to(device)
            class_labels = labels.to(device) if use_class_cond else None
            x_0 = noise_sampler.sample_like(x_1)
            t = torch.rand(x_1.shape[0], 1, 1, 1, device=device)

            x_t = model.interpolate(x_0, x_1, t)
            velocity_target = model.get_velocity(x_0, x_1)
            velocity_pred = model(x_t, t, class_labels=class_labels)

            loss = ((velocity_pred - velocity_target) ** 2).mean()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            global_step += 1

            if batch_idx % 100 == 0:
                grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
                print("\t\tCFM loss:", loss.item(), " grad_norm:", grad_norm)
                log_metrics({"batch_loss": loss.item(), "grad_norm": grad_norm}, step=global_step)

        epoch_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(epoch_loss))
        print("\tEpoch", epoch + 1, "complete!\tCFM loss:", epoch_loss)
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
        torch.save({"model_state_dict": model.state_dict(), "settings": settings}, last_path)
    if settings.get("wandb_log_model_artifact", False) and save_last and last_path.is_file():
        log_model_artifact(last_path, "last-model", metadata={"epoch": settings["n_epochs"]})
    save_training_curve(run_dir, history)

    h, w, c = settings["img_size"]
    sample_steps = settings["sample_steps"]
    num_sample_images = settings["num_sample_images"]

    def _rand_labels(n):
        if use_class_cond:
            return torch.randint(0, settings["num_classes"], (n,), device=device)
        return None

    model.eval()
    save_sample_grid(model, run_dir, sample_steps, num_sample_images, settings["img_size"], device,
                     noise_sampler=noise_sampler, class_labels=_rand_labels(num_sample_images))

    n_log = min(64, num_sample_images)
    generated_for_log = model.sample(t_steps=sample_steps, shape=[n_log, c, h, w], device=device,
                                     noise_sampler=noise_sampler, class_labels=_rand_labels(n_log))
    log_images("samples", generated_for_log, caption=f"steps={sample_steps}")

    num_fid = settings.get("num_fid_samples", 1024)
    if num_fid > 0:
        def gen_fn(n):
            return model.sample(t_steps=sample_steps, shape=[n, c, h, w], device=device,
                                noise_sampler=noise_sampler, class_labels=_rand_labels(n))
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
        settings["run_dir"] = str(repo_root / "runs" / "CFM" / timestamp)

    run_dir = prepare_run_dir(settings["run_dir"], "CFM")

    init_wandb(settings, run_dir, pipeline_name="cfm-custom")
    out = train(settings, run_dir)

    metrics = {
        "pipeline": "cfm-custom",
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
