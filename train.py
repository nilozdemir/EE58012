"""Training of the clean model: train_one_epoch() and fit().

Same loop and recipe as lab-2/train.py (Self-Study Assignment 2, Section 4.2):
per-iteration scheduler with 1 warm-up epoch, weight decay 5e-4, logged gradient
norm, CSV log every epoch, checkpoint checkpoints/resnet18_clean_seed<seed>.pt.
"""

import os
import time

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LinearLR, MultiStepLR, PolynomialLR, SequentialLR,
)

from test import evaluate
from utils import checkpoint_path, load_rows, save_checkpoint, save_log

WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 1


def train_one_epoch(model, loader, optimizer, device, scheduler=None):
    """Train the model for one epoch.

    Args:
        model (nn.Module): Network to train. It is switched to train mode.
        loader (DataLoader): Yields mini-batches (x, y); x has shape (B, 3, 32, 32).
        optimizer (torch.optim.Optimizer): Optimiser over model.parameters().
        device (str): "cpu" or "cuda".
        scheduler (optional): Learning-rate scheduler, stepped once per iteration.

    Returns:
        tuple[float, float]: Mean training loss, and mean global gradient norm.
    """
    model.train()
    total, grad_norm = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        grad_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset), grad_norm / len(loader)


def build_optimizer(params, name, lr, weight_decay):
    """Return SGD, SGD with momentum 0.9, or AdamW.

    Args:
        params (iterable): model.parameters().
        name (str): "sgd", "momentum" or "adam".
        lr (float): Base learning rate.
        weight_decay (float): Weight-decay coefficient.

    Returns:
        torch.optim.Optimizer: The optimiser.
    """
    if name == "sgd":
        return torch.optim.SGD(params, lr, weight_decay=weight_decay)
    if name == "momentum":
        return torch.optim.SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.AdamW(params, lr, weight_decay=weight_decay)
    raise ValueError(name)


def build_scheduler(optimizer, name, total_iters, warmup_iters=0):
    """Return the per-iteration scheduler: linear warm-up, then the main schedule.

    With T = total_iters - warmup_iters (at least 1), "step" multiplies by 0.1 at
    0.5 T and 0.75 T, "poly" gives lr * (1 - t / T)^0.9 and "cosine" gives
    lr * (1 + cos(pi t / T)) / 2.

    Args:
        optimizer (torch.optim.Optimizer): Optimiser whose learning rate is set.
        name (str): "constant", "step", "poly" or "cosine".
        total_iters (int): Number of iterations of the whole run.
        warmup_iters (int): Number of iterations of the linear warm-up.

    Returns:
        LRScheduler or None: The scheduler, None for a constant rate without warm-up.
    """
    T = max(total_iters - warmup_iters, 1)
    if name == "constant":
        main = None
    elif name == "step":
        main = MultiStepLR(optimizer, milestones=[int(0.5 * T), int(0.75 * T)], gamma=0.1)
    elif name == "poly":
        main = PolynomialLR(optimizer, total_iters=T, power=0.9)
    elif name == "cosine":
        main = CosineAnnealingLR(optimizer, T_max=T, eta_min=0.0)
    else:
        raise ValueError(name)
    if warmup_iters == 0:
        return main
    warm = LinearLR(optimizer, start_factor=1 / warmup_iters, total_iters=warmup_iters)
    if main is None:
        return warm
    return SequentialLR(optimizer, [warm, main], milestones=[warmup_iters])


def fit(model, train_loader, val_loader, test_loader, args, device):
    """Train for args.epochs epochs, log and save a checkpoint every epoch.

    Writes results/<run>.csv and checkpoints/<run>.pt after every epoch. If that
    checkpoint already exists from an interrupted run (fewer than args.epochs
    epochs, with optimiser and scheduler state), training resumes after its
    last epoch; if it is already complete, nothing is trained.

    Args:
        model (nn.Module): Network to train, already on device.
        train_loader (DataLoader): Shuffled, augmented training batches.
        val_loader (DataLoader): Validation batches.
        test_loader (DataLoader): Test batches (logged only).
        args (argparse.Namespace): The arguments of parameters.get_parameters()
            (arch, optimizer, scheduler, lr, epochs, seed).
        device (str): "cpu" or "cuda".

    Returns:
        str: Path of the saved checkpoint.
    """
    optimizer = build_optimizer(model.parameters(), args.optimizer, args.lr, WEIGHT_DECAY)
    iters = len(train_loader)
    scheduler = build_scheduler(
        optimizer, args.scheduler, args.epochs * iters, WARMUP_EPOCHS * iters
    )
    run = f"{args.arch}_clean_seed{args.seed}"
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    path, log_path = checkpoint_path(args.arch, args.seed), f"results/{run}.csv"
    log, first = [], 1
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        done = ckpt.get("epoch", args.epochs)
        if done >= args.epochs:
            print(f"{path} already trained for {done} epochs; nothing to do")
            model.load_state_dict(ckpt["model"])
            return path
        if "scheduler" in ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            first = done + 1
            if os.path.exists(log_path):
                log = [r for r in load_rows(log_path) if r["epoch"] <= done]
            print(f"resuming {path} after epoch {done}")
    start = time.time() - (log[-1]["time"] if log else 0.0)
    for epoch in range(first, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss, grad_norm = train_one_epoch(model, train_loader, optimizer, device, scheduler)
        val_loss, val_acc = evaluate(model, val_loader, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        log.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_acc": val_acc, "test_loss": test_loss, "test_acc": test_acc,
            "lr": lr, "grad_norm": grad_norm, "time": time.time() - start,
        })
        print(", ".join(f"{k} {v:.4g}" for k, v in log[-1].items()), flush=True)
        save_log(log_path, log)
        save_checkpoint(path, model, optimizer, epoch, scheduler)
    return path
