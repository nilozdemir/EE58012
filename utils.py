"""Logging, and saving and loading checkpoints."""

import csv

import torch


def checkpoint_path(arch, seed):
    """Return the file of the clean model, e.g. "checkpoints/resnet18_clean_seed0.pt"."""
    return f"checkpoints/{arch}_clean_seed{seed}.pt"


def save_log(path, rows):
    """Write the per-epoch log of a run as a CSV file.

    Args:
        path (str): File to write, for example "results/mnist_sgd_step_seed0.csv".
        rows (list[dict]): One dict per epoch, all with the same keys; the
            keys become the header line.
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, rows[0])
        writer.writeheader()
        writer.writerows(rows)


def load_log(path):
    """Read a log written by save_log.

    Args:
        path (str): File to read.

    Returns:
        dict[str, list[float]]: Maps each column, for example "test_acc", to
            its values, one per epoch.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {key: [float(row[key]) for row in rows] for key in rows[0]}


def save_checkpoint(path, model, optimizer, epoch):
    """Save the model weights and the training state to a file.

    Args:
        path (str): File to write, for example "checkpoints/mnist_sgd_step_seed0.pt".
        model (nn.Module): Network whose state_dict is saved.
        optimizer (torch.optim.Optimizer): Optimiser whose state is saved,
            needed only to resume training.
        epoch (int): Number of epochs trained so far.
    """
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """Load a file written by save_checkpoint into an existing model.

    map_location lets the file load on a machine without GPU.

    Args:
        path (str): File to read.
        model (nn.Module): Network of the same architecture; its weights are
            replaced in place.
        optimizer (torch.optim.Optimizer, optional): If given, its state is
            restored as well.
        device (str): Device the tensors are loaded to, "cpu" or "cuda".

    Returns:
        int: The epoch stored in the checkpoint.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"]
