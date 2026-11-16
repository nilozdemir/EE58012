"""CIFAR-10 datasets and build_loaders(), as in lab-2/utils.py of the course repository.

ToTensor() scales pixels to [0, 1] and is the only transform besides the
training augmentation; the standardisation is the model's Normalize layer.
CIFAR-10 is downloaded from the course's mirror; torchvision still checks the MD5.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

datasets.CIFAR10.url = (
    "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"
)

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def get_splits(root="data"):
    """Return the training, validation and test datasets of CIFAR-10.

    Training images are cropped (padding 4) and flipped; a fixed permutation
    (seed 0) holds out 5,000 of them for validation.

    Args:
        root (str): Directory the dataset is downloaded to.

    Returns:
        dict[str, Dataset]: "train" (augmented), "val" and "test"; each item is an
            image tensor of shape (3, 32, 32) in [0, 1] and an integer label.
    """
    plain = transforms.ToTensor()
    augment = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train_aug = datasets.CIFAR10(root, train=True, download=True, transform=augment)
    train = datasets.CIFAR10(root, train=True, download=True, transform=plain)
    test = datasets.CIFAR10(root, train=False, download=True, transform=plain)
    perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(0))
    val_idx, train_idx = perm[:5000].tolist(), perm[5000:].tolist()
    return {
        "train": Subset(train_aug, train_idx),
        "val": Subset(train, val_idx),
        "test": test,
    }


def build_loaders(args):
    """Return the training, validation and test loaders.

    Args:
        args (argparse.Namespace): Uses args.batch_size and args.workers.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Shuffled, augmented training
            batches, and validation and test batches without augmentation.
    """
    splits = get_splits()
    train_loader = DataLoader(
        splits["train"], batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(splits["val"], batch_size=args.batch_size)
    test_loader = DataLoader(splits["test"], batch_size=args.batch_size)
    return train_loader, val_loader, test_loader


def get_attack_set(root="data"):
    """Return the CIFAR-10 test set as tensors in [0, 1].

    Args:
        root (str): Directory the dataset is downloaded to.

    Returns:
        tuple[Tensor, Tensor]: Images, shape (10000, 3, 32, 32) in [0, 1], and
            labels, shape (10000,).
    """
    test = datasets.CIFAR10(root, train=False, download=True)
    x = torch.tensor(test.data).permute(0, 3, 1, 2).float() / 255.0
    y = torch.tensor(test.targets)
    return x, y
