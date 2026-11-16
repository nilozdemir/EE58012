"""build_model(): the Normalize layer followed by the CIFAR ResNet-18.

Same as lab-2/models/__init__.py of the course repository (Self-Study
Assignment 2), so checkpoints/resnet18_clean_seed0.pt written by either project
loads in the other. Images enter the model in [0, 1]; the standardisation is
the model's first layer, so x' = x + m * delta is formed in pixel space and
normalised afterwards, as the assignment requires.
"""

import torch
import torch.nn as nn

from models.resnet import ResNet18

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class Normalize(nn.Module):
    """Standardise every colour channel: (x - mean) / std."""

    def __init__(self, mean, std):
        """Store the per-channel mean and standard deviation as buffers.

        Args:
            mean (tuple[float]): Mean of each channel.
            std (tuple[float]): Standard deviation of each channel.
        """
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        """Return the standardised images, x of shape (B, C, H, W) in [0, 1]."""
        return (x - self.mean) / self.std


def build_model(arch="resnet18"):
    """Return the untrained model with the Normalize layer in front.

    Args:
        arch (str): Only "resnet18" is used in this assignment.

    Returns:
        nn.Sequential: Normalize, then ResNet-18 with 10 classes; model[1] is
            the network itself (model[1].layer3 is the Grad-CAM layer).
    """
    if arch != "resnet18":
        raise ValueError(arch)
    return nn.Sequential(Normalize(MEAN, STD), ResNet18(num_classes=10))
