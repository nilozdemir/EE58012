"""All command-line arguments and their defaults (training and attacks)."""

import argparse


def get_parameters():
    """Parse the arguments for training or testing the clean ResNet-18.

    The defaults are the ResNet-18 recipe of Self-Study Assignment 2 (Section 4.2):
    SGD with momentum 0.9, lr 0.1, cosine schedule with 1 warm-up epoch, 100 epochs,
    batch 128, weight decay 5e-4 (fixed in train.py).
    """
    p = argparse.ArgumentParser(description="Train / test the clean CIFAR-10 ResNet-18")
    p.add_argument("--arch", default="resnet18", choices=["resnet18"])
    p.add_argument("--optimizer", default="momentum", choices=["sgd", "momentum", "adam"])
    p.add_argument("--scheduler", default="cosine",
                   choices=["constant", "step", "poly", "cosine"])
    p.add_argument("--lr", type=float, default=0.1, help="base learning rate")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["train", "test"], default="train")
    p.add_argument("--checkpoint", default=None,
                   help="weights to load (test mode default: checkpoints/resnet18_clean_seed0.pt)")
    return p.parse_args()


def add_attack_arguments(p):
    """Add the arguments shared by every attack script to a parser.

    Args:
        p (argparse.ArgumentParser): Parser to extend.

    Returns:
        argparse.ArgumentParser: The same parser.
    """
    p.add_argument("--checkpoint", default="checkpoints/resnet18_clean_seed0.pt",
                   help="clean ResNet-18 weights (Self-Study 2 checkpoint)")
    p.add_argument("--eps", type=float, default=2.0, help="L-inf bound, in units of 1/255")
    p.add_argument("--steps", type=int, default=100, help="PGD steps for delta")
    p.add_argument("--step-size", type=float, default=None,
                   help="PGD step in units of 1/255 (default 2.5 * eps / steps)")
    p.add_argument("--cam-layer", default="layer3",
                   choices=["layer1", "layer2", "layer3", "layer4"],
                   help="Grad-CAM layer (layer3: 256x8x8, layer4: 512x4x4)")
    p.add_argument("--cam-score", default="auto",
                   choices=["auto", "true", "target", "margin"],
                   help="class score explained by Grad-CAM; auto = true class for "
                        "untargeted, target-minus-true margin for targeted")
    p.add_argument("--mask-steps", type=int, default=100,
                   help="joint mask+delta steps of the learnable mask (Part B)")
    p.add_argument("--mask-lr", type=float, default=0.05, help="Adam lr of the mask scores")
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p


def get_attack_parameters():
    """Parse the arguments of experiments/run_attacks.py (the ASR tables)."""
    p = add_attack_arguments(argparse.ArgumentParser(description="Sparse attack ASR"))
    p.add_argument("--modes", nargs="+", default=["untargeted", "targeted"],
                   choices=["untargeted", "targeted"])
    p.add_argument("--rhos", nargs="+", type=float, default=[0.25, 0.5])
    p.add_argument("--methods", nargs="+",
                   default=["gradcam", "random", "learned_gradcam", "learned_random"],
                   choices=["gradcam", "random", "learned_gradcam", "learned_random"])
    p.add_argument("--n-images", type=int, default=None,
                   help="attack only the first N correctly classified test images "
                        "(default: all of them)")
    p.add_argument("--out", default="results/asr", help="prefix of the .csv/.md/.tex tables")
    return p.parse_args()


def get_figure_parameters():
    """Parse the arguments of experiments/figures.py."""
    p = add_attack_arguments(argparse.ArgumentParser(description="Mask figures"))
    p.add_argument("--figure", required=True, choices=["gradcam_masks", "learned_vs_gradcam"])
    p.add_argument("--n-show", type=int, default=None,
                   help="number of images (default 5 for gradcam_masks, 3 otherwise)")
    p.add_argument("--rho", type=float, default=0.25, help="sparsity of the side-by-side figure")
    return p.parse_args()
