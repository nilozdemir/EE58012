"""Entry point for the clean model: reads the parameters, then trains and/or tests.

python main.py                  # train: writes checkpoints/resnet18_clean_seed0.pt
python main.py --mode test      # test accuracy of that checkpoint
"""

import torch

from data import build_loaders
from models import build_model
from parameters import get_parameters
from test import evaluate
from train import fit
from utils import checkpoint_path, load_checkpoint


def main():
    """Build the data and the model, then train and/or test."""
    args = get_parameters()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = build_loaders(args)
    model = build_model(args.arch).to(device)
    path = args.checkpoint
    if path is None and args.mode == "test":
        path = checkpoint_path(args.arch, args.seed)
    if path is not None:
        load_checkpoint(path, model, device=device)
    if args.mode == "train":
        fit(model, train_loader, val_loader, test_loader, args, device)
    loss, acc = evaluate(model, test_loader, device)
    print(f"test loss {loss:.4f}   test accuracy {acc:.2%}")


if __name__ == "__main__":
    main()
