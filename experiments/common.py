"""Shared helpers of the experiment scripts: model loading, image selection, attacks."""

import torch

from attacks.gradcam import get_layer
from attacks.learnable_mask import learn_mask
from attacks.masks import gradcam_scores, num_pixels, random_scores, topk_mask
from attacks.sparse_pgd import sparse_pgd
from data import get_attack_set
from models import build_model
from utils import load_checkpoint


def load_attack_model(checkpoint, device):
    """Load the clean ResNet-18 (Normalize in front, input in [0, 1]), frozen, in eval mode.

    Args:
        checkpoint (str): checkpoints/resnet18_clean_seed0.pt, written by this
            project's train.fit or by lab-2 of the course repository.
        device (str): "cpu" or "cuda".

    Returns:
        nn.Sequential: The model.
    """
    model = build_model("resnet18")
    load_checkpoint(checkpoint, model, device=device)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def correctly_classified(model, device, n_images=None, batch_size=1000):
    """Return the test images the clean model classifies correctly.

    Args:
        model (nn.Module): Network in eval mode, input in [0, 1].
        device (str): "cpu" or "cuda".
        n_images (int, optional): Keep only the first n of them (test-set order).
        batch_size (int): Evaluation batch size.

    Returns:
        tuple[Tensor, Tensor, Tensor, float]: Images (N, 3, 32, 32) in [0, 1],
            labels (N,), their indices in the test set (N,), and the clean
            test accuracy.
    """
    x, y = get_attack_set()
    pred = torch.cat([model(x[i:i + batch_size].to(device)).argmax(1).cpu()
                      for i in range(0, len(x), batch_size)])
    correct = pred == y
    idx = torch.nonzero(correct).squeeze(1)
    if n_images is not None:
        idx = idx[:n_images]
    return x[idx], y[idx], idx, correct.float().mean().item()


def attack_batch(model, x, y, method, targeted, rho, args, generator):
    """Run one attack configuration on a batch.

    Args:
        model (nn.Module): Network in eval mode, input in [0, 1].
        x (Tensor): Clean images on the device, shape (N, 3, 32, 32).
        y (Tensor): True labels, shape (N,).
        method (str): "gradcam", "random", "learned_gradcam" or "learned_random".
        targeted (bool): Targeted or untargeted attack.
        rho (float): Sparsity ratio; k = round(rho * 32 * 32).
        args (argparse.Namespace): eps, steps, step_size, cam_layer, cam_score,
            mask_steps, mask_lr.
        generator (torch.Generator): RNG of random masks and random starts.

    Returns:
        tuple[Tensor, Tensor, Tensor]: Adversarial images, binary masks
            (N, 1, 32, 32) and success flags (N,).
    """
    k = num_pixels(rho, *x.shape[-2:])
    eps, step = args.eps / 255, step_size(args)
    layer = get_layer(model, args.cam_layer)
    if method.endswith("gradcam"):
        scores = gradcam_scores(model, layer, x, y, targeted, args.cam_score)
    else:
        scores = random_scores(x, generator)
    if method in ("gradcam", "random"):
        mask = topk_mask(scores, k)
        x_adv, success = sparse_pgd(model, x, y, mask, eps, args.steps, step,
                                    targeted, generator=generator)
        return x_adv, mask, success
    return learn_mask(model, x, y, k, eps, args.steps, step, targeted, scores,
                      args.mask_steps, args.mask_lr, generator=generator)


def step_size(args):
    """Return the PGD step in [0, 1] units: --step-size/255, or 2.5 * eps / steps."""
    if args.step_size is not None:
        return args.step_size / 255
    return 2.5 * (args.eps / 255) / args.steps
