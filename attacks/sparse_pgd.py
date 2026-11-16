"""PGD on delta for a fixed binary mask: x' = x + m * delta, ||delta||_inf <= eps.

Used for the Grad-CAM and random masks of Part A and to finish the learned
masks of Part B, so that all masks get the same optimisation of delta.
"""

import torch
import torch.nn.functional as F


def attack_objective(logits, y, t, targeted):
    """Return the per-image objective that the attack maximises.

    Untargeted: cross-entropy of the true class y. Targeted: minus the
    cross-entropy of the target t.

    Args:
        logits (Tensor): Shape (N, K).
        y (Tensor): True labels, shape (N,).
        t (Tensor): Target labels, shape (N,).
        targeted (bool): Which attack.

    Returns:
        Tensor: Objective per image, shape (N,).
    """
    if targeted:
        return -F.cross_entropy(logits, t, reduction="none")
    return F.cross_entropy(logits, y, reduction="none")


def is_success(logits, y, t, targeted):
    """Return a bool per image: argmax != y (untargeted) or argmax == t (targeted)."""
    pred = logits.argmax(1)
    return pred == t if targeted else pred != y


def project(delta, x, eps):
    """Clip delta to [-eps, eps] and so that x + delta stays in [0, 1]."""
    delta = delta.clamp(-eps, eps)
    return (x + delta).clamp(0, 1) - x


class SuccessTracker:
    """Keep, per image, the first adversarial example that succeeded."""

    def __init__(self, x, mask):
        """Start with no success; x (N, 3, H, W), mask (N, 1, H, W)."""
        self.success = torch.zeros(len(x), dtype=torch.bool, device=x.device)
        self.x_adv = x.clone()
        self.mask = mask.clone()

    def update(self, x_adv, mask, succ):
        """Record x_adv and mask for the images that succeed for the first time."""
        new = succ & ~self.success
        self.x_adv[new] = x_adv.detach()[new]
        self.mask[new] = mask.detach()[new]
        self.success |= succ

    def finish(self, x_adv, mask):
        """Fill the images that never succeeded with the last iterate; return (x_adv, mask, success)."""
        fail = ~self.success
        self.x_adv[fail] = x_adv.detach()[fail]
        self.mask[fail] = mask.detach()[fail]
        return self.x_adv, self.mask, self.success


def sparse_pgd(model, x, y, mask, eps, steps, step_size, targeted,
               delta_init=None, generator=None):
    """Optimise delta with signed-gradient PGD for a fixed binary mask.

    Every iterate x + m * delta is a valid sparse adversarial example, so an
    image counts as a success if any iterate is classified as wanted.

    Args:
        model (nn.Module): Network in eval mode, input in [0, 1].
        x (Tensor): Clean images, shape (N, 3, H, W) in [0, 1].
        y (Tensor): True labels, shape (N,).
        mask (Tensor): Binary mask, shape (N, 1, H, W).
        eps (float): L-inf bound in [0, 1] pixel units.
        steps (int): Number of PGD steps.
        step_size (float): PGD step in [0, 1] pixel units.
        targeted (bool): Targeted attack to (y + 1) mod 10, or untargeted.
        delta_init (Tensor, optional): Starting delta (N, 3, H, W); default
            uniform in [-eps, eps] (random start).
        generator (torch.Generator, optional): RNG of the random start.

    Returns:
        tuple[Tensor, Tensor]: Adversarial images (N, 3, H, W) and a success
            flag per image (N,).
    """
    t = (y + 1) % 10
    if delta_init is None:
        delta_init = (torch.rand(x.shape, generator=generator).to(x.device) * 2 - 1) * eps
    delta = project(delta_init, x, eps)
    tracker = SuccessTracker(x, mask)
    for _ in range(steps):
        delta.requires_grad_(True)
        x_adv = x + mask * delta
        logits = model(x_adv)
        tracker.update(x_adv, mask, is_success(logits, y, t, targeted))
        grad = torch.autograd.grad(attack_objective(logits, y, t, targeted).sum(), delta)[0]
        delta = project(delta.detach() + step_size * grad.sign(), x, eps)
    x_adv = x + mask * delta
    with torch.no_grad():
        tracker.update(x_adv, mask, is_success(model(x_adv), y, t, targeted))
    x_adv, _, success = tracker.finish(x_adv, mask)
    return x_adv, success


def check_constraints(x, x_adv, mask, k, eps, tol=1e-5):
    """Assert that x_adv is a valid sparse example: Eq. (1) of the assignment.

    Checks x_adv in [0, 1], ||x_adv - x||_inf <= eps, a binary mask with at
    most k ones per image, and no change outside the mask.

    Args:
        x (Tensor): Clean images, shape (N, 3, H, W).
        x_adv (Tensor): Adversarial images, same shape.
        mask (Tensor): Binary masks, shape (N, 1, H, W).
        k (int): Pixel budget.
        eps (float): L-inf bound.
        tol (float): Floating-point tolerance.
    """
    diff = (x_adv - x).abs()
    assert x_adv.min() >= -tol and x_adv.max() <= 1 + tol, "x_adv outside [0, 1]"
    assert diff.max() <= eps + tol, f"L-inf {diff.max().item():.6f} > eps {eps:.6f}"
    assert ((mask == 0) | (mask == 1)).all(), "mask not binary"
    assert (mask.flatten(1).sum(1) <= k).all(), "mask has more than k pixels"
    changed = diff.amax(1, keepdim=True) > tol
    assert not (changed & (mask == 0)).any(), "pixel changed outside the mask"
