"""Part B: learn the binary mask m jointly with delta, separately for every image.

Parametrisation. Every pixel has a real score s_ij; the mask is its top-k
projection, m = TopK(s, k), which is binary with exactly k ones at every step.
TopK has no useful gradient, so s is trained with a straight-through
estimator: the backward pass treats m as if it were s, so

    dL/ds_ij := dL/dm_ij = sum_c delta_cij * dL/dx'_cij.

dL/dm_ij is the first-order change of the objective when pixel (i, j) is
switched on with its current perturbation (or kept on). delta is defined
on every pixel and is updated by PGD with the gradient at x' (not masked), so
a pixel that is switched on brings the perturbation it would have had.

Algorithm, per batch of images:
  1. s <- standardised init scores (Grad-CAM or random), delta <- U(-eps, eps).
  2. For mask_steps iterations:
        m = TopK(s, k);  x' = x + m * delta;  g = dL/dx'
        delta <- Proj(delta + alpha * sign(g))                  (all pixels)
        s     <- Adam step on s with gradient -sum_c delta_c * g_c   (ascent)
  3. Final binary mask m = TopK(s, k); delta is then finished with the same
     fixed-mask PGD as Part A (sparse_pgd), starting from the current delta.

Every iterate x + m * delta uses a binary top-k mask, so it is a valid sparse
adversarial example and an image counts as a success if any iterate of
step 2 or 3 succeeds. The ASR is computed only with binary masks.
"""

import torch

from attacks.masks import topk_mask
from attacks.sparse_pgd import (
    SuccessTracker, attack_objective, is_success, project, sparse_pgd,
)


def standardise(scores):
    """Return per-image standardised scores (zero mean, unit std), shape (N, H, W)."""
    mean = scores.mean((1, 2), keepdim=True)
    std = scores.std((1, 2), keepdim=True)
    return (scores - mean) / (std + 1e-8)


def learn_mask(model, x, y, k, eps, steps, step_size, targeted, init_scores,
               mask_steps=100, mask_lr=0.05, generator=None):
    """Learn a binary mask and delta for every image (see the module docstring).

    Args:
        model (nn.Module): Network in eval mode, input in [0, 1].
        x (Tensor): Clean images, shape (N, 3, H, W) in [0, 1].
        y (Tensor): True labels, shape (N,).
        k (int): Pixel budget.
        eps (float): L-inf bound in [0, 1] pixel units.
        steps (int): PGD steps of the final fixed-mask stage (same as Part A).
        step_size (float): PGD step for delta, in [0, 1] pixel units.
        targeted (bool): Targeted attack to (y + 1) mod 10, or untargeted.
        init_scores (Tensor): Initial pixel scores (N, H, W): Grad-CAM or random.
        mask_steps (int): Iterations of the joint mask and delta optimisation.
        mask_lr (float): Adam learning rate of the scores s.
        generator (torch.Generator, optional): RNG of the random start of delta.

    Returns:
        tuple[Tensor, Tensor, Tensor]: Adversarial images (N, 3, H, W), the final
            binary masks (N, 1, H, W) and a success flag per image (N,).
    """
    t = (y + 1) % 10
    s = standardise(init_scores.detach().clone()).requires_grad_(True)
    opt = torch.optim.Adam([s], lr=mask_lr)
    delta = (torch.rand(x.shape, generator=generator).to(x.device) * 2 - 1) * eps
    delta = project(delta, x, eps)
    mask = topk_mask(s.detach(), k)
    tracker = SuccessTracker(x, mask)

    for _ in range(mask_steps):
        mask = topk_mask(s.detach(), k)
        x_adv = (x + mask * delta).requires_grad_(True)
        logits = model(x_adv)
        tracker.update(x_adv, mask, is_success(logits, y, t, targeted))
        g = torch.autograd.grad(attack_objective(logits, y, t, targeted).sum(), x_adv)[0]
        delta = project(delta + step_size * g.sign(), x, eps)
        opt.zero_grad()
        s.grad = -(delta * g).sum(1)          # straight-through dL/dm, ascent on L
        opt.step()

    mask = topk_mask(s.detach(), k)
    x_fin, success_fin = sparse_pgd(
        model, x, y, mask, eps, steps, step_size, targeted, delta_init=delta
    )
    # Images that already succeeded in the joint stage keep that example and mask.
    fail = ~tracker.success
    tracker.update(x_fin, mask, success_fin & fail)
    x_adv, mask_out, success = tracker.finish(x_fin, mask)
    return x_adv, mask_out, success
