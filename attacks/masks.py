"""Binary sparsity masks: top-k of a per-pixel score (Grad-CAM, random, learned).

A mask has shape (N, 1, H, W) with values in {0, 1}; it is shared by the three
colour channels and has exactly k = round(rho * H * W) ones per image.
"""

import torch

from attacks.gradcam import class_score, grad_cam, margin_score, upsample


def num_pixels(rho, height=32, width=32):
    """Return k = round(rho * H * W), the number of pixels that may change."""
    return int(round(rho * height * width))


def target_class(y, num_classes=10):
    """Return the target class t = (y + 1) mod 10 of the targeted attack."""
    return (y + 1) % num_classes


def topk_mask(scores, k):
    """Return the binary mask of the k highest-scoring pixels of every image.

    Args:
        scores (Tensor): Per-pixel scores, shape (N, H, W).
        k (int): Number of pixels to keep.

    Returns:
        Tensor: Mask in {0, 1}, shape (N, 1, H, W), with exactly k ones per image.
    """
    n, h, w = scores.shape
    idx = scores.reshape(n, -1).topk(k, dim=1).indices
    mask = torch.zeros(n, h * w, device=scores.device)
    mask.scatter_(1, idx, 1.0)
    return mask.view(n, 1, h, w)


def resolve_cam_score(cam_score, targeted):
    """Return the class score Grad-CAM explains: "true", "target" or "margin".

    "auto" uses the true class for the untargeted attack and the margin
    z_t - z_y for the targeted one.
    """
    if cam_score != "auto":
        return cam_score
    return "margin" if targeted else "true"


def gradcam_scores(model, layer, x, y, targeted, cam_score="auto"):
    """Return the per-pixel Grad-CAM scores used to rank pixels for the mask.

    Untargeted: the map of the true class y, i.e. where the evidence for y
    lies; perturbing there removes it. Targeted: the map of the margin
    z_t - z_y, whose channel weights are grad z_t - grad z_y; it marks where
    the features both add evidence for t and take it from y. The map is used
    before the ReLU so that every pixel has a distinct rank even when fewer
    than k locations are positive (the ReLU does not change the order of the
    positive locations).

    Args:
        model (nn.Module): Network in eval mode, input in [0, 1].
        layer (nn.Module): Grad-CAM layer.
        x (Tensor): Clean images, shape (N, 3, H, W).
        y (Tensor): True labels, shape (N,).
        targeted (bool): Build the mask for the targeted attack.
        cam_score (str): "auto", "true", "target" or "margin".

    Returns:
        Tensor: Scores upsampled to the input size, shape (N, H, W).
    """
    t = target_class(y)
    kind = resolve_cam_score(cam_score, targeted)
    score_fn = {
        "true": class_score(y),
        "target": class_score(t),
        "margin": margin_score(t, y),
    }[kind]
    cam = grad_cam(model, layer, x, score_fn, relu=False)
    return upsample(cam, size=x.shape[-2:])


def random_scores(x, generator):
    """Return i.i.d. uniform scores (N, H, W); their top-k is a uniformly random mask."""
    n, _, h, w = x.shape
    return torch.rand(n, h, w, generator=generator).to(x.device)
