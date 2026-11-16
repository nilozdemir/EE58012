"""Grad-CAM on the CIFAR ResNet-18 for an arbitrary class score.

Follows lab-1/analysis/gradcam.py: L = ReLU(sum_k alpha_k A^k) with alpha_k the
spatial mean of d score / d A^k, where the score is a logit (or a difference of
logits), never a softmax probability.

Layer sizes of the CIFAR ResNet-18 for a 32x32 input:
layer1 64x32x32, layer2 128x16x16, layer3 256x8x8, layer4 512x4x4.
"""

import torch
import torch.nn.functional as F


def get_layer(net, name):
    """Return the sub-module `name` ("layer1" ... "layer4") of the ResNet.

    Args:
        net (nn.Module): nn.Sequential(Normalize, ResNet18) of models.build_model,
            or a bare ResNet18.
        name (str): Attribute name of the stage.

    Returns:
        nn.Module: The stage whose output Grad-CAM explains.
    """
    inner = net[1] if isinstance(net, torch.nn.Sequential) else net
    return getattr(inner, name)


def forward_with_activations(model, layer, x):
    """Run the model on x; return the logits and A, the output of layer.

    Args:
        model (nn.Module): Network to run.
        layer (nn.Module): Sub-module of model whose output is wanted.
        x (Tensor): Images, shape (N, 3, 32, 32).

    Returns:
        tuple[Tensor, Tensor]: Logits (N, K) and A (N, C, h, w), attached to the graph.
    """
    stored = {}

    def store_output(module, inputs, output):
        """Keep the output of the hooked module."""
        stored["A"] = output

    hook = layer.register_forward_hook(store_output)
    try:
        logits = model(x)
    finally:
        hook.remove()
    return logits, stored["A"]


def class_score(cls):
    """Return a score function: the logit of class cls[i] for image i.

    Args:
        cls (Tensor): Class per image, shape (N,).

    Returns:
        callable: Maps logits (N, K) to scores (N,).
    """
    return lambda logits: logits.gather(1, cls[:, None]).squeeze(1)


def margin_score(target, true):
    """Return a score function: logit of the target minus logit of the true class.

    Args:
        target (Tensor): Target class per image, shape (N,).
        true (Tensor): True class per image, shape (N,).

    Returns:
        callable: Maps logits (N, K) to scores (N,).
    """
    return lambda logits: (
        logits.gather(1, target[:, None]) - logits.gather(1, true[:, None])
    ).squeeze(1)


def grad_cam(model, layer, x, score_fn, relu=True):
    """Return the Grad-CAM maps of `layer` for the scores of score_fn, not upsampled.

    In eval mode the images of a batch are independent, so summing the scores
    gives every image the gradient of its own score.

    Args:
        model (nn.Module): Network in eval mode.
        layer (nn.Module): Stage whose output A is explained.
        x (Tensor): Images, shape (N, 3, 32, 32).
        score_fn (callable): Maps logits (N, K) to one score per image (N,).
        relu (bool): Apply the ReLU of Grad-CAM. Without it the map keeps the
            ordering of the negative locations too, which is used to rank pixels.

    Returns:
        Tensor: Maps sum_k alpha_k A^k (after ReLU if relu), shape (N, h, w).
    """
    with torch.enable_grad():
        # The input requires grad so that A is in the graph even when the
        # model's parameters are frozen.
        x = x.detach().requires_grad_(True)
        logits, A = forward_with_activations(model, layer, x)
        grads = torch.autograd.grad(score_fn(logits).sum(), A)[0]
    alpha = grads.mean(dim=(2, 3), keepdim=True)
    cam = (alpha * A).sum(dim=1).detach()
    return F.relu(cam) if relu else cam


def upsample(maps, size=(32, 32)):
    """Bilinearly upsample maps (N, h, w) to the input resolution (N, H, W)."""
    return F.interpolate(maps[:, None], size=size, mode="bilinear", align_corners=False)[:, 0]


def normalise01(maps):
    """Min-max normalise each map (N, H, W) to [0, 1] for display."""
    lo = maps.amin((1, 2), keepdim=True)
    hi = maps.amax((1, 2), keepdim=True)
    return (maps - lo) / (hi - lo + 1e-8)
