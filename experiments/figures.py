"""Figures of the report.

python -m experiments.figures --figure gradcam_masks      --checkpoint <ckpt>
python -m experiments.figures --figure learned_vs_gradcam --checkpoint <ckpt>

gradcam_masks (Part A, Task 1): for 5 test images, the Grad-CAM maps of the
untargeted (true class) and targeted (margin) scores and the resulting masks
for rho = 25% and 50%.

learned_vs_gradcam (Part B, Task 3): for 3 images, the Grad-CAM mask and the
learned masks (Grad-CAM init, random init) side by side, for both attacks.
"""

import os

import matplotlib.pyplot as plt
import torch

from attacks.gradcam import forward_with_activations, get_layer, normalise01
from attacks.masks import gradcam_scores, num_pixels, target_class, topk_mask
from data import CLASSES
from experiments.common import attack_batch, correctly_classified, load_attack_model
from parameters import get_figure_parameters


def show(ax, img, title=None, overlay=None, cmap="jet"):
    """Draw an image (3, H, W) in [0, 1], optionally with a heatmap or mask overlay."""
    ax.imshow(img.permute(1, 2, 0).cpu().numpy())
    if overlay is not None:
        ax.imshow(overlay.cpu().numpy(), cmap=cmap, alpha=0.5, vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


def gradcam_masks_figure(model, x, y, args):
    """Save results/fig_gradcam_masks.png (Grad-CAM maps and masks of Part A)."""
    layer = get_layer(model, args.cam_layer)
    with torch.no_grad():
        _, A = forward_with_activations(model, layer, x[:1])
    size = "x".join(str(s) for s in A.shape[1:])
    t = target_class(y)
    rows = []
    for targeted in (False, True):
        scores = gradcam_scores(model, layer, x, y, targeted, args.cam_score)
        rows.append(("Grad-CAM\n" + ("$z_t - z_y$" if targeted else "$z_y$"),
                     normalise01(scores.clamp(min=0)), "jet"))
        for rho in (0.25, 0.5):
            mask = topk_mask(scores, num_pixels(rho))[:, 0]
            rows.append((f"{'targ.' if targeted else 'untarg.'} mask\n"
                         f"$\\rho$={int(100 * rho)}%", mask, "gray"))
    fig, axes = plt.subplots(len(rows) + 1, len(x), figsize=(1.5 * len(x), 1.45 * (len(rows) + 1)))
    for j in range(len(x)):
        show(axes[0, j], x[j], f"{CLASSES[y[j]]} → {CLASSES[t[j]]}")
        for i, (_, maps, cmap) in enumerate(rows, start=1):
            if cmap == "gray":
                axes[i, j].imshow(maps[j].cpu(), cmap="gray", vmin=0, vmax=1)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
            else:
                show(axes[i, j], x[j], overlay=maps[j])
    for ax, (label, _, _) in zip(axes[:, 0], [("image\n(true → target)", 0, 0)] + rows):
        ax.set_ylabel(label, fontsize=7)
    fig.suptitle(f"Grad-CAM layer: {args.cam_layer} ({size}); white = perturbed pixels",
                 fontsize=8)
    fig.tight_layout()
    path = "results/fig_gradcam_masks.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def learned_vs_gradcam_figure(model, x, y, args):
    """Save results/fig_learned_vs_gradcam.png (Grad-CAM vs learned masks, Part B)."""
    methods = [("gradcam", "Grad-CAM mask"), ("learned_gradcam", "learned\n(Grad-CAM init)"),
               ("learned_random", "learned\n(random init)")]
    cols = []
    for targeted in (False, True):
        for method, name in methods:
            gen = torch.Generator().manual_seed(args.seed)
            x_adv, mask, success = attack_batch(model, x, y, method, targeted, args.rho, args, gen)
            cols.append((f"{'targ.' if targeted else 'untarg.'} {name}", mask[:, 0], success))
    fig, axes = plt.subplots(len(x), len(cols) + 1, figsize=(1.45 * (len(cols) + 1), 1.6 * len(x)),
                             squeeze=False)
    t = target_class(y)
    for i in range(len(x)):
        show(axes[i, 0], x[i], f"{CLASSES[y[i]]} → {CLASSES[t[i]]}")
        for j, (name, mask, success) in enumerate(cols, start=1):
            axes[i, j].imshow(mask[i].cpu(), cmap="gray", vmin=0, vmax=1)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            title = (name + "\n" if i == 0 else "") + ("success" if success[i] else "fail")
            axes[i, j].set_title(title, fontsize=6)
    fig.suptitle(f"Binary masks, $\\rho$ = {int(100 * args.rho)}% "
                 f"(k = {num_pixels(args.rho)}), white = perturbed pixels", fontsize=8)
    fig.tight_layout()
    path = "results/fig_learned_vs_gradcam.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    """Build the requested figure on the first correctly classified test images."""
    args = get_figure_parameters()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_attack_model(args.checkpoint, device)
    n = args.n_show or (5 if args.figure == "gradcam_masks" else 3)
    x, y, idx, _ = correctly_classified(model, device, n)
    x, y = x.to(device), y.to(device)
    os.makedirs("results", exist_ok=True)
    build = gradcam_masks_figure if args.figure == "gradcam_masks" else learned_vs_gradcam_figure
    print("saved", build(model, x, y, args), "for test images", idx.tolist())


if __name__ == "__main__":
    main()
