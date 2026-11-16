"""ASR tables of Part A and Part B.

python -m experiments.run_attacks --checkpoint checkpoints/<run>.pt

For every attack (untargeted, targeted), sparsity ratio (25%, 50%) and mask
(Grad-CAM, random, learned with Grad-CAM init, learned with random init),
attacks the correctly classified CIFAR-10 test images, checks Eq. (1) on every
adversarial example, and writes <out>.csv, <out>.md and <out>.tex. The tables are
rewritten after every configuration, and configurations already in <out>.csv (same
number of images) are skipped, so an interrupted run (e.g. a Colab disconnect)
continues where it stopped.
"""

import csv
import os
import time

import torch

from attacks.masks import num_pixels
from attacks.sparse_pgd import check_constraints
from experiments.common import (
    attack_batch, correctly_classified, load_attack_model, step_size,
)
from parameters import get_attack_parameters

LABELS = {
    "gradcam": "Grad-CAM mask (A)",
    "random": "Random mask (A)",
    "learned_gradcam": "Learned, Grad-CAM init (B)",
    "learned_random": "Learned, random init (B)",
}


def load_rows(path, n_images):
    """Return the rows of an earlier run with the same number of images, or [].

    Args:
        path (str): CSV written by write_tables.
        n_images (int): Number of attacked images of the current run.

    Returns:
        list[dict]: Rows with "rho" and "asr" as floats.
    """
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["rho"], r["asr"] = float(r["rho"]), float(r["asr"])
    return [r for r in rows if int(r["n_images"]) == n_images]


def write_tables(rows, out, header):
    """Write the results as CSV, a Markdown table and a LaTeX table.

    Args:
        rows (list[dict]): One dict per (mode, rho, method) with key "asr".
        out (str): Path prefix of the three files.
        header (str): Settings line printed above the Markdown table.
    """
    with open(out + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, rows[0])
        writer.writeheader()
        writer.writerows(rows)
    cols = [(m, r) for m in ("untargeted", "targeted") for r in (0.25, 0.5)
            if any(row["mode"] == m and row["rho"] == r for row in rows)]
    methods = [m for m in LABELS if any(row["method"] == m for row in rows)]
    asr = {(r["mode"], r["rho"], r["method"]): r["asr"] for r in rows}

    def cell(m, mode, rho):
        """Return the ASR of one configuration as a percentage string."""
        v = asr.get((mode, rho, m))
        return "-" if v is None else f"{100 * v:.1f}"

    names = [f"{mode} rho={int(100 * rho)}%" for mode, rho in cols]
    md = [header, "", "| Mask | " + " | ".join(names) + " |",
          "|---" * (len(cols) + 1) + "|"]
    md += ["| " + LABELS[m] + " | " + " | ".join(cell(m, *c) for c in cols) + " |"
           for m in methods]
    with open(out + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    tex = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
           "Mask & " + " & ".join(f"{mo[:4]}. $\\rho={int(100 * r)}\\%$" for mo, r in cols)
           + " \\\\", "\\midrule"]
    tex += [LABELS[m] + " & " + " & ".join(cell(m, *c) for c in cols) + " \\\\"
            for m in methods]
    tex += ["\\bottomrule", "\\end{tabular}"]
    with open(out + ".tex", "w") as f:
        f.write("\n".join(tex) + "\n")


def main():
    """Run every configuration and write the ASR tables."""
    args = get_attack_parameters()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_attack_model(args.checkpoint, device)
    x_all, y_all, _, clean_acc = correctly_classified(model, device, args.n_images)
    n = len(x_all)
    print(f"clean test accuracy {clean_acc:.2%}; attacking {n} correctly classified images")
    eps = args.eps / 255
    header = (f"Clean accuracy {clean_acc:.2%}; ASR (%) over {n} correctly classified "
              f"test images; eps={args.eps:g}/255, PGD steps={args.steps}, "
              f"step={255 * step_size(args):.3g}/255, Grad-CAM layer={args.cam_layer}, "
              f"cam score={args.cam_score}, mask steps={args.mask_steps}, "
              f"mask lr={args.mask_lr}, seed={args.seed}.")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = load_rows(args.out + ".csv", n)
    done = {(r["mode"], r["rho"], r["method"]) for r in rows}
    for mode in args.modes:
        targeted = mode == "targeted"
        for rho in args.rhos:
            k = num_pixels(rho)
            for method in args.methods:
                if (mode, rho, method) in done:
                    print(f"skip {mode} rho={rho} {method}: already in {args.out}.csv")
                    continue
                generator = torch.Generator().manual_seed(args.seed)
                start, n_success, n_mismatch = time.time(), 0, 0
                for i in range(0, n, args.batch_size):
                    x = x_all[i:i + args.batch_size].to(device)
                    y = y_all[i:i + args.batch_size].to(device)
                    x_adv, mask, success = attack_batch(
                        model, x, y, method, targeted, rho, args, generator
                    )
                    check_constraints(x, x_adv, mask, k, eps)
                    # The ASR is counted on a fresh evaluation of the returned
                    # binary-mask examples, not on the flags recorded during the attack.
                    with torch.no_grad():
                        pred = model(x_adv).argmax(1)
                    ok = pred == (y + 1) % 10 if targeted else pred != y
                    n_success += ok.sum().item()
                    n_mismatch += (ok != success).sum().item()
                row = {"mode": mode, "rho": rho, "k": k, "method": method,
                       "asr": n_success / n, "n_images": n, "flag_mismatch": n_mismatch,
                       "seconds": round(time.time() - start, 1)}
                rows.append(row)
                print(row, flush=True)
                write_tables(rows, args.out, header)
    print(open(args.out + ".md").read())


if __name__ == "__main__":
    main()
