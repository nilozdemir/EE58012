# Graded Assignment 1 — Sparse Adversarial Attacks with Grad-CAM and Learnable Masks

EE 48012 / 58012, CIFAR-10, ResNet-18, PyTorch.

The clean model is the ResNet-18 of **Self-Study Assignment 2** (`lab-2/` of the
[course repository](https://github.com/SU-Intelligent-systems-Lab/EE48012-58012-Security-Privacy-Robustness-ML)),
which is the CIFAR-variant ResNet-18 of Self-Study Assignment 1 (3x3 stride-1 stem, no
max-pool, written from scratch, not `torchvision.models.resnet18`) with 10 classes and a
`Normalize` layer as its first layer. The model, data pipeline, training recipe and
checkpoint format are the same as in `lab-2`, so `checkpoints/resnet18_clean_seed0.pt`
trained there can be copied here and used unchanged.

## Layout

| Path | Job |
|---|---|
| `main.py` | entry point for training / testing the clean model |
| `parameters.py` | every command-line argument (training, attacks, figures) |
| `train.py`, `test.py`, `utils.py`, `data.py` | training loop, evaluation, checkpoints, CIFAR-10 |
| `models/resnet.py` | ResNet-18, CIFAR variant (from lab-2 / lab-1) |
| `models/__init__.py` | `build_model()`: `Normalize` layer followed by the ResNet; inputs are in [0, 1] |
| `attacks/gradcam.py` | Grad-CAM for any logit score (from lab-1, generalised) |
| `attacks/masks.py` | top-k binary masks from Grad-CAM or random scores |
| `attacks/sparse_pgd.py` | PGD on δ for a fixed mask, and the Eq. (1) constraint check |
| `attacks/learnable_mask.py` | Part B: joint learning of the mask and δ |
| `experiments/run_attacks.py` | ASR tables (Part A and Part B) |
| `experiments/figures.py` | figures of the report |

## Setup

```bash
pip install -r requirements.txt
```

CIFAR-10 is downloaded to `data/` on the first run. A GPU is used when available.

### Running on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nilozdemir/EE58012/blob/main/colab.ipynb)

`colab.ipynb` runs every command below on a Colab GPU: it clones this repository, keeps
`checkpoints/` and `results/` on Google Drive (`MyDrive/ee58012-ga1/`), trains the clean
model (skipped if the checkpoint is already there) and produces every figure and table.
Choose a GPU runtime and run all cells; after a disconnect, run all cells again and
finished steps are skipped.

## Commands for every figure and table

**1. Clean model** (Self-Study 2 recipe: SGD + momentum 0.9, lr 0.1, cosine schedule
with 1 warm-up epoch, 100 epochs, batch 128, weight decay 5e-4, crop + flip; reference
clean accuracy about 94-95%):

```bash
python main.py --optimizer momentum --scheduler cosine --lr 0.1 --epochs 100
python main.py --mode test
```

Writes `checkpoints/resnet18_clean_seed0.pt` and the training log
`results/resnet18_clean_seed0.csv` after every epoch; rerunning the same command after an
interruption resumes from the last finished epoch (and does nothing once all epochs are
done). Every command below uses this checkpoint by default
(`--checkpoint` to change it).

**2. Figure: Grad-CAM maps and masks for 5 test images (Part A, Task 1)**

```bash
python -m experiments.figures --figure gradcam_masks
```

→ `results/fig_gradcam_masks.png`

**3. Table: ASR of all masks (Part A, Task 2 and Part B, Task 2)**

```bash
python -m experiments.run_attacks --out results/asr
```

→ `results/asr.csv`, `results/asr.md`, `results/asr.tex`. The table has rows for
the Grad-CAM mask, the random mask, the learned mask with Grad-CAM initialisation and
the learned mask with random initialisation, and columns for untargeted/targeted ×
ρ = 25% / 50%. To run only Part A: `--methods gradcam random`. To run on a subset (for
example on a CPU): `--n-images 1000`.

Ablation of how Grad-CAM is used for the targeted mask (Part A, Task 3):

```bash
python -m experiments.run_attacks --modes targeted --methods gradcam --cam-score true   --out results/asr_cam_true
python -m experiments.run_attacks --modes targeted --methods gradcam --cam-score target --out results/asr_cam_target
python -m experiments.run_attacks --modes targeted --methods gradcam --cam-score margin --out results/asr_cam_margin
```

Control rows of Table 1 (Part A with 200 PGD steps, the same number of iterations as
the learned masks):

```bash
python -m experiments.run_attacks --methods gradcam random --steps 200 --out results/asr_steps200
```

**4. Figure: Grad-CAM mask vs learned masks for 3 images (Part B, Task 3)**

```bash
python -m experiments.figures --figure learned_vs_gradcam --rho 0.25
```

→ `results/fig_learned_vs_gradcam.png`

**5. Report**

```bash
cd report && latexmk -pdf report.tex
```

→ `report/report.pdf`; it reads the figures from `results/`. The numbers in its
tables are those of `results/asr.md`, `results/asr_cam_*.md` and
`results/asr_steps200.md`.


## Method summary

* **Threat model**: x' = x + m ⊙ δ, m ∈ {0,1}^{32×32} shared by the 3 channels,
  Σm ≤ k = ρ·1024, ‖δ‖∞ ≤ ε = 2/255 in [0, 1] pixel space, x' ∈ [0, 1]; normalisation is
  the model's first layer, applied after x' is formed. Every returned example is
  checked against these constraints (`check_constraints`), and the ASR is computed on a
  fresh evaluation of the returned binary-mask examples.
* **Grad-CAM layer**: `layer3`, output 256×8×8 (`layer4` is 512×4×4 with the CIFAR stem;
  selectable with `--cam-layer`).
* **Part A masks**: top-k pixels of the bilinearly upsampled Grad-CAM map (before the
  ReLU, so that every pixel has a rank). Untargeted: map of the true-class logit z_y.
  Targeted to t = (y+1) mod 10: map of the margin z_t − z_y. Random baseline: k
  uniformly random pixels.
* **δ optimisation (all masks)**: signed-gradient PGD, random start in [−ε, ε], 100
  steps, step 2.5·ε/100. Objective: cross-entropy of y (untargeted), minus
  cross-entropy of t (targeted). An image counts as a success if any iterate succeeds.
* **Part B learned mask**: per-pixel scores s, mask m = TopK(s, k) (binary at every
  step), straight-through gradient dL/ds := dL/dm = Σ_c δ_c ∂L/∂x'_c, Adam on s,
  δ updated on all pixels by PGD. After 100 joint steps the final binary mask TopK(s, k)
  is fixed and δ is finished with the same 100-step PGD as Part A. The scores start from
  the Grad-CAM map or from uniform noise (`learned_gradcam` / `learned_random`).
