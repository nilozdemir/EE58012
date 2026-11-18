#!/bin/bash
# SLURM job that produces every figure and table of the report.
# Submit it from the repository root (the folder that contains main.py):
#     cd EE58012 && mkdir -p logs && sbatch job.sh
# Adjust the partition/time lines and the conda setup to your cluster.
# Resubmitting after a time-out resumes: training continues after its last
# epoch and finished ASR configurations are skipped.
#SBATCH --job-name=ga1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
if [ ! -f main.py ] || [ ! -d experiments ]; then
    echo "ERROR: run from the repository root (folder with main.py); current dir: $(pwd)" >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ee58012
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python main.py --optimizer momentum --scheduler cosine --lr 0.1 --epochs 100 --workers 4
python main.py --mode test
python -m experiments.figures --figure gradcam_masks
python -m experiments.run_attacks --batch-size 500 --out results/asr
for s in true target margin; do
    python -m experiments.run_attacks --batch-size 500 --modes targeted --methods gradcam \
        --cam-score "$s" --out "results/asr_cam_$s"
done
python -m experiments.figures --figure learned_vs_gradcam --rho 0.25
echo "done"
