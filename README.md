# envrioment
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install einops numpy matplotlib pandas tqdm
# conda in windows handle OpenMp runtime problem
# 在当前 conda 环境中永久设置(conda use intel NKL connected libraries, others may cause problem)
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE
conda activate your_env_name

## Repo note
Datasets and model weights are intentionally kept local and ignored by Git (`dataset/`, `1unet/data/mnist/`, `*.npy`, `*.pth`, `*.pt`) so the repo can push to GitHub without large-file errors.

