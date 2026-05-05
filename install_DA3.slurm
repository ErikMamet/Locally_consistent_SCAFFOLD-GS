#!/bin/bash
#SBATCH --account=def-scoulombe
#SBATCH --job-name=DA3_create_dataset
#SBATCH --output=DA3-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --gpus=a100

######################## ON LOGIN NODE ####################
#'''
#python3 -m pip install --user huggingface_hub
#python -c "
#from huggingface_hub import snapshot_download
#snapshot_download(
#    repo_id='depth-anything/DA3NESTED-GIANT-LARGE',
#    local_dir='./shared/models/DA3NESTED-GIANT-LARGE'
#)
#"
#This creates 
#./shared/models/DA3NESTED-GIANT-LARGE/
#    config.json
#    model.safetensors
#    etc...
#'''

# -------------------------
# 1. Load system modules FIRST
# -------------------------
module purge
module load StdEnv/2023
module load gcc/12.3
module load python/3.11 cuda/12.9 cmake
module load opencv/4.12 scipy-stack/2023b ffmpeg
module load boost eigen colmap
# 2. Create clean venv
# -------------------------
rm -rf ~/wheel_env
python -m venv ~/wheel_env
source ~/wheel_env/bin/activate
export PYTHONNOUSERSITE=1
export PIP_USER=0
export PIP_NO_INDEX=1
export PIP_INDEX_URL=""
export PIP_DISABLE_PIP_VERSION_CHECK=1
# -------------------------

pip install --no-index torch==2.11.0 torchvision==0.26.0 torchtext==0.6.0+computecanada torchaudio==2.11.0+computecanada


which python
which pip
python --version

# -------------------------
# 3. Upgrade pip inside venv only
# -------------------------
pip install --no-index \
  --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic \
  pip "setuptools<82" wheel
pip wheel ~/src_pkgs/pycolmap -w ~/wheelhouse --no-build-isolation --no-deps
echo "\n 1 yayaYAYA \n"
pip install --no-index --find-links=~/wheelhouse pycolmap
echo "\n 2 yayaYAYA \n"
# -------------------------
# 5. Install pip deps (NO SciPy, NO system libs)
# -------------------------
cd DA3
pip install -e . \
  --no-index \
  --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic \
  --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic \
  --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3 
cd ..

echo "we're going into python"
export PYTHONPATH=$PYTHONPATH:$(pwd)
python ./DA3/test.py
