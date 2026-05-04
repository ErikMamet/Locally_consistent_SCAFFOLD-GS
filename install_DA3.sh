# Load modules BEFORE creating/activating venv
export PYTHONNOUSERSITE=1
export PIP_USER=0

deactivate 2>/dev/null
module load python/3.10 cuda/12.9 gcc cmake

pip install --upgrade pip setuptools wheel

# sanity check (IMPORTANT)
which python
which pip

export PYTHONNOUSERSITE=1
export PIP_USER=0
rm -rf ~/wheel_env

python -m venv ~/wheel_env
source ~/wheel_env/bin/activate

module load cuda/12.9 python/3.10 opencv/4.12 scipy-stack/2023b ffmpeg
module load gcc cmake    # needed for builds
which python
python --version
python -c "import site; print(site.getsitepackages())"

pip install --upgrade pip setuptools wheel

pip install -r requirements_wheels_DA3.txt \
  --no-index \
  --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic \
  --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic \
  --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3



export WHEELHOUSE=./tmpdir/wheelhouse
mkdir -p $WHEELHOUSE
pip wheel evo -w $WHEELHOUSE

module load cmake boost eigen
module load colmap

pip install --no-index --find-links=$WHEELHOUSE evo gsplat

