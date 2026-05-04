pip install requirements_wheels_DA3.txt --no-index
module load python/3.10  # or your version
module load gcc cmake    # needed for builds

python -m venv ~/wheel_env
source ~/wheel_env/bin/activate

# pip install -e . 
pip install --upgrade pip setuptools wheel build

export WHEELHOUSE=$SLURM_TMPDIR/wheelhouse
mkdir -p $WHEELHOUSE
pip wheel evo -w $WHEELHOUSE

module load cmake boost eigen
module load colmap

pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70 # for gaussian head

pip install -e ".[app]" # Gradio, python>=3.10
pip install -e ".[all]" # ALL