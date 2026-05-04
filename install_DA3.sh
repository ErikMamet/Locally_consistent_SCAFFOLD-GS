# Load modules BEFORE creating/activating venv
module load cuda/12.9 python/3.10 opencv/4.12 scipy-stack/2023b ffmpeg
module load gcc cmake    # needed for builds


python -m venv ~/wheel_env
source ~/wheel_env/bin/activate

pip install -r ./requirements_wheels_DA3.txt --no-index

# pip install -e . 
pip install --upgrade pip setuptools wheel build


export WHEELHOUSE=./tmpdir/wheelhouse
mkdir -p $WHEELHOUSE
pip wheel evo -w $WHEELHOUSE

module load cmake boost eigen
module load colmap

pip install --no-index --find-links=$WHEELHOUSE evo gsplat

