#!/bin/bash
#SBATCH --array=0-0
#SBATCH --account=def-scoulombe
#SBATCH --job-name=GIFStream
#SBATCH --output=nerf-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:35:00
#SBATCH --gpus=a100

EXP_NAMES=(
  SfM_init
)

SEQUENCES=(
  coffee_martini
)

IDX=$SLURM_ARRAY_TASK_ID
EXP_NAME=${EXP_NAMES[$IDX]}
SEQUENCE=${SEQUENCES[$IDX]}


echo "Running experiment $EXP_NAME on sequence $SEQUENCE"
echo "job run at $(date +"%T")"

# Load modules BEFORE creating/activating venv
module load cuda/13.2 python/3.10 opencv/4.12 scipy-stack/2023b

# Create and activate venv
# Virtual environment in temporary folder
VENV_DIR=$SLURM_TMPDIR/GIFStream
virtualenv -q --no-download $VENV_DIR || { echo "Failed to create venv"; exit 1; }
source $VENV_DIR/bin/activate
# Install build & deps
cd examples
pip install --no-index --find-links=$SLURM_TMPDIR/wheelhouse -r requirements_server.txt

# only run once
export PIP_NO_INDEX=1
export PIP_FIND_LINKS=$HOME/wheelhouse
export PIP_DISABLE_PIP_VERSION_CHECK=1
pip wheel ~/src_pkgs/pycolmap -w ~/wheelhouse --no-build-isolation --no-deps
pip wheel ~/src_pkgs/fused-ssim -w ~/wheelhouse --no-build-isolation --no-deps
pip wheel ~/src_pkgs/PLAS -w ~/wheelhouse --no-build-isolation --no-deps

#install pre-built wheels 
cp -r ~/wheelhouse $SLURM_TMPDIR/
pip install -q --no-index --upgrade pip
echo "wheels have been built -- hurray !"

echo "wheels have been installed using pip -- hurray !"


cd ../third_party/MLEntropy
mkdir build
cd build
cmake ../cpp -DCMAKE_BUILD_TYPE=Release
make -j

### Rebuild and install local modules
#mkdir -p ./dist_SOGS
#CFLAGS="-w" CXXFLAGS="-w" NVCC_FLAGS="--disable-warnings" python -m build --no-isolation ../SOGS/SOGS/submodules/diff-gaussian-rasterization -w --outdir ./dist_SOGS
#CFLAGS="-w" CXXFLAGS="-w" NVCC_FLAGS="--disable-warnings" python -m build --no-isolation ../SOGS/SOGS/submodules/simple-knn -w --outdir ./dist_SOGS
#CFLAGS="-w" CXXFLAGS="-w" NVCC_FLAGS="--disable-warnings" python -m build --no-isolation ../SOGS/SOGS/submodules/PLAS -w --outdir  ./dist_SOGS


### Rendering and testing after sorting 
python main.py \
  --exp "$EXP_NAME" \
  --sequence "$SEQUENCE"\
  --test
  #--compress \
  #--decompress \
  ### move the logfile into the output folder 

# Get array index
IDX=${SLURM_ARRAY_TASK_ID}

# Resolve exp and sequence names
EXP=${EXP_NAMES[$IDX]}
SEQ=${SEQUENCES[$IDX]}

# Log file name
LOG_FILE="nerf-${SLURM_ARRAY_JOB_ID}_${IDX}.out"

# Destination directory
DEST_DIR="./logs/${EXP}/${SEQ}"

# Create destination directory if needed
mkdir -p "${DEST_DIR}"


## Move log file
#rm -rf ./latest_logs
#mkdir -p ./latest_logs
#touch ./latest_logs/exp_info.txt
#$EXP >> ./latest_logs/exp_info.txt
#$SEQ >> ./latest_logs/exp_info.txt
#cp ${LOG_FILE} ./latest_logs/

mv "${LOG_FILE}" "${DEST_DIR}/"
cp ./launch.slurm "${DEST_DIR}/"launch.slurm_{IDX}
