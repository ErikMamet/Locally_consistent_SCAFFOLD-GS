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
  SfM_init
  SfM_init
  SfM_init
  SfM_init
  SfM_init
)

OFFSET_LAMBDA_LIST=(
  0.01
  0.1
  0.2
  0.3
  0.4
  0.5
)

SEQUENCES=(
  coffee_martini
)

ENTROPY_LAMBDA_LIST=(0.0005 0.001 0.002 0.004)
RATE=1 #carefull, changing rate changes rd tradeoff AND QUANTIZATION STEPS
IDX=$SLURM_ARRAY_TASK_ID
EXP_NAME=${EXP_NAMES[$IDX]}
OFFSET_LAMBDA=${OFFSET_LAMBDA_LIST[$IDX]}
SEQUENCE=${SEQUENCES[0]}

echo "Running experiment $EXP_NAME on sequence $SEQUENCE"
echo "job run at $(date +"%T")"

# Load modules BEFORE creating/activating venv
module load cuda/13.2 python/3.10 opencv/4.12 scipy-stack/2023b ffmpeg

# Create and activate venv
# Virtual environment in temporary folder
VENV_DIR=$SLURM_TMPDIR/GIFStream
virtualenv -q --no-download $VENV_DIR || { echo "Failed to create venv"; exit 1; }
source $VENV_DIR/bin/activate
# Install build & deps
cd examples
pip install --no-index torch torchvision torchtext torchaudio
pip install --no-index --find-links=$SLURM_TMPDIR/wheelhouse -r requirements_server.txt
cd ..
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

# run once on login node
#cd ../third_party/MLEntropy
#mkdir build
#cd build
#cmake ../cpp -DCMAKE_BUILD_TYPE=Release
#make -j


##########################
###################           
#                             RUN 
#                                         ###################
#                                   #########################

python dataset_process/n3d_video_process.py --root_dir /home/erikmam/projects/def-scoulomb/erikmam/Locally_consistent_SCAFFOLD-GS/Neur3D
echo "DONE PROCESSING DATASET"



# Data factor for training
DATA_FACTOR=2
# Number of frames per GOP (Group of Pictures)
GOP=60
# The index of the first frame to process
FIRST_FRAME=0
# Total number of frames to process
TOTAL_FRAME=300
#INIT_TYPE
INIT_TYPE="SfM"
# Loop over each scene in the scene list
for SCENE in $SCENE_LIST;
do
  # Set TYPE based on the scene name
  if [ "$SCENE" = "coffee_martini" ]; then
      TYPE=neur3d_2
  elif [ "$SCENE" = "flame_salmon_1" ]; then
      TYPE=neur3d_1
  else
      TYPE=neur3d_0
  fi
  # Loop over each GOP segment
  for ((GOP_ID=1; GOP_ID < $(((TOTAL_FRAME + GOP - 1)/GOP)) ; GOP_ID++));
  do
    # If this is the first GOP, train from scratch
    CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer_GIFStream.py $TYPE --disable_viewer --data_factor $DATA_FACTOR \
        --render_traj_path $RENDER_TRAJ_PATH --data_dir $SCENE_DIR/$SCENE/ --result_dir $EXP_NAME \
        --eval_steps 7000 30000 --save_steps 7000 30000 \
        --compression_sim --rd_lambda ${ENTROPY_LAMBDA_LIST[RATE]} --entropy_model_opt --rate $RATE \
        --batch_size 1 --offset_lambda $OFFSET_LAMBDA --GOP_size $(( MAX_GOP < GOP ? MAX_GOP : GOP)) --knn --start_frame $GOP_START_FRAME

    # Run evaluation and rendering after training
    CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer_GIFStream.py $TYPE --disable_viewer --data_factor $DATA_FACTOR \
        --render_traj_path $RENDER_TRAJ_PATH --data_dir $SCENE_DIR/$SCENE/ --result_dir $EXP_NAME \
        --ckpt $EXP_NAME/ckpts/ckpt_29999_rank0.pt \
        --compression end2end  --rate $RATE \
        --GOP_size $(( MAX_GOP < GOP ? MAX_GOP : GOP)) --knn --start_frame $GOP_START_FRAME 
  done
done

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
