#!/bin/bash
#SBATCH --array=0-6
#SBATCH --account=def-scoulombe
#SBATCH --job-name=GIFStream
#SBATCH --output=nerf-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --gpus=a100

EXP_NAMES=(
  SfM_init
  SfM_init
  SfM_init
  SfM_init
  SfM_init
  SfM_init
  SfM_init
)

OFFSET_LAMBDA_LIST=(
  0
  0.05
  0.1
  0.2
  0.3
  0.4
  0.5
)

SCENE_LIST=(sear_steak flame_steak) # cook_spinach cut_roasted_beef)


ENTROPY_LAMBDA_LIST=(0.0005 0.001 0.002 0.004)
RATE=1 #carefull, changing rate changes rd tradeoff AND QUANTIZATION STEPS
IDX=$SLURM_ARRAY_TASK_ID
OFFSET_LAMBDA=${OFFSET_LAMBDA_LIST[$IDX]}
EXP_NAME0=${EXP_NAMES[$IDX]}/${OFFSET_LAMBDA}

echo "job run at $(date +"%T")"

# Load modules BEFORE creating/activating venv
module load cuda/12.9 python/3.11 opencv/4.12 scipy-stack/2023b ffmpeg colmap

# Create and activate venv
# Virtual environment in temporary folder
VENV_DIR=~/venvs/GIFStream4

#virtualenv -q --no-download $VENV_DIR || { echo "Failed to create venv"; exit 1; }
source $VENV_DIR/bin/activate
cp -r ~/wheelhouse $SLURM_TMPDIR/
# Install build & deps
pip install --no-index torch==2.9.0 torchvision==0.24.0 torchtext==0.6.0+computecanada torchaudio==2.9.0+computecanada
python -c "import torch; print('Torch version:', torch.__version__)"
#pip wheel ~/src_pkgs/pycolmap -w ~/wheelhouse --no-build-isolation --no-deps
#pip wheel ~/src_pkgs/fused-ssim -w ~/wheelhouse --no-build-isolation --no-deps
#pip wheel ~/src_pkgs/PLAS -w ~/wheelhouse --no-build-isolation --no-deps
echo "\n \n entering pip install requirements_server.txt"
#pip install . \
#  --no-build-isolation \
#  --no-index \
#  --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic \
#  --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic \
#  --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3 \
#  --find-links=$SLURM_TMPDIR/wheelhouse
#  
#
#echo "------ pip install -e . done"
#
#cd examples
#pip install -r requirements_server.txt \
#  --no-index \
#  --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic \
#  --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic \
#  --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3 \
#  --find-links=$SLURM_TMPDIR/wheelhouse 
#  
#echo "\n \n exiting pip install requirements_server.txt"
#cd ..
#
# only run once
#export PIP_NO_INDEX=1
#export PIP_DISABLE_PIP_VERSION_CHECK=1
#
#pip install third_party/gridencoder --no-build-isolation \
#  --no-index \
#  --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic \
#  --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic \
#  --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3 \
#  --find-links=$SLURM_TMPDIR/wheelhouse 

#check what version of pytorch is intalled and make sure it is the same as before 
python -c "import torch; print('Torch:', torch.__version__, '| CUDA:', torch.version.cuda, '| Available:', torch.cuda.is_available())"
pip list | grep torch

#install pre-built wheels 
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
export QT_QPA_PLATFORM=offscreen
#python ./dataset_process/n3d_video_process.py --root_dir /home/erikmam/projects/def-scoulomb/erikmam/Locally_consistent_SCAFFOLD-GS/Neur3D --extract_frames
echo "DATASET ALREADY PROCESSED"


# Data factor for training
DATA_FACTOR=2
# Number of frames per GOP (Group of Pictures)
GOP=60
# The index of the first frame to process
FIRST_FRAME=0
#where to save stuff
RESULT_DIR="results/Neur3D"
# Total number of frames to process
TOTAL_FRAME=300

SCENE_DIR="/home/erikmam/projects/def-scoulomb/erikmam/Locally_consistent_SCAFFOLD-GS/Neur3D"
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
  for ((GOP_ID=0; GOP_ID < 2 ; GOP_ID++)); #$(((TOTAL_FRAME + GOP - 1)/GOP)) ; GOP_ID++));
  do
    echo "Running $SCENE GOP $GOP_ID"
      # Calculate the starting frame for this GOP
    GOP_START_FRAME=$((FIRST_FRAME + GOP_ID * GOP ))
    EXP_NAME=${RESULT_DIR}/${EXP_NAME0}/${SCENE}/GOP_$GOP_ID/r$RATE
    # Calculate the maximum number of frames for this GOP
    MAX_GOP=$((TOTAL_FRAME + FIRST_FRAME - GOP_START_FRAME))
    if ((GOP_ID == 0)); then
      # If this is the first GOP, train from scratch
      CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer_GIFStream.py $TYPE --disable_viewer --data_factor $DATA_FACTOR \
          --data_dir $SCENE_DIR/$SCENE/ --result_dir ${EXP_NAME}/ \
          --eval_steps 7000 30000 --save_steps 7000 30000 \
          --compression_sim --rd_lambda ${ENTROPY_LAMBDA_LIST[RATE]} --entropy_model_opt --rate $RATE \
          --batch_size 1 --offset_lambda $OFFSET_LAMBDA --GOP_size $(( MAX_GOP < GOP ? MAX_GOP : GOP)) --knn --start_frame $GOP_START_FRAME

      # Run evaluation and rendering after training
      CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer_GIFStream.py $TYPE --disable_viewer --data_factor $DATA_FACTOR \
          --data_dir $SCENE_DIR/$SCENE/ --result_dir $EXP_NAME \
          --ckpt $EXP_NAME/ckpts/ckpt_29999_rank0.pt \
          --compression end2end  --rate $RATE \
          --GOP_size $(( MAX_GOP < GOP ? MAX_GOP : GOP)) --knn --start_frame $GOP_START_FRAME 
    else 
      # For subsequent GOPs, continue training from first checkpoint
      CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer_GIFStream.py $TYPE --disable_viewer --data_factor $DATA_FACTOR \
          --data_dir $SCENE_DIR/$SCENE/ --result_dir $EXP_NAME \
          --eval_steps 7000 30000 --save_steps 7000 30000 \
          --compression_sim --offset_lambda $OFFSET_LAMBDA --rd_lambda ${ENTROPY_LAMBDA_LIST[RATE]} --entropy_model_opt --rate $RATE \
          --batch_size 1 --GOP_size $(( MAX_GOP < GOP ? MAX_GOP : GOP)) --knn --start_frame $GOP_START_FRAME \
          --ckpt ${RESULT_DIR}/${EXP_NAME0}/${SCENE}/GOP_$((GOP_ID - 1))/r$RATE/ckpts/ckpt_6999_rank0.pt --continue_training
      # Run evaluation and rendering after training
      CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer_GIFStream.py $TYPE --disable_viewer --data_factor $DATA_FACTOR \
          --data_dir $SCENE_DIR/$SCENE/ --result_dir $EXP_NAME \
          --ckpt $EXP_NAME/ckpts/ckpt_29999_rank0.pt \
          --compression end2end  --rate $RATE \
          --GOP_size $(( MAX_GOP < GOP ? MAX_GOP : GOP)) --knn --start_frame $GOP_START_FRAME 
    fi
  done
done


## Resolve exp and sequence name
#
## Log file name
#LOG_FILE="nerf-${SLURM_ARRAY_JOB_ID}_${IDX}.out"
#
## Destination directory
#DEST_DIR="./logs/${EXP_NAME}"
#
## Create destination directory if needed
#mkdir -p "${DEST_DIR}"
#
#
### Move log file
##rm -rf ./latest_logs
##mkdir -p ./latest_logs
##touch ./latest_logs/exp_info.txt
##$EXP >> ./latest_logs/exp_info.txt
##$SEQ >> ./latest_logs/exp_info.txt
##cp ${LOG_FILE} ./latest_logs/
#
#mv "${LOG_FILE}" "${DEST_DIR}/"
#cp ./launch.slurm "${DEST_DIR}/launch.slurm_${IDX}"


