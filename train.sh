#!/bin/bash
# Requested resources
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p p100 
#SBATCH --mem=48G
#SBATCH --cpus-per-task=10
# Wall time and job details
#SBATCH --time=24:00:00
#SBATCH --job-name=t2t-train
#SBATCH --output=t2t-train_%j.out 
# Emails me when job starts, ends or fails
#SBATCH --mail-user=subercui@gmail.com
#SBATCH --mail-type=END,FAIL
# Use this command to run the same job interactively
# salloc --mem=32G --cpus-per-task=10 --gres=gpu:1 --time=3:00:00

PROJECT_NAME="t2t"
ENV="t2tCLR"
WORK="/scratch/ssd001/home/haotian/Code/$PROJECT_NAME"
OUTPUT="$WORK/output"
​
# Path to the AllenNLP config
CONFIG_FILEPATH="$WORK/configs/contrastive.jsonnet"
# Directory to save model, vocabulary and training logs
SERIALIZED_DIR="$OUTPUT/tmp"
​
source /pkgs/anaconda3/bin/activate $ENV
cd $WORK

# clear the output folder
rm -r $SERIALIZED_DIR

# Run the job
allennlp train $CONFIG_FILEPATH \
	--serialization-dir $SERIALIZED_DIR \
	--include-package t2t