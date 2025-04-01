#!/bin/bash
# This script is used to generate heuristics from a finetuned model.

# Parse arguments
# w    --ckpt_path)
#       CKPT_PATH="$2"
#       shift 2;;
#     --candidate_num)
#       CANDIDATE_NUM="$2"
#       shift 2;;
#     --example_num)
#       EXAMPLE_NUM="$2"
#       shift 2;;
#     --version)
#       VERSION="$2"
#       shift 2;;
#     *)
#       echo "Unknown argument: $1"
#       exit 1;;
#   esac
# donehile [[ $# -gt 0 ]]; do
#   case "$1" in
#     --gpu)
#       GPU="$2"
#       shift 2;;
#     --task)
#       TASK="$2"
#       shift 2;;
# 

TASK=${TASK:-mec} # task name
GPU=${GPU:-0} # GPU id(s) you want to use, default '0'
CKPT_PATH=${CKPT_PATH:-"./outputs/ckpts/bartLmn_emo_clip_realtime_4_TAV/emotion_epoch13.pkl"} # path to the pretrained model, default is the result from our experiments
CANDIDATE_NUM=${CANDIDATE_NUM:-10} # number of candidates to be generated
EXAMPLE_NUM=${EXAMPLE_NUM:-100} # number of examples to be generated
TASK_TYPE=emotion #preEmo_precause_pair
VERSION=${VERSION:-"heuristics_bartLmn_emo_clip_realtime_4_TAV"} # version name, default 'heuristics1_for_$TASK'
s_len=100000
# CUDA_VISIBLE_DEVICES=$GPU \
python main.py \
    --task $TASK --run_mode heuristics \
    --version $VERSION \
    --task_type $TASK_TYPE \
    --sizelen $s_len \
    --cfg configs/train.yml \
    --ckpt_path $CKPT_PATH \
    --candidate_num $CANDIDATE_NUM \
    --example_num $EXAMPLE_NUM \
    --gpu $GPU