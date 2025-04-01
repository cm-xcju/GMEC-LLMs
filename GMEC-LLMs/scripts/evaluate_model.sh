#!/bin/bash
# This script is used to directly train the pretrained MECPEC model.

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2;;
    --task)
      TASK="$2"
      shift 2;;
    --pretrained_model)
      PRETRAINED_MODEL_PATH="$2"
      shift 2;;
    --version)
      VERSION="$2"
      shift 2;;
    --task_type)
      TASK_TYPE="$2"
      shift 2;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

emonu="train_s3_emonu__1";
emo="trainin_s3_emo_1";
cau="trainin_s3_cau_1";
ae_ac="trainin_s3_aeac_1";
ae_ac_nu="trainin_s3_aeacnu_1";
ae_pc="trainin_s3_aepc_1";
ae_pc_nu="trainin_s3_aepcnu_1";
pe_pc="bartLmn_pepc_clip_realtime_3_TAV";
M_TAV=TAV
# ver=$emonu
# tt=emotion_neu


# for ver in {$emonu,$emo,$cau,$ae_ac,$ae_ac_nu}
for ver in $ae_ac_nu
do
if [[ "$ver" == "$emo" ]];
then tt='emotion'
elif [[ "$ver" == "$emonu" ]];
then tt='emotion_neu'
elif [[ "$ver" == "$cau" ]];
then tt='cause'
elif [[ "$ver" == "$ae_ac" ]];
then tt='AnnoEmo_Annocause_pair'
elif [[ "$ver" == "$ae_ac_nu" ]];
then tt='AnnoEmo_Annocause_pair_neu'
elif [[ "$ver" == "$ae_ac_nu" ]];
then tt='AnnoEmo_precause_pair'
elif [[ "$ver" == "$ae_ac_nu" ]];
then tt='AnnoEmo_precause_pair_neu'
elif [[ "$ver" == "$ae_ac_nu" ]];
then tt='PreEmo_precause_pair'
else
echo 'nothing to do'
fi

TASK=${TASK:-mec} # task name, one of ['ok', 'aok_val', 'aok_test'], default 'ok'
# TASK_TYPE=${TASK_TYPE:-$tt} # [emotion emotion_neu cause AnnoEmo_Annocause_pair preEmo_precause_pair_neu]
TASK_TYPE=$tt # [emotion emotion_neu cause AnnoEmo_Annocause_pair preEmo_precause_pair_neu]
GPU=${GPU:-1} # GPU id(s) you want to use, default '0'
# PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-None} # path to the pretrained model, default is the result from our experiments
# VERSION=${VERSION:-$ver} # version name, default 'training_for_$TASK'
VERSION=$ver # version name, default 'training_for_$TASK'
# ckpts_path=/outputs/ckpts/training_mecpeNeu/AnnoEmo_Annocause_pair_neu_epoch2.pkl
#  --ckpt_path $ckpts_path\
log_path=./ots/$VERSION.log
ckpt_path=./outputs/ckpts/bartLmn_pepc_clip_realtime_3_TAV/preEmo_precause_pair_epoch3.pkl
echo $VERSION $ver $tt $TASK_TYPE $log_path
# --pretrained_model $PRETRAINED_MODEL_PATH \
# run python script
# CUDA_VISIBLE_DEVICES=$GPU \

python -u main.py \
    --task $TASK --run_mode train_test \
    --ckpt_path $ckpt_path \
    --cfg configs/train.yml \
    --version $VERSION \
    --task_type $TASK_TYPE \
    --gpu $GPU --seed 99 --grad_accu 10 \
    # 1>$log_path 2>&1
done

# bash scripts/train.sh --task mec --version mecpec_train_1 --gpu 0 