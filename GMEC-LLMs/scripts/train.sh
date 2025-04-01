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

emonu="bartLmn_emonu_clip_3";
emo="bartLmn_emo_clip_test";
cau="bartLmn_cau_test";
ae_ac="bartLmn_aeac_test";
ae_ac_nu="bartLmn_aeacnu_test";
ae_pc="bartLmn_aepc_test";
ae_pc_nu="bartLmn_aepcnu_1";
pe_pc="bartLmn_pepc_clip_realtime_size_2_TAVtest";

# ver=$emonu
# tt=emotion_neu
M_TAV=TAV
# s_len=10000
PATIENCE=0.5
#  $pe_pc
# for ver in {$emonu,$emo,$cau,$ae_ac,$ae_ac_nu}
for ver in $pe_pc 
do
if [[ "$ver" == "$emo" ]];
then tt='emotion'
elif [[ "$ver" == "$emonu" ]];
then tt='emotion_neu'
elif [[ "$ver" == "$cau" ]];
then tt='cause'
# elif [[ "$ver" == "$ae_ac" ]];
# then tt='AnnoEmo_Annocause_pair'
# elif [[ "$ver" == "$ae_ac_nu" ]];
# then tt='AnnoEmo_Annocause_pair_neu'
# elif [[ "$ver" == "$ae_pc" ]];
# then tt='AnnoEmo_precause_pair'
# elif [[ "$ver" == "$ae_pc_nu" ]];
# then tt='AnnoEmo_precause_pair_neu'
elif [[ "$ver" == "$pe_pc" ]];
then tt='preEmo_precause_pair'
else
echo 'nothing to do'
fi

TASK=${TASK:-mec} # task name,
# TASK_TYPE=${TASK_TYPE:-$tt} # [emotion emotion_neu cause AnnoEmo_Annocause_pair preEmo_precause_pair_neu]
TASK_TYPE=$tt # [emotion emotion_neu cause AnnoEmo_Annocause_pair preEmo_precause_pair_neu]
GPU=${GPU:-0} # GPU id(s) you want to use, default '0'
# PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-None} # path to the pretrained model, default is the result from our experiments
# VERSION=${VERSION:-$ver} # version name, default 'training_for_$TASK'
VERSION=$ver # version name, default 'training_for_$TASK'
# ckpts_path=/outputs/ckpts/training_mecpeNeu/AnnoEmo_Annocause_pair_neu_epoch2.pkl
#  --ckpt_path $ckpts_path\
log_path=./ots/$VERSION.log
# ckpt_path=./outputs/ckpts/trainin_emonu_5e5_optim2_8/emotion_neu_epoch17.pkl
echo $VERSION $ver $tt $TASK_TYPE $log_path
# --pretrained_model $PRETRAINED_MODEL_PATH \
# run python script
# CUDA_VISIBLE_DEVICES=$GPU \
python -u main.py \
    --task $TASK --run_mode train \
    --cfg configs/train.yml \
    --version $VERSION \
    --task_type $TASK_TYPE \
    --m_tav $M_TAV \
    --patience $PATIENCE \
    --gpu $GPU --seed 99 --grad_accu 20 
    # 1>$log_path 2>&1
done

# bash scripts/train.sh --task mec --version mecpec_train_1 --gpu 0 