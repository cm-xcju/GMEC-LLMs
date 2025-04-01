#!/bin/bash
# This script is used to prompt GPT-3 to generate final answers.

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2;;
    --version)
      VERSION="$2"
      shift 2;;
    --examples_path)
      EXAMPLES_PATH="$2"
      shift 2;;
    --candidates_path)
      CANDIDATES_PATH="$2"
      shift 2;;
    --captions_path)
      CAPTIONS_PATH="$2"
      shift 2;;
    --openai_key)
      OPENAI_KEY="$2"
      shift 2;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done


# emonu="prompt_bartLmn_pepc_clip_realtime_3_TAV_50";
emo="prompt_bartLmn_emo_clip_realtime_3_TAV_e20i1_MinGPT4_2";
# cau="prompt_mec_cau";
# ae_ac="prompt_mec_aeac";
# ae_ac_nu="prompt_mec_aeacnu";
pe_pc="prompt_bartLmn_pepc_clip_realtime_3_TAV_e20i1_3";

# select the task
# for ver in {$emonu,$emo,$cau,$ae_ac,$ae_ac_nu}
for ver in $emo
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
elif [[ "$ver" == "$pe_pc" ]];
then tt='preEmo_precause_pair'
else
echo 'nothing to do'
fi

exa_path="outputs/results/heuristics_bartLmn_emo_clip_realtime_4_TAV/examples.json"
cad_path="outputs/results/heuristics_bartLmn_emo_clip_realtime_4_TAV/candidates.json"
cap_path="../../datasets/MELD-ECPE/Captions/13B-mingpt4.json"
utts_path="question_files/prompt_texts.json"
TASK_TYPE=$tt
TASK=${TASK:-mec} # task name, one of ['ok', 'aok_val', 'aok_test'], default 'ok'
# VERSION=${VERSION:-"prompt_mec_emotion"} # version name, default 'prompt_for_$TASK'
VERSION=$ver
EXAMPLES_PATH=$exa_path
CANDIDATES_PATH=$cad_path
CAPTIONS_PATH=$cap_path
UTTS_PATH=$utts_path
# EXAMPLES_PATH=${EXAMPLES_PATH:-"./outputs/results/heuristics_mec/examples.json"} # path to the examples, default is the result from our experiments
# CANDIDATES_PATH=${CANDIDATES_PATH:-"./outputs/results/heuristics_mec/candidates.json"} # path to the candidates, default is the result from our experiments
# CAPTIONS_PATH=${CAPTIONS_PATH:-"../../datasets/MELD-ECPE/Captions/Image_caption.json"} # path to the captions, default is the result from our experiments
OPENAI_KEY=${OPENAI_KEY:-"sk-oKN3vUmiJaSo9lKeSrAPBz9FSvxxxx"} # path to the captions

# echo $TASK_TYPE $EXAMPLES_PATH $CANDIDATES_PATH $CAPTIONS_PATH $UTTS_PATH

# CUDA_VISIBLE_DEVICES=$GPU \
python -u main.py \
    --task $TASK --run_mode prompt \
    --version $VERSION \
    --utts_path $UTTS_PATH \
    --task_type $TASK_TYPE  \
    --cfg configs/prompt.yml \
    --examples_path $EXAMPLES_PATH \
    --candidates_path $CANDIDATES_PATH \
    --captions_path $CAPTIONS_PATH \
    --openai_key $OPENAI_KEY
done