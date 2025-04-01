# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Runner that handles the heuristics generations process
# ------------------------------------------------------------------------------ #

import os, sys
# sys.path.append(os.getcwd())

from datetime import datetime
import pickle, random, math, time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.utils.data as Data
import argparse
from pathlib import Path
import yaml
from copy import deepcopy
from tqdm import tqdm

from configs.task_cfgs import Cfgs
from .utils.load_data import CommonData, DataSet
from .model.mcan_for_finetune import MCANForFinetune
from .utils.optim import get_optim_for_finetune as get_optim
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    set_seed
)
from .utils.load_data import CommonData, DataSet
from .model.mecpe_for_train import MECPEForTrain
from .model.gpt2 import *
from .model.bart import *
from pdb import set_trace as stop
class Runner(object):
    def __init__(self, __C, *args, **kwargs):
        self.__C = __C
        self.net = None

    # heuristics generation
    @torch.no_grad()
    def eval(self, dataset,common_data):
        data_size = dataset.data_size

        if self.net is None:
            # Load parameters
            path = self.__C.CKPT_PATH
            print('Loading ckpt {}'.format(path))
            config = common_data.config
            config.video_embed_dim = self.__C.video_embed_dim
            config.audio_embed_dim = self.__C.audio_embed_dim

            net= BartForSequenceClassification.from_pretrained(self.__C.BERT_VERSION, config=config,ignore_mismatched_sizes=True)
            net.resize_token_embeddings(len(common_data.tokenizer))
            ckpt = torch.load(path, map_location='cpu')
            net.load_state_dict(ckpt['state_dict'], strict=False)
            net.cuda()
            if self.__C.N_GPU > 1:
                net = nn.DataParallel(net, device_ids=self.__C.GPU_IDS)
            print('Finish!')
            self.net = net
        else:
            net = self.net


        net.eval()

        
        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.__C.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=True
        )

        qid_idx = 0
        topk_results = {}
        latent_results = []
        k = self.__C.CANDIDATE_NUM
        
        # Idxs2Emotion={value:key for key,value in common_data.Emotion2Idxs.items()}
        # Idxs2YesNo ={value:key for key,value in common_data.YesNo2Idxs.items()}
        Emotion2index=common_data.Emotion2index
        Index2Emotion = {value:key for key,value in common_data.Emotion2index.items()}
        YesNo2index=common_data.YesNo2index
        Index2YesNo = {value:key for key,value in common_data.YesNo2index.items()}

        for step, input_tuple in enumerate(dataloader):
            print("\rEvaluation: [step %4d/%4d]" % (
                step,
                int(data_size / self.__C.EVAL_BATCH_SIZE),
            ), end='          ')

            input_tuple = [x.cuda() for x in input_tuple]
            pad_utt_prompt_ids,pad_utt_prompt_attention_masks,pad_token_type_ids,pad_gpt2_labels,pad_video_feats,pad_audio_feats,video_blank_ids,audio_blank_ids=input_tuple
            
            if self.__C.M_TAV == 'T':
                pad_audio_feats=None
                pad_video_feats=None
            elif self.__C.M_TAV == 'TA':
                pad_video_feats=None
            elif self.__C.M_TAV == 'TV':
                pad_audio_feats=None
                    
            # input_fordis={'input_ids':F.pad(pad_utt_prompt_ids,(0,2)),'attention_mask':F.pad(pad_utt_prompt_attention_masks,(0,2),"constant",1),'token_type_ids':F.pad(pad_token_type_ids,(0,2),"constant",1),'labels':pad_gpt2_labels,\
                            # 'video_features':pad_video_feats,'audio_features':pad_audio_feats,'video_idx':F.pad(video_blank_ids,(0,2),"constant",0),'audio_idx':F.pad(audio_blank_ids,(0,2),"constant",0)}
            input_info = {'input_ids':pad_utt_prompt_ids,'attention_mask':pad_utt_prompt_attention_masks,'token_type_ids':pad_token_type_ids,'labels':pad_gpt2_labels,\
                            'video_features':pad_video_feats,'audio_features':pad_audio_feats,'video_idx':video_blank_ids,'audio_idx':audio_blank_ids}
            input_info.pop('token_type_ids')
            with torch.no_grad():
                hidden,lm_logits= net(**input_info,output_answer_latent=True)
                logits =lm_logits
                predict_logit = logits
                # predict_logit =lm_logits[:,-3]
                hiddens= hidden.cpu() 
             
                pad_attmask_ids=pad_utt_prompt_attention_masks.cpu()
                # 1 only for question
                mask_padAqs = pad_attmask_ids
                hidden_len = mask_padAqs.sum(1) 
                new_hiddens = hiddens * (mask_padAqs.unsqueeze(2))
                new_hid_r = new_hiddens.sum(1).div(hidden_len.unsqueeze(1))
                new_hid = new_hid_r.unsqueeze(1)
                # 2 for context +question
                # new_hiddens = hiddens[:,:-2,:].unsqueeze(2)
                # new_hid = new_hiddens.mean(1)
                
                logits_lm = predict_logit.cpu()
               
                for i, (logit,label) in enumerate(zip(predict_logit.cpu(),pad_gpt2_labels.cpu())):
                   
                    truelas = label.numpy().tolist()[0]
                    ans_item=[]
                    qid = dataset.key_names[qid_idx]
                    qid_idx += 1

                    if self.__C.TASK_TYPE in ['emotion']:
                        # distribution_tensor = item[common_data.emotionids]
                        distribution = torch.softmax(logit,-1).numpy().tolist()
                        # true_label = Idxs2Emotion[truelas]
                        for idx, (key,value) in enumerate(Emotion2index.items()): #common_data.Emotion2Idxs.items()):
                            ans_item.append(
                                {
                                    'answer': key,
                                    'confidence': float(distribution[idx])
                                })
                       
                        topk_results[qid] = ans_item
                        latent_np = new_hid[i].numpy()
                        latent_results.append(latent_np)
                        # NEW_ANSWER_LATENTS_DIR = self.__C.ANSWER_LATENTS_DIR+f'_{self.__C.TASK_TYPE}'
                        np.save(
                            os.path.join(self.__C.ANSWER_LATENTS_DIR, f'{qid}.npy'),
                            latent_np ) 
                     
                    else:
                        # distribution_tensor = item[common_data.Y..esNoids]
                        distribution = torch.softmax(logit,-1).numpy().tolist()
                        true_label = Index2YesNo[truelas]
                        
                        for idx,(key,value) in enumerate(YesNo2index.items()):
                            ans_item.append(
                                {
                                    'answer': key,
                                    'confidence': float(distribution[value])
                                })
                        
                        topk_results[qid] = ans_item
                        latent_np = new_hid[i].numpy()
                        latent_results.append(latent_np)
                        # NEW_ANSWER_LATENTS_DIR = self.__C.ANSWER_LATENTS_DIR+f'_{self.__C.TASK_TYPE}'
                        np.save(
                            os.path.join(self.__C.ANSWER_LATENTS_DIR, f'{qid}.npy'),
                            latent_np
                        )
                        
      
        
        return topk_results, latent_results

    def run(self):
        # Set ckpts and log path
        ## where checkpoints will be saved
        Path(self.__C.CKPTS_DIR).mkdir(parents=True, exist_ok=True)
        ## where the result file of topk candidates will be saved
        Path(self.__C.CANDIDATE_FILE_PATH).parent.mkdir(parents=True, exist_ok=True)
        ## where answer latents will be saved
        Path(self.__C.ANSWER_LATENTS_DIR).mkdir(parents=True, exist_ok=True)

        # build dataset entities        
        common_data = CommonData(self.__C)

        train_set = DataSet(
            self.__C,
            common_data,
            self.__C.TRAIN_SPLITS,
            train_for_heuris=True
        )
        
        test_set = DataSet(
            self.__C,
            common_data,
            self.__C.TEST_SPLITS
        )
        
        # forward VQA model
        train_topk_results, train_latent_results = self.eval(train_set,common_data)
        test_topk_results, test_latent_results = self.eval(test_set,common_data)
        
        # save topk candidates
        topk_results = train_topk_results | test_topk_results
        # NEW_CANDIDATE_FILE_PATH = self.__C.CANDIDATE_FILE_PATH[:-5]+f'_{self.__C.TASK_TYPE}_'+ self.__C.CANDIDATE_FILE_PATH[-5:]
        json.dump(
            topk_results,
            open( self.__C.CANDIDATE_FILE_PATH, 'w'),
            indent=4
        )
      
        # search similar examples
        train_features = np.vstack(train_latent_results)
        train_features = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)

        test_features = np.vstack(test_latent_results)
        test_features = test_features / np.linalg.norm(test_features, axis=1, keepdims=True)

        # compute top-E similar examples for each testing input
        E = self.__C.EXAMPLE_NUM
        similar_qids = {}
        print(f'\ncompute top-{E} similar examples for each testing input')
   
        for i, test_qid in enumerate(tqdm(test_set.key_names)):
            # cosine similarity
            dists = np.dot(test_features[i], train_features.T)
            top_E = np.argsort(-dists)[:E]
            similar_qids[test_qid] = [train_set.key_names[j] for j in top_E]
        
        # save similar qids
        # NEW_EXAMPLE_FILE_PATH = self.__C.EXAMPLE_FILE_PATH[:-5]+f'_{self.__C.TASK_TYPE}_'+ self.__C.EXAMPLE_FILE_PATH[-5:]
        with open(self.__C.EXAMPLE_FILE_PATH, 'w') as f:
            json.dump(similar_qids, f)

def heuristics_login_args(parser):
    parser.add_argument('--task', dest='TASK', help='task name, e.g., ok, aok_val, aok_test', type=str, required=True)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', type=str, required=True)
    parser.add_argument('--version', dest='VERSION', help='version name', type=str, required=True)
    parser.add_argument('--ckpt_path', dest='CKPT_PATH', help='checkpoint path for heuristics', type=str, default=None)
    parser.add_argument('--gpu', dest='GPU', help='gpu id', type=str, default=None)
    parser.add_argument('--candidate_num', dest='CANDIDATE_NUM', help='topk candidates', type=int, default=None)
    parser.add_argument('--example_num', dest='EXAMPLE_NUM', help='number of most similar examples to be searched, default: 200', type=int, default=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for pretraining')
    heuristics_login_args(parser)
    args = parser.parse_args()
    __C = Cfgs(args)
    with open(args.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    __C.override_from_dict(yaml_dict)
    print(__C)
    runner = Runner(__C)
    runner.run()
