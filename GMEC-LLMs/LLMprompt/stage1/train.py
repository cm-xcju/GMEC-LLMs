# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Runner that handles the finetuning and evaluation process
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
import torch.utils.data as Data
import argparse
from pathlib import Path
from copy import deepcopy
import yaml

from configs.task_cfgs import Cfgs
from .utils.load_data import CommonData, DataSet
from .model.mecpe_for_train import MECPEForTrain
from .model.gpt2 import *
from .model.bart import *
from .utils.optim import get_optim_for_train2 as get_optim
from .utils.optim import get_optim_for_train3 as get_optim_train3
from pdb import set_trace as stop
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    set_seed
)
import torch.nn.functional as F
class Runner(object):
    def __init__(self, __C, evaluater):
        self.__C = __C
        self.evaluater = evaluater
        
    def train(self, train_set, valid_set=None,test_set=None,common_data=None):
        data_size = train_set.data_size
       
        # Define the MCAN model
        # net = MECPEForTrain(self.__C)
        # net = GPT2LMHeadModel(self.__C)
        config = common_data.config
        config.video_embed_dim = self.__C.video_embed_dim
        config.audio_embed_dim = self.__C.audio_embed_dim
        
        net= BartForSequenceClassification.from_pretrained(self.__C.BERT_VERSION, config=config,ignore_mismatched_sizes=True)
        net.resize_token_embeddings(len(common_data.tokenizer))
        net.cuda()
        
        ## load the pretrained model
        if self.__C.PRETRAINED_MODEL_PATH is not None:
            print(f'Loading pretrained model from {self.__C.PRETRAINED_MODEL_PATH}')
            ckpt = torch.load(self.__C.PRETRAINED_MODEL_PATH, map_location='cpu')
            net.load_state_dict(ckpt['state_dict'], strict=False)
            net.parameter_init()
            print('Finish loading.')

        # Define the optimizer
        if self.__C.RESUME:
            raise NotImplementedError('Resume training is not needed as the finetuning is fast')
        else:
            # the optim needed to be modified when use GPT2
            # training_steps_all = self.__C.MAX_EPOCH * len(train_set)
            # SUB_BATCH_SIZE = self.__C.BATCH_SIZE // self.__C.GRAD_ACCU_STEPS
            # training_steps =training_steps_all // SUB_BATCH_SIZE
            # optim = get_optim_train3(self.__C, net,training_steps)
            optim = get_optim(self.__C, net)
            start_epoch = 0

        # load to gpu
        net.cuda()
        # Define the multi-gpu training if needed
        if self.__C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__C.GPU_IDS)
        
        # Define the binary cross entropy loss
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
        epoch_loss = 0

        # Define multi-thread dataloader
        dataloader = Data.DataLoader(
            train_set,
            batch_size=self.__C.BATCH_SIZE,
            shuffle=True,
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=self.__C.PIN_MEM,
            drop_last=True
        )
        valid_f1_max = -1e-8
        # Training script
        for epoch in range(start_epoch, self.__C.MAX_EPOCH):
            net.train()
            # Save log information
            with open(self.__C.LOG_PATH, 'a+') as logfile:
                logfile.write(
                    f'nowTime: {datetime.now():%Y-%m-%d %H:%M:%S}\n'
                )

            time_start = time.time()
            # train_set.shuffle()
            # Iteration
          
            for step, input_tuple in enumerate(dataloader):
            # for i in range(0,data_size,self.__C.BATCH_SIZE):
                # input_tuple = train_set[i:i+self.__C.BATCH_SIZE]
                
                iteration_loss = 0
                optim.zero_grad()
                input_tuple = [x.cuda() for x in input_tuple]
                SUB_BATCH_SIZE = self.__C.BATCH_SIZE // self.__C.GRAD_ACCU_STEPS
                for accu_step in range(self.__C.GRAD_ACCU_STEPS):
                    sub_tuple = [x[accu_step * SUB_BATCH_SIZE:
                        (accu_step + 1) * SUB_BATCH_SIZE] for x in input_tuple]
                    pad_utt_prompt_ids,pad_utt_prompt_attention_masks,pad_token_type_ids,pad_gpt2_labels,pad_video_feats,pad_audio_feats,video_blank_ids,audio_blank_ids=sub_tuple
                    if self.__C.M_TAV == 'T':
                        pad_audio_feats=None
                        pad_video_feats=None
                    elif self.__C.M_TAV == 'TA':
                        pad_video_feats=None
                    elif self.__C.M_TAV == 'TV':
                        pad_audio_feats=None
                    input_info = {'input_ids':pad_utt_prompt_ids,'attention_mask':pad_utt_prompt_attention_masks,'token_type_ids':pad_token_type_ids,'labels':pad_gpt2_labels,\
                                  'video_features':pad_video_feats,'audio_features':pad_audio_feats,'video_idx':video_blank_ids,'audio_idx':audio_blank_ids}
                  
                    input_info.pop('token_type_ids')
                    # sub_ans_iter = sub_tuple[-1]
                    output = net(**input_info)
                
                    loss = output.loss
                    loss.backward()
                    loss_item = loss.item()
                    iteration_loss += loss_item
                    epoch_loss += loss_item # * self.__C.GRAD_ACCU_STEPS

                print("\r[version %s][epoch %2d][step %4d/%4d][Task %s][Mode %s] loss: %.4f, lr: %.2e" % (
                    self.__C.VERSION,
                    epoch + 1,
                    step,
                    int(data_size / self.__C.BATCH_SIZE),
                    self.__C.TASK,
                    self.__C.RUN_MODE,
                    iteration_loss / self.__C.BATCH_SIZE,
                    optim.current_lr(),
                ), end='          ')

                optim.step()
        
            time_end = time.time()
            print('Finished in {}s'.format(int(time_end - time_start)))

            # Logging
            with open(self.__C.LOG_PATH, 'a+') as logfile:
                logfile.write(f'epoch = {epoch + 1}  loss = {epoch_loss / data_size}\nlr = {optim.current_lr()}\n\n')
            
            optim.schedule_step(epoch)

            # Save checkpoint
            state = {
                'state_dict': net.state_dict() if self.__C.N_GPU == 1 \
                    else net.module.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'warmup_lr_scale': optim.warmup_lr_scale,
                'decay_lr_scale': optim.decay_lr_scale,
            }
           
            if epoch < int(self.__C.MAX_EPOCH*self.__C.PATIENCE):
                continue
            
            # Eval after every epoch
            if valid_set is not None:
                valid_scores = self.eval(
                    valid_set,
                    net,
                    eval_now=True,
                    common_data=common_data
                )
            
            valid_f1= valid_scores['f1']
            self.evaluater.clear()
            
            if test_set is not None: #and valid_f1 > valid_f1_max:
                test_scores = self.eval(
                    test_set,
                    net,
                    eval_now=True,
                    common_data=common_data
                )
                if valid_f1 > valid_f1_max:
                    valid_f1_max = valid_f1
                    torch.save(
                        state,
                        f'{self.__C.CKPTS_DIR}/{self.__C.TASK_TYPE}_epoch{epoch + 1}.pkl'
                        )
                RESULT_PATH = self.__C.RESULT_PATH
                RESULT_NEW_PATH = RESULT_PATH[:-5]+f'_{self.__C.TASK_TYPE}_{epoch + 1}_test'+RESULT_PATH[-5:]
                self.evaluater.save(RESULT_NEW_PATH)

                LOG_PATH = self.__C.LOG_PATH
                LOG_NEW_PATH = LOG_PATH[:-4]+f'_{self.__C.TASK_TYPE}_{epoch + 1}_test_'+LOG_PATH[-4:]
                with open(LOG_NEW_PATH, 'a+') as logfile:
                   print(str(test_scores) + '\n', file=logfile)

            epoch_loss = 0
            self.evaluater.clear()

    # Evaluation
    @torch.no_grad()
    def eval(self, dataset, net=None, eval_now=False,common_data=None):
        data_size = dataset.data_size
        
        self.evaluater.init_2(common_data=common_data,dataset=dataset,task_type=self.__C.TASK_TYPE)
        # if eval_now and self.evaluater is None:
        #     self.build_evaluator(dataset)
        only_for_test = False
        if net is None:
            only_for_test=True
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
                net = nn.DataParallel(net, device_ids=self.__C.GPU)
            print('Finish!')

        # if self.__C.TASK_TYPE=

        net.eval()
        
        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.__C.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=True
        )

        qid_idx = 0
        
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
            input_info = {'input_ids':pad_utt_prompt_ids,'attention_mask':pad_utt_prompt_attention_masks,'token_type_ids':pad_token_type_ids,'labels':pad_gpt2_labels,\
                            'video_features':pad_video_feats,'audio_features':pad_audio_feats,'video_idx':video_blank_ids,'audio_idx':audio_blank_ids}

            input_info.pop('token_type_ids')

            # input_fordis={'input_ids':F.pad(pad_utt_prompt_ids,(0,2)),'attention_mask':F.pad(pad_utt_prompt_attention_masks,(0,2),"constant",1),'token_type_ids':F.pad(pad_token_type_ids,(0,2),"constant",1),'labels':pad_gpt2_labels,\
            #                 'video_features':pad_video_feats,'audio_features':pad_audio_feats,'video_idx':F.pad(video_blank_ids,(0,2),"constant",0),'audio_idx':F.pad(audio_blank_ids,(0,2),"constant",0)}
            # input_fordis.pop('token_type_ids')
            with torch.no_grad():
                
                # for probalitation distribution.
                output_class = net(**input_info)
                logits = output_class.logits
                predict_logit = logits
                # predict_logit =logits[:,-3,:]
             
                # ids  = torch.argmax(predict_logit,-1)
                # test_res = common_data.tokenizer.batch_decode(ids, skip_special_tokens=False)
               
                # print(test_res)
                # input_ids = input_info['input_ids']
                # input_info.pop('input_ids')
                # attention_mask = input_info['attention_mask']
                # input_info.pop('attention_mask')
                # labels = input_info['labels']
                # input_info.pop('labels')

                # indexed_tokens=[]
                # output1, past =net(input_ids,past_key_values=None,return_dict=False)
                # token = torch.argmax(output1[..., -1, :])
                # indexed_tokens += [token.tolist()]
                # sequence = common_data.tokenizer.decode(indexed_tokens)
                #  return_dict_in_generate=True, top_k=50, top_p=0.95, do_sample=False, 
                # outputs2 = net.generate(input_ids, attention_mask=attention_mask, **input_info)
                # 
                
                # outputs = net.generate(input_ids, attention_mask=attention_mask,pad_token_id=common_data.tokenizer.pad_token_id, num_beams=1 ,max_new_tokens=1, **input_info)
                # batch_ans_ids = outputs[:,-1] # b * 1 
                # res1 = common_data.tokenizer.batch_decode(batch_ans_ids, skip_special_tokens=False) # b* 1
                # print(test_res,res1)
                
                # res2 = common_data.tokenizer.batch_decode(outputs2, skip_special_tokens=True)
                
                
                
                logits_lm = predict_logit.cpu()
                # batch_ans = batch_ans_ids.cpu().numpy().tolist()
                for i in range(len(logits_lm)):
                    vid=step*self.__C.EVAL_BATCH_SIZE+i
                    self.evaluater.add(vid, logits_lm[i])
                    # self.evaluater.add(vid, logits_lm[i],batch_ans[i])

                # if self.__C.TASK_TYPE == 'emo_cause_pair' or self.__C.TASK_TYPE == 'emo_cause_pair_neu':
                #     for vid in range(step*self.__C.EVAL_BATCH_SIZE,step*self.__C.EVAL_BATCH_SIZE+len(logits_lm)):
                #         result = self.evaluater.result_file[vid]
                #         prob_label= result['prob_label']
                #         if prob_label not in ['neutral','Yes']:
                #             prompt_ids, att_mask,token_type_id,prompt_video_feat,prompt_audio_feat,label,video_blank_id,audio_blank_id =dataset.get_Cause_prompt(vid,prob_label)
                #             input_cause = {'input_ids':prompt_ids,'attention_mask':att_mask,'token_type_ids':token_type_id,'labels':label,\
                #             'video_features':prompt_video_feat,'prompt_audio_feat':prompt_audio_feat,'video_idx':video_blank_id,'audio_idx':audio_blank_id}
                #             input_cause.pop('token_type_ids')

                #             output_class_cas = net(**input_cause)
                #             logits_cas = output_class_cas.logits
                #             predict_logit_cas =logits_cas[:,-1,:]

                #             input_ids = input_cause['input_ids']
                #             input_cause.pop('input_ids')
                #             attention_mask = input_cause['attention_mask']
                #             input_cause.pop('attention_mask')
                #             labels = input_cause['labels']
                #             input_cause.pop('labels')
                #             outputs_cas = net.generate(input_ids, attention_mask=attention_mask,pad_token_id=common_data.tokenizer.pad_token_id, num_beams=1 ,max_new_tokens=2, **input_info)
                #             output_ans_cas = outputs_cas[:,-2:]
                #             res1_cas = common_data.tokenizer.batch_decode(output_ans_cas, skip_special_tokens=False)
                #             res_cas=res1_cas[0]

                #             distribution_cas, prob_label_id_cas,prob_label_cas, gpt2_label_id_cas,true_label_cas = self.evaluater.get_logit(vid,logits_lm)
                #             res_id_cas, generate_label_cas = self.get_label(res_cas)

                #             self.evaluater.result_file[vid].update({
                #                 'Cause_prob':distribution_cas,
                #                 'Cause_prob_label_id':prob_label_id_cas,
                #                 'Cause_prob_label':prob_label_cas,
                #                 'Cause_label_generate_id':res_id_cas,
                #                 'Cause_label_generate':generate_label_cas,
                #                 'Cause_true_label_id':gpt2_label_id_cas,
                #                 'Cause_true_label':true_label_cas,

                #             })






        





            # pred_np = pred.cpu().numpy()
            # pred_argmax = np.argmax(pred_np, axis=1)
            
            # collect answers for every batch
            # for i in range(len(pred_argmax)):
            #     qid = dataset.qids[qid_idx]
            #     qid_idx += 1
            #     ans_id = int(pred_argmax[i])
            #     ans = dataset.ix_to_ans[ans_id]
            #     # log result to evaluater
            #     self.evaluater.add(qid, ans)
        
   
        # new_RESULT_PATH=
        # self.evaluater.save(self.__C.RESULT_PATH)
        # evaluate if eval_now is True
       
        if only_for_test:
            RESULT_PATH = self.__C.RESULT_PATH
            RESULT_NEW_PATH = RESULT_PATH[:-5]+f'_{self.__C.TASK_TYPE}_onlytest'+RESULT_PATH[-5:]
            self.evaluater.save(RESULT_NEW_PATH)

        scores=None
        if eval_now:
            with open(self.__C.LOG_PATH, 'a+') as logfile:
                scores = self.evaluater.evaluate(logfile)
        return scores

    # def build_evaluator(self, valid_set):
    #     if 'aok' in self.__C.TASK:
    #         from evaluation.aokvqa_evaluate import Evaluater
    #     elif 'ok' in self.__C.TASK:
    #         from evaluation.okvqa_evaluate import Evaluater
    #     else:
    #         raise ValueError('Unknown dataset')
    #     self.evaluater = Evaluater(
    #         valid_set.annotation_path,
    #         valid_set.question_path,
    #     )

    def run(self):
        # Set ckpts and log path
        ## where checkpoints will be saved
    
        Path(self.__C.CKPTS_DIR).mkdir(parents=True, exist_ok=True)
        ## where logs will be saved
        Path(self.__C.LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        ## where eval results will be saved
        Path(self.__C.RESULT_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(self.__C.LOG_PATH, 'w') as f:
            f.write(str(self.__C) + '\n')

        # build dataset entities        
        common_data = CommonData(self.__C)

        if self.__C.RUN_MODE == 'train':
            train_set = DataSet(
                self.__C, 
                common_data,
                self.__C.TRAIN_SPLITS
            )
            print("train_set_length:",len(train_set))
            valid_set = None
            if self.__C.EVAL_NOW:
                valid_set = DataSet(
                    self.__C,
                    common_data,
                    self.__C.DEV_SPLITS
                )
            print("valid_set_length:",len(valid_set))
            test_set = DataSet(
                self.__C,
                common_data,
                self.__C.TEST_SPLITS
            )
            print("test_set_length:",len(test_set))
            self.train(train_set, valid_set,test_set,common_data=common_data)
        elif self.__C.RUN_MODE == 'train_test':
            test_set = DataSet(
                self.__C,
                common_data,
                self.__C.TEST_SPLITS
            )
            self.eval(test_set, eval_now=self.__C.EVAL_NOW,common_data=common_data)
        else:
            raise ValueError('Invalid run mode')

def train_login_args(parser):
    parser.add_argument('--task', dest='TASK', help='task name, e.g. mec', type=str, required=True)
    parser.add_argument('--run_mode', dest='RUN_MODE', help='run mode', type=str, required=True)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', type=str, required=True)
    parser.add_argument('--version', dest='VERSION', help='version name', type=str, required=True)
    parser.add_argument('--resume', dest='RESUME', help='resume training', type=bool, default=False)
    parser.add_argument('--resume_version', dest='RESUME_VERSION', help='checkpoint version name', type=str, default='')
    parser.add_argument('--resume_epoch', dest='RESUME_EPOCH', help='checkpoint epoch', type=int, default=1)
    parser.add_argument('--resume_path', dest='RESUME_PATH', help='checkpoint path', type=str, default='')
    parser.add_argument('--ckpt_path', dest='CKPT_PATH', help='checkpoint path for test', type=str, default=None)
    parser.add_argument('--gpu', dest='GPU', help='gpu id', type=str, default=None)
    parser.add_argument('--seed', dest='SEED', help='random seed', type=int, default=None)
    parser.add_argument('--grad_accu', dest='GRAD_ACCU_STEPS', help='random seed', type=int, default=None)
    parser.add_argument('--pretrained_model', dest='PRETRAINED_MODEL_PATH', help='pretrained model path', type=str, default=None)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for training')
    train_login_args(parser)
    args = parser.parse_args()
    __C = Cfgs(args)

    with open(args.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    __C.override_from_dict(yaml_dict)
    print(__C)
    runner = Runner(__C)
    runner.run()
