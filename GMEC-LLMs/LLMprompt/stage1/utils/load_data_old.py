# --------------------------------------------------------------------------------- #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Data loading and preprocessing. Note that for the sake of simplicity,
#              the code only supports the following datasets for now:
#              * VQA 2.0
#              * OK-VQA
#              * A-OKVQA
#              Transferring to other datasets is easy. You may need to modify a few 
#              line of code in this file.
# --------------------------------------------------------------------------------- #

import numpy as np
import glob, json, pickle, random
import torch
import torch.utils.data as Data
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    set_seed
)
from evaluation.ans_punct import prep_ans
# from .transforms import _transform
from statistics import *
from pdb import set_trace as stop
import random
from typing import Optional, Tuple, Union

def soft_target(answers, ans_to_ix, preprocess=True):
    ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
    for ans in answers:
        if preprocess:
            ans = prep_ans(ans)
        if ans in ans_to_ix:
            ans_score[ans_to_ix[ans]] = min(1.0, ans_score[ans_to_ix[ans]] + 0.3)
    return ans_score


class CommonData:
    """
    load common data for all dataset objects:
    * imgid_to_path
    * bert tokenizer
    * ans_to_ix, ix_to_ans
    """
    def __init__(self, __C) -> None:
        print('Loading common data...')
        self.__C = __C
        # load imgid_to_path
        self.img_feat_path=__C.IMAGE_FEATURE_DIR['feature']
        self.video_id_map_path = __C.IMAGE_FEATURE_DIR['v2id']
        # load the audio
        self.audio_feat_path=__C.AUDIO_FEATURE_DIR['feature']
        # load the captions 
     

        # load bert tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(__C.BERT_VERSION,pad_token='<pad>')
        self.tokenizer = AutoTokenizer.from_pretrained(__C.BERT_VERSION)
        self.tokenizer.padding_side="left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_tokens(["<video>"], special_tokens=True)
        self.tokenizer.add_tokens(["<audio>"], special_tokens=True)
        self.tokenizer.add_tokens(["<text>"], special_tokens=True)
        self.tokenizer.add_special_tokens({"sep_token": "<sep>"})
        self.tokenizer.add_tokens(["<videoblankpos>"], special_tokens=True)
        self.tokenizer.add_tokens(["<audioblankpos>"], special_tokens=True)
        # self.tokenizer.add_tokens(["[mask]"], special_tokens=True)
        # self.tokenizer.add_tokens(["<pad>"], special_tokens=True)
        # self.tokenizer.add_tokens("anger")
        # self.tokenizer.add_tokens("surprise")
        # self.tokenizer.add_tokens("disgust")
        # self.tokenizer.add_tokens("neutral")
        # self.tokenizer.add_tokens("sadness")
        # self.tokenizer.add_tokens("joy")
        # self.tokenizer.add_tokens("fear")
        self.token_size = self.tokenizer.vocab_size
        print(f'== BertTokenizer loaded, vocab size: {self.token_size}')
        # load text
        self.text_path = __C.TEXT_DIR
        self.config = AutoConfig.from_pretrained(__C.BERT_VERSION)


    
        print('Common data process is done.\n')
        

class DataSet(Data.Dataset):
    def __init__(self, __C, common_data, split_name_list):
        self.__C = __C
        print(f'Loading dataset for {self.__C.TASK}|{self.__C.RUN_MODE}({split_name_list})')
        self.split_name_list = split_name_list
      
   
        # load all attributes from common data
        self.img_feat_path = common_data.img_feat_path
        self.tokenizer = common_data.tokenizer
        self.token_size = common_data.token_size
        self.video_id_map_path = common_data.video_id_map_path
        self.audio_feat_path = common_data.audio_feat_path
        self.text_path = common_data.text_path[split_name_list[0]]
        # self.caption_path = common_data.caption_path
        
        # Loading the img and audio
        video_idx, video_embedding, audio_embedding = self.load_embedding_from_npy(self.video_id_map_path, self.img_feat_path, self.audio_feat_path)
        self.video_idx= video_idx
        self.video_embedding= video_embedding
        self.audio_embedding= audio_embedding
    
        self.dialogues = self.load_text(self.text_path)
        # self.list_of_dialogues = [(key,value) for key,value in  self.dialogues.items()]
        # word_idx_rev, word_idx, spe_idx_rev, spe_idx, word_embedding, _ = self.load_w2v(__C.embedding_dim, __C.embedding_dim_pos, self.text_path['all'], self.ecf_path)
        
       
        self.max_dia_len = max([len(dia) for key,dia in self.dialogues.items()])
        self.max_utt_len = max([len(utt['Utterance']) for key,dia in self.dialogues.items()for utt in dia  ])
        self.mean_utt_len = mean([len(utt['Utterance']) for key,dia in self.dialogues.items()for utt in dia  ])
        print(f'max_dia_len == {self.max_dia_len}\n')
        print(f'max_utt_len == {self.max_utt_len}\n')
        print(f'mean_utt_len == {self.mean_utt_len}\n')
        
        self.renew_the_dataset()
        # self.max_prompt_len= max_token_len
        # self.max_av_len = 26

        # if self.split_name_list[0]=='train':
        #     self.utt_prompt_ids,\
        #     self.utt_prompt_attention_masks,\
        #     self.token_type_ids,\
        #     self.video_feats,\
        #     self.audio_feats,\
        #     self.gpt2_labels,\
        #     self.utt_prompts,\
        #     self.key_names,\
        #     self.Emotion_names,\
        #     self.Cause_names,\
        #     self.question_datas,\
        #     self.question_data_ids =self.renew_the_dataset()
        # else 

        self.data_size = len(self.utt_prompt_ids)
        print(f'== data size: {self.data_size}\n')
        self.pad_data()

        # self.small_sample_for_test(1468)
    def small_sample_for_test(self,sample_size=None):
        self.pad_utt_prompt_ids = self.pad_utt_prompt_ids[:sample_size]
        self.pad_utt_prompt_attention_masks=self.pad_utt_prompt_attention_masks[:sample_size]
        self.pad_token_type_ids = self.pad_token_type_ids[:sample_size]
        self.pad_video_feats=self.pad_video_feats[:sample_size]
        self.pad_audio_feats=self.pad_audio_feats[:sample_size]
        self.pad_gpt2_labels = self.pad_gpt2_labels[:sample_size]

        self.utt_prompt_ids = self.utt_prompt_ids[:sample_size]
        self.utt_prompt_attention_masks=self.utt_prompt_attention_masks[:sample_size]
        self.token_type_ids = self.token_type_ids[:sample_size]
        self.video_feats=self.video_feats[:sample_size]
        self.audio_feats=self.audio_feats[:sample_size]
        self.gpt2_labels = self.gpt2_labels[:sample_size]

        self.utt_prompts = self.utt_prompts[:sample_size]
        self.key_names=self.key_names[:sample_size]
        self.Emotion_names=self.Emotion_names[:sample_size]
        self.Cause_names=self.Cause_names[:sample_size]
        self.question_datas = self.question_datas[:sample_size]
        self.question_data_ids = self.question_data_ids[:sample_size]

        self.data_size = len(self.pad_utt_prompt_ids)
        print(f'== data size: {self.data_size}\n')
        
    def renew_the_dataset(self,):
        __C = self.__C
       
        self.utt_prompt_ids = []
        self.utt_prompt_attention_masks=[]
        self.token_type_ids = []
        self.video_feats=[]
        self.audio_feats=[]
        self.gpt2_labels = []
        

        self.utt_prompts = []
        self.key_names=[]
        self.Emotion_names = []
        self.Cause_names = []
        self.question_datas = []
        self.question_data_ids = []
        self.max_emo_cause_pair_span = 0
        self.min_emo_cause_pair_span = 0
        self.cause_after_emo = 0
        
        # self.cause_question=f'. if not neutral, which utterance is the cause of this emotion ? <|endoftext|>'
        
        for dia_key,dialogue in self.dialogues.items():
       
            # text
            # video (13620, 4096)
            # audio (13620, (13620, 6373))
            speaker_ids, text_ids, video_feats_all, audio_feats_all = [],[],[],[]
            prompt_utt_texts = []
            # sample about "<1-A> <T> </T> <A></A> <V></V>" which utteance is the cause of the this emotion"
            for utt in dialogue:
               
                text= utt['Utterance']
                Utterance_ID= utt['Utterance_ID']
                Speaker=utt['Speaker']
                Emotion = utt['Emotion']
                Utterance = utt['Utterance']
                # txt_id = self.bert_tokenize(text, __C.MAX_TOKEN)
                # sp_id = self.bert_tokenize(Speaker, __C.MAX_SPEAKER_TOKEN)
                utt_key = 'dia{}utt{}'.format(dia_key,Utterance_ID)
                pos = self.video_idx[utt_key]
                video_feat = self.video_embedding[pos]
                audio_feat = self.audio_embedding[pos]

                # Emotion_idx = self.tokenizer.encode(Emotion)
            
                # assert len(Emotion_idx)==1
                Cause_idxs = []
                Causes_txt=''
                question_data= {'question':[f'What is the emotion of utterance {Utterance_ID},Speaker {Speaker} ? <|endoftext|>'], 'answer':[f'{Emotion}<|endoftext|>']}                   
                
                if 'expanded emotion cause evidence' in utt.keys():
                   
                    Causes= utt['expanded emotion cause evidence']
                    Causes.sort(reverse = True)
                    max_cause_emo_span = max([eval(Utterance_ID) - ca for ca in Causes])
                    min_cause_emo_span = min([eval(Utterance_ID) - ca for ca in Causes])
                    if min_cause_emo_span <0:
                        self.cause_after_emo +=1
                    if max_cause_emo_span > self.max_emo_cause_pair_span:
                        self.max_emo_cause_pair_span =max_cause_emo_span
                    if min_cause_emo_span < self.min_emo_cause_pair_span:
                        self.min_emo_cause_pair_span =min_cause_emo_span
                  
                    Causes_txt = ','.join([str(i) for i in Causes])
                    question_data['question'].append(f'. if not neutral, which utterance is the cause of this emotion ? <|endoftext|>')
                    question_data['answer'].append(f'{Causes_txt}<|endoftext|>' )
                    # cause_one = Causes[0]
                    # for cas in Causes:
                    #     cas_id = self.tokenizer.encode(str(cas))
                    #     assert len(cas_id) ==1
                    #     Cause_idxs.append(cas_id[0])


                
                video_feats_all.append(video_feat)
                audio_feats_all.append(audio_feat)
                ip_t=f'Utterance {Utterance_ID}, Speaker {Speaker} Says: <text>{Utterance}<video><videoblankpos><videoblankpos><audio><audioblankpos><audioblankpos><sep> '
                prompt_utt_texts.append(ip_t)
                # ip_t_ids = self.tokenizer.encode(ip_t)
                
                # question_input = [ "".join([q,a]) for q, a in zip(question_data["question"], question_data["answer"])]
                question_data_id={"question":[],'answer':[]}
                qs_ids=[]
                question_input=''
                qs_label=[]
                for q, a in zip(question_data["question"], question_data["answer"]):
                    question_input+=q+a
                    
                    q_ids = self.tokenizer.encode(q)
                    a_ids =self.tokenizer.encode(a)
                    question_data_id["question"].append(q_ids)
                    question_data_id['answer'].append(a_ids)
                    qs_ids+=q_ids+a_ids
                    qs_label+=[-100]*len(q_ids)+a_ids
               
                if self.split_name_list[0] not in ['train']:
                    qs_ids = question_data_id["question"][0]
                    qs_label=[-100]*len(q_ids)+a_ids
                
                
                    
                #  Find ! all the length of dialogue is not more than 1024.
                
              
                uttItemstr = ''.join(prompt_utt_texts)+'<|endoftext|>'
                ttItemstr_qs =uttItemstr+', '+ question_input
                utts_ids = self.tokenizer.encode(uttItemstr)
                prompt_ids = utts_ids + qs_ids
                
                sep_id = self.tokenizer.sep_token_id
                pos_type_id = [i for i, id in enumerate(prompt_ids) if id == sep_id]
                token_type_id=[]
                
                pre_id = -1
                question_token_id = self.tokenizer.encode('question')
                assert len(question_token_id)==1
                for j, pos_type in enumerate(pos_type_id):
                    type_id_list = self.tokenizer.encode(f'{j+1}')
                    assert len(type_id_list)==1
                    token_type_id+=type_id_list*(pos_type-pre_id)
                    pre_id = pos_type
                
                token_type_id+=question_token_id*len(qs_ids)
                
                # token_type_id=[0]*(len(utts_ids))+[1]*len(qs_ids)

                label=[-100]*(len(utts_ids))+qs_label
                att_mask = [1]*len(prompt_ids)

              
                self.utt_prompt_ids.append(prompt_ids)
                self.utt_prompt_attention_masks.append(att_mask)
                self.token_type_ids.append(token_type_id)
                self.video_feats.append(video_feats_all)
                self.audio_feats.append(audio_feats_all)
                self.gpt2_labels.append(label)

                self.utt_prompts.append(ttItemstr_qs)
                self.key_names.append(utt_key)
                self.Emotion_names.append(str(Emotion))
                self.Cause_names.append(Causes_txt)
                self.question_datas.append(question_data)
                self.question_data_ids.append(question_data_id)


        self.max_video_len = max([len(item) for item in self.video_feats])
        self.max_audio_len = max([len(item) for item in self.audio_feats])
        assert self.max_video_len == self.max_audio_len
        self.max_av_len = self.max_video_len

        self.max_token_len = max([len(item) for item in self.utt_prompt_ids])
        print('max_video_len--',self.max_video_len)
        print('max_token_len--',self.max_token_len)
        self.max_prompt_len =self.max_token_len



    def pad_data(self,):
        self.pad_utt_prompt_ids = []
        self.pad_utt_prompt_attention_masks=[]
        self.pad_token_type_ids = []
        self.pad_video_feats=[]
        self.pad_audio_feats=[]
        self.pad_gpt2_labels = []

        for utt_prompt_id,utt_prompt_attention_mask,token_type_id,video_feat,audio_feat,gpt2_label \
            in zip(self.utt_prompt_ids,self.utt_prompt_attention_masks,self.token_type_ids,self.video_feats,self.audio_feats,self.gpt2_labels):
            
            pad_token_id = self.tokenizer.pad_token_id
            pad_utt_prompt_id= [pad_token_id]*(self.max_prompt_len-len(utt_prompt_id)) + utt_prompt_id 
            pad_utt_prompt_attention_mask= [0]*(self.max_prompt_len-len(utt_prompt_attention_mask)) + utt_prompt_attention_mask
            pad_token_type_id = [pad_token_id]*(self.max_prompt_len-len(token_type_id)) + token_type_id

            video_feat_ts= torch.tensor(np.concatenate([item[np.newaxis,:] for item in video_feat],0))
            audio_feat_ts = torch.tensor(np.concatenate([item[np.newaxis,:] for item in audio_feat],0))
            # print(self.max_video_len,video_feat_ts.shape[1])
            pad_video_feat = torch.zeros(self.max_av_len,video_feat_ts.shape[1])
            pad_audio_feat = torch.zeros(self.max_av_len,audio_feat_ts.shape[1])
            pad_video_feat[:video_feat_ts.shape[0]]=video_feat_ts
            pad_audio_feat[:audio_feat_ts.shape[0]]=audio_feat_ts

            pad_gpt2_label= [-100]*(self.max_prompt_len-len(gpt2_label)) + gpt2_label

            self.pad_utt_prompt_ids.append(pad_utt_prompt_id)
            self.pad_utt_prompt_attention_masks.append(pad_utt_prompt_attention_mask)
            self.pad_token_type_ids.append(pad_token_type_id)
            self.pad_video_feats.append(pad_video_feat)
            self.pad_audio_feats.append(pad_audio_feat)
            self.pad_gpt2_labels.append(pad_gpt2_label)

            # if len(pad_utt_prompt_id) != self.max_prompt_len or len(pad_utt_prompt_attention_mask) !=self.max_prompt_len\
            #     or len(pad_utt_prompt_attention_mask)!= self.max_prompt_len or len(pad_token_type_id)!=self.max_prompt_len\
            #         or len(pad_gpt2_label) != self.max_prompt_len:


    def get_Cause_prompt(self,vid=None,Emotion=None):
        emotion_answer = f'{Emotion}<|endoftext|>'
        emotion_answer_ids = self.tokenizer.encode(emotion_answer)
        casue_question_ids = self.tokenizer.encode(f'. if not neutral, which utterance is the cause of this emotion ? <|endoftext|>')
        prompt_ids = self.utt_prompt_ids[vid]+emotion_answer_ids+casue_question_ids
        att_mask =[1]*len(prompt_ids)
        token_type_id =self.token_type_ids[vid]+[1]*len(emotion_answer)+[1]*len(casue_question_ids)
        video_feat =self.pad_video_feats[vid]
        audio_feat =self.pad_audio_feats[vid]
        gpt2_label =self.gpt2_labels[vid]+emotion_answer+[-100]*len(casue_question_ids)

        video_blank_idx = self.tokenizer.encode('<videoblankpos>')[0]
        audio_blank_idx = self.tokenizer.encode('<audioblankpos>')[0]
        video_blank_id = torch.tensor([0 if id!=video_blank_idx else 1 for id in prompt_ids])
        audio_blank_id = torch.tensor([0 if id!=audio_blank_idx else 1 for id in prompt_ids])

        return prompt_ids, att_mask,token_type_id,video_feat,audio_feat,gpt2_label,video_blank_id,audio_blank_id



    def __getitem__(self, idx):
    # def getitem(self, idx):
        # get question in token ids, image in features,
        # and answer in binary-label vector

        __C = self.__C

        pad_utt_prompt_id =self.pad_utt_prompt_ids[idx]
        pad_utt_prompt_attention_mask =self.pad_utt_prompt_attention_masks[idx]
        token_type_id= self.pad_token_type_ids[idx]
        gpt2_label = self.pad_gpt2_labels[idx]
        video_feat = self.pad_video_feats[idx]
        audio_feat = self.pad_audio_feats[idx]


        # pad_utt_prompt = self.pad_utt_prompts[idx]
        # pad_utt_emo = self.pad_utt_emos[idx]
        # pad_utt_cause = self.pad_utt_causes[idx]
        # key_name = self.key_names[idx]

        ### pad this feats
       
        # video_feat_ts= torch.tensor(np.concatenate([item[np.newaxis,:] for item in self.video_feats[idx]],0))
        # audio_feat_ts = torch.tensor(np.concatenate([item[np.newaxis,:] for item in self.audio_feats[idx]],0))
        # # print(self.max_video_len,video_feat_ts.shape[1])
        # video_feat = torch.zeros(self.max_video_len,video_feat_ts.shape[1])
        # audio_feat = torch.zeros(self.max_audio_len,audio_feat_ts.shape[1])
        # video_feat[:video_feat_ts.shape[0]]=video_feat_ts
        # audio_feat[:audio_feat_ts.shape[0]]=audio_feat_ts

        ### get video audio idx 
        video_blank_idx = self.tokenizer.encode('<videoblankpos>')[0]
        audio_blank_idx = self.tokenizer.encode('<audioblankpos>')[0]
        video_blank_id = torch.tensor([0 if id!=video_blank_idx else 1 for id in pad_utt_prompt_id])
        audio_blank_id = torch.tensor([0 if id!=audio_blank_idx else 1 for id in pad_utt_prompt_id])


     
        return torch.LongTensor(pad_utt_prompt_id),torch.tensor(pad_utt_prompt_attention_mask).float(),torch.tensor(token_type_id).long(),torch.tensor(gpt2_label),\
               video_feat,audio_feat,video_blank_id.long(),audio_blank_id.long()
                
    def __len__(self):
        return self.data_size

    def bert_tokenize(self, text, max_token):
        text = text.lower().replace('?', '')
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > max_token - 2:
            tokens = tokens[:max_token-2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = ids + [0] * (max_token - len(ids))
        ids = np.array(ids, np.int64)

        return ids
    
    def load_embedding_from_npy(self,video_id_mapping_file, video_emb_file, audio_emb_file, path_dir = ''):
        def normalize(x):
            x1 = x[1:,:]
            min_x = np.min(x1, axis=0, keepdims=True)
            max_x = np.max(x1, axis=0, keepdims=True)
            x1 = (x1-min_x)/(max_x-min_x+1e-8)
            x[1:,:] = x1
            return x

        # v_id_map = np.load(video_id_mapping_file, allow_pickle=True) # dia1utt1: 1 (13620, 4096)
        v_id_map = eval(str(np.load(video_id_mapping_file, allow_pickle=True))) # dia1utt1: 1
        v_emb = normalize(np.load(video_emb_file, allow_pickle=True)) # (13620, 4096)
        a_emb = normalize(np.load(audio_emb_file, allow_pickle=True)) # (13620, 6373)
        
        print('\nload video_emb_file: {}\nload audio_emb_file: {}\n'.format(video_emb_file, audio_emb_file))
        return v_id_map, v_emb, a_emb
    def load_text(self,text_path):
        with open(text_path, 'r') as fp:
            dialogues=json.load(fp)
        return dialogues
    
    def shuffle(self,):
        # random.shuffle(zip())
        pass






                # for  i in range(0,len(prompt_utt_texts)):
                #     uttItemstr = ''.join(prompt_utt_texts[i:])+''
                    
                #     uttItemstr_qs =uttItemstr+', '+ question1
                #     # txt_ids = self.tokenizer.encode(uttItemstr_qs)
                #     utts_ids = self.tokenizer.encode(uttItemstr)

                #     txt_ids = utts_ids+qs_ids
                #     # newqs = self.tokenizer.decode(txt_ids)
                #     # question1_ids = self.tokenizer.encode(question1)
                    
                #     if len(txt_ids)<=1024:
                #         pad_it = txt_ids+pad_id*(1024-len(txt_ids))
                #         pad_mask = [1]*len(txt_ids)+[0]*(1024-len(txt_ids))
                #         self.pad_utt_prompt_ids.append(pad_it)
                #         self.pad_utt_prompt_attention_masks.append(pad_mask)
                        
                #         # pos_type_id = [i for i, id in enumerate(txt_ids) if id == sep_id]
                #         # token_type_id=[]
                        
                #         # pre_id = -1
                #         # for j, pos_type in enumerate(pos_type_id):
                #         #     type_id_list = self.tokenizer.encode(f'{i+j+1}')
                #         #     assert len(type_id_list)==1
                #         #     token_type_id+=type_id_list*(pos_type-pre_id)
                #         #     pre_id = pos_type
                        
                #         # token_type_id+=question_token_id_list*len(qs_ids)+pad_token_id_list*(1024-len(txt_ids))
                #         # 0 for content 1 for question answer.
                #         token_type_id=[0]*(len(utts_ids))+[1]*(1024-len(utts_ids))

                        
                #         self.token_type_ids.append(token_type_id)

                #         label=[-100]*(len(utts_ids))+qs_label
                #         label_pad = label+[-100]*(1024-len(txt_ids))
                     

                       
                #         self.gpt2_labels.append(label_pad)
                #         self.pad_utt_prompts.append(uttItemstr_qs)
                #         self.pad_utt_emos.append(Emotion_idx)
                #         self.pad_utt_causes.append(Cause_idxs)
                #         self.key_names.append(utt_key)
                      
                #         self.video_feats.append(video_feats_all[i:])
                #         self.audio_feats.append(audio_feats_all[i:])
                #         break
       
    # if 'expanded emotion cause evidence' in utt.keys():
                    
                #     # question1 = f"What is the emotion of Utterance {Utterance_ID},Speaker {Speaker} ? <|endoftext|>{Emotion}<|endoftext|>. if not neutral, the cause of this emotion is Utterance {Causes_txt}<|endoftext|>"
                #     # pref = self.tokenizer.encode(f"The emotion of Utterance {Utterance_ID},Speaker {Speaker} is ")
                #     # mid=self.tokenizer.encode(". if not neutral,  the cause of this emotion is Utterance ")
                #     qs_ids = pref+self.tokenizer.encode(Emotion)+mid+self.tokenizer.encode(Causes_txt)
                #     qs_label = [-100]*len(pref)+self.tokenizer.encode(Emotion)+[-100]*len(mid)+self.tokenizer.encode(Causes_txt)
                    
                # else:
                #     question1 = f"The emotion of Utterance {Utterance_ID},Speaker {Speaker} is {Emotion}."
                #     pref = self.tokenizer.encode(f"The emotion of Utterance {Utterance_ID},Speaker {Speaker} is ")
                #     mid=self.tokenizer.encode(".")
                #     qs_ids = pref+self.tokenizer.encode(Emotion)+mid
                #     qs_label = [-100]*len(pref)+self.tokenizer.encode(Emotion)+[-100]*len(mid)
                    

                # question1 = " Does the utterance{ID},Speaker {Speaker} have the emotion? response yes/no"
                # question1 = f"The emotion of Utterance {Utterance_ID},Speaker {Speaker} is {Emotion}. if not neutral,  the cause of this emotion is Utterance {}"
                # if __C.EmoAnnotation:
                #     question1 = f"The emotion of Utterance {Utterance_ID},Speaker {Speaker} is {Emotion}. if not neutral,  the cause of this emotion is Utterance [mask]"
                # else:
                #     question1 = f"The emotion of Utterance {Utterance_ID},Speaker {Speaker} is [mask]. if not neutral,  the cause of this emotion is Utterance [mask]"
                # we need to careful precess the token_type_id, for the content_sente
                # question_token_id_list = self.tokenizer.encode('question') # [id]
                # pad_token_id_list = self.tokenizer.encode('none') # [id]
                # sep_id = self.tokenizer.encode('<sep>')[0]