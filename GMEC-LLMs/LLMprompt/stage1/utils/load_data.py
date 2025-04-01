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
random.seed(24) 
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
        self.tokenizer.padding_side="right" # "left" for gpt2
        # self.tokenizer.pad_token = self.tokenizer.eos_token
     
        self.tokenizer.add_tokens(["<video>"], special_tokens=True)
        self.tokenizer.add_tokens(["<audio>"], special_tokens=True)
        self.tokenizer.add_tokens(["<text>"], special_tokens=True)
        self.tokenizer.add_special_tokens({"sep_token": "<sep>"})
        self.tokenizer.add_tokens(["<question>"], special_tokens=True)
        self.tokenizer.add_tokens(["<videoblankpos>"], special_tokens=True)
        self.tokenizer.add_tokens(["<audioblankpos>"], special_tokens=True)
        # self.tokenizer.add_tokens(["[mask]"], special_tokens=True)
        # self.tokenizer.add_tokens(["<pad>"], special_tokens=True)
        self.tokenizer.add_tokens("anger")
        self.tokenizer.add_tokens("surprise")
        self.tokenizer.add_tokens("disgust")
        self.tokenizer.add_tokens("neutral")
        self.tokenizer.add_tokens("sadness")
        self.tokenizer.add_tokens("joy")
        self.tokenizer.add_tokens("fear")
        self.token_size = self.tokenizer.vocab_size
        print(f'== BertTokenizer loaded, vocab size: {self.token_size}')
        # load text
        self.text_path = __C.TEXT_DIR
        self.config = AutoConfig.from_pretrained(__C.BERT_VERSION)
        self.config.num_labels=7 if __C.TASK_TYPE == "emotion" else 2
    
        self.Emotion2Idxs={
            'neutral': 29797,
            'anger':2564,
            'surprise':50263,
            'sadness':50265,
            'joy':2633,
            'disgust':50264,
            'fear':50266,
        }
        self.emotionids=[item for _,item in self.Emotion2Idxs.items()]
        self.Emotion2index={k:i for i,(k,_) in enumerate(self.Emotion2Idxs.items())}

        self.YesNo2Idxs ={
            'Yes':5297,
            'No':2949,
        }
        self.YesNoids=[item for _, item in self.YesNo2Idxs.items()]
        self.YesNo2index={k:i for i,(k,_) in enumerate(self.YesNo2Idxs.items())}
       
      
    
        print('Common data process is done.\n')
        

class DataSet(Data.Dataset):
    def __init__(self, __C, common_data, split_name_list,train_for_heuris=None):
        self.__C = __C
        print(f'Loading dataset for {self.__C.TASK}|{self.__C.RUN_MODE}({split_name_list})')
        self.split_name_list = split_name_list
      
        self.common_data = common_data
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
        if self.__C.M_TAV=='T':
            video_embedding=video_embedding*0
            audio_embedding=audio_embedding*0
        elif self.__C.M_TAV=='TA':
            video_embedding=video_embedding*0
        elif self.__C.M_TAV=='TV':
            audio_embedding=audio_embedding*0
        
        self.video_idx= video_idx
        self.video_embedding= video_embedding
        self.audio_embedding= audio_embedding
    
        self.dialogues = self.load_text(self.text_path)
        
       
        if split_name_list[0]=='train' and self.__C.SIZELEN<1001:
            cut_num=self.__C.SIZELEN #00 #1001/3
         
           
            keyns= list(self.dialogues.keys())[:cut_num]
            self.dialogues={k:self.dialogues[k] for k in keyns}
        self.dia279_txt_len  = 39
        # self.list_of_dialogues = [(key,value) for key,value in  self.dialogues.items()]
        # word_idx_rev, word_idx, spe_idx_rev, spe_idx, word_embedding, _ = self.load_w2v(__C.embedding_dim, __C.embedding_dim_pos, self.text_path['all'], self.ecf_path)
    
       
        self.max_dia_len = max([len(dia) for key,dia in self.dialogues.items()])
        self.max_utt_len = max([len(utt['Utterance']) for key,dia in self.dialogues.items()for utt in dia  ])
        self.mean_utt_len = mean([len(utt['Utterance']) for key,dia in self.dialogues.items()for utt in dia  ])
        print(f'max_dia_len == {self.max_dia_len}\n')
        print(f'max_utt_len == {self.max_utt_len}\n')
        print(f'mean_utt_len == {self.mean_utt_len}\n')
        self.preEmo_test_file='./outputs/results/training_mecpeNeu/result_20230530160539_emotion_neu_2_test.json'
        self.precause_test_file = './outputs/results/training_mecpeNeu/result_20230530161748_cause_1_test.json'
        if train_for_heuris:
            self.split_name_list=['train_for_heuris']
      
        self.make_questions()
        self.make_new_dataset()

        self.data_size = len(self.utt_prompt_ids)
        print(f'== data size: {self.data_size}\n')
        self.pad_data()

        # self.small_sample_for_test(20)

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
        self.sample_video_feats=self.sample_video_feats[:sample_size]
        self.sample_audio_feats=self.sample_audio_feats[:sample_size]
        self.gpt2_labels = self.gpt2_labels[:sample_size]

        # self.utt_prompts = self.utt_prompts[:sample_size]
        self.key_names=self.key_names[:sample_size]
        # self.Emotion_names=self.Emotion_names[:sample_size]
        # self.Cause_names=self.Cause_names[:sample_size]
        # self.question_datas = self.question_datas[:sample_size]
        # self.question_data_ids = self.question_data_ids[:sample_size]

        self.data_size = len(self.pad_utt_prompt_ids)
        print(f'== data size: {self.data_size}\n')

    def make_questions(self,):
        self.Emotion_quetions = {}
        self.Cause_questions = {}
        self.Emotion_Cause_questions={}
        self.prompt_texts={}
        self.video_feats = {}
        self.audio_feats = {}
        for dia_key,dialogue in self.dialogues.items():
            video_feats_dia, audio_feats_dia = {},{}
            prompt_dia_texts = []

            Dia_Emotion_questions=[]
            Dia_Cause_questions = []
            Dia_Emotion_Cause_questions=[]
            dia_len = len(dialogue)
            causes_sum = []
            for utt in dialogue:
               
                Utterance_ID= eval(utt['Utterance_ID'])
                Speaker=utt['Speaker']
                Emotion = utt['Emotion']
                Emotion_neu = 'neutral' if Emotion == 'neutral' else 'not neutral'
                Emotion_neu_yesno = 'Yes' if Emotion_neu=='neutral' else 'No'
                Utterance = utt['Utterance']
                if dia_key == '279':
                    Utterance=Utterance[:self.dia279_txt_len]

                # txt_id = self.bert_tokenize(text, __C.MAX_TOKEN)
                # sp_id = self.bert_tokenize(Speaker, __C.MAX_SPEAKER_TOKEN)
                utt_key = 'dia{}utt{}'.format(dia_key,Utterance_ID)
                pos = self.video_idx[utt_key]
               
                # video_feat = self.video_embedding[pos]
                # audio_feat = self.audio_embedding[pos]
                video_feats_dia[str(Utterance_ID)]=pos
                audio_feats_dia[str(Utterance_ID)]=pos


                ip_t=f'Utterance {Utterance_ID}, Speaker {Speaker} Says: <text>{Utterance}<video><videoblankpos><videoblankpos><audio><audioblankpos><audioblankpos>'
                prompt_dia_texts.append(ip_t)


                Emotion_idx = self.tokenizer.encode(Emotion)[1:-1]
           
                assert len(Emotion_idx)==1
               
                Cause_idxs = []
                Causes_txt=''
                Causes=[]
                Dia_Emotion_questions.append({'question':f'The emotion of utterance {Utterance_ID},Speaker {Speaker}: <text>{Utterance}<video><videoblankpos><videoblankpos><audio><audioblankpos><audioblankpos> is ? ',
                                               'answer':f'{Emotion}',
                                               'utt_key':utt_key,
                                               'emotion':Emotion,
                                               'emotion_token_id':Emotion_idx[0],
                                               'Emotion_utterance_ID':Utterance_ID,
                                               'Emotion_neu':Emotion_neu,
                                               'question_neu':f'The emotion of utterance {Utterance_ID},Speaker {Speaker}: <text>{Utterance}<video><videoblankpos><videoblankpos><audio><audioblankpos><audioblankpos> is neutral ? ',
                                               'answer_neu':f'{Emotion_neu_yesno}',
                                               })
                
                        
                if 'expanded emotion cause evidence' in utt.keys():
                   
                    Causes= utt['expanded emotion cause evidence']
                    causes_sum+=Causes
                Dia_Cause_questions.append({
                    'question':f' The utterance {Utterance_ID},Speaker {Speaker}: <text>{Utterance}<video><videoblankpos><videoblankpos><audio><audioblankpos><audioblankpos> is a causal utterance ? ',
                    'answer':f'No',
                    'answer_token_id': self.tokenizer.encode('No')[1:-1][0],
                    'cause':'No',
                    'utt_key':utt_key,
                    'Cause_utterance_ID':Utterance_ID,
                })
             
                # for emotion_cause 
                for utt2 in dialogue:
                    text2= utt2['Utterance']
                    Utterance_ID2= eval(utt2['Utterance_ID'])
                    Speaker2=utt2['Speaker']
                    Emotion2 = utt2['Emotion']
                    Utterance2 = utt2['Utterance']
                    if Utterance_ID2 in Causes:
                        Dia_Emotion_Cause_questions.append({
                            'question':'The {}'+ f' utterance {Utterance_ID2},Speaker {Speaker2}: <text>{Utterance2}<video><videoblankpos><videoblankpos><audio><audioblankpos><audioblankpos> is the cause of the '+'{}'+f' utterance {Utterance_ID},Speaker {Speaker}: <text>{Utterance}<video><videoblankpos><videoblankpos><audio><audioblankpos><audioblankpos> ? ',
                            'answer':'Yes',
                            'answer_token_id': self.tokenizer.encode('Yes')[1:-1][0],
                            'cause':'Yes',
                            'Emotion_utterance_ID':Utterance_ID,
                            'Cause_utterance_ID':Utterance_ID2,
                            'Emotion':Emotion,
                            'Emotion_neu':Emotion_neu
                        })
                      
                    else:
                        Dia_Emotion_Cause_questions.append({
                           'question':'The {}'+ f' utterance {Utterance_ID2},Speaker {Speaker2}: <text>{Utterance2}<video><videoblankpos><videoblankpos><audio><audioblankpos><audioblankpos> is the cause of the '+'{}'+f' utterance {Utterance_ID},Speaker {Speaker}: <text>{Utterance}<video><videoblankpos><videoblankpos><audio><audioblankpos><audioblankpos> ? ',
                               'answer':'No',
                            'answer_token_id': self.tokenizer.encode('No')[1:-1][0],
                            'cause':'No',
                            'Emotion_utterance_ID':Utterance_ID,
                            'Cause_utterance_ID':Utterance_ID2,
                            'Emotion':Emotion,
                            'Emotion_neu':Emotion_neu
                        })

            for item in Dia_Emotion_Cause_questions:
                Cause_utterance_ID= item['Cause_utterance_ID']
                Emotion_utterance_ID= item['Emotion_utterance_ID']
                if Cause_utterance_ID in list(set(causes_sum)):
                     item['cause_label'] = 'causal'
             
            for item in Dia_Cause_questions:
                Utterance_ID= item['Cause_utterance_ID']
                if Utterance_ID in list(set(causes_sum)):
                          
                    item['answer'] = f'Yes'
                    item['answer_token_id'] =self.tokenizer.encode('Yes')[1:-1][0]

            self.Emotion_quetions[dia_key]= Dia_Emotion_questions
            self.Cause_questions[dia_key]= Dia_Cause_questions
            self.Emotion_Cause_questions[dia_key]=Dia_Emotion_Cause_questions
            self.prompt_texts[dia_key]=prompt_dia_texts
            self.video_feats[dia_key] = video_feats_dia
            self.audio_feats[dia_key] = audio_feats_dia
        
        self.save_json_data(self.Emotion_quetions,'./question_files/Emotion_quetions_'+self.split_name_list[0]+'.json')
        self.save_json_data(self.Cause_questions,'./question_files/Cause_questions_'+self.split_name_list[0]+'.json')
        self.save_json_data(self.Emotion_Cause_questions,'./question_files/Emotion_Cause_questions_'+self.split_name_list[0]+'.json')
        self.save_json_data(self.prompt_texts,'question_files/prompt_texts_'+self.split_name_list[0]+'.json')
       
            

        
    def save_json_data(self, data, tgt_path):
        with open(tgt_path, "w",encoding='utf8') as fp:
            fp.write(json.dumps(data, indent=4, ensure_ascii=False))

    def make_new_dataset(self,):
        self.utt_prompt_ids = []
        self.utt_prompt_attention_masks=[]
        self.token_type_ids = []
        # self.token_type_ids2 = []
        self.sample_video_feats=[]
        self.sample_audio_feats=[]
        self.gpt2_labels = []

        self.key_names = []

        for dia_key,dialogue in self.dialogues.items():
            Dia_Emotion_questions = self.Emotion_quetions[dia_key]
            Dia_Cause_questions = self.Cause_questions[dia_key]
            Dia_Emotion_Cause_questions = self.Emotion_Cause_questions[dia_key]
            prompt_dia_texts = self.prompt_texts[dia_key]
           
            video_feats_dia = self.video_feats[dia_key]
            audio_feats_dia = self.audio_feats[dia_key]

            if self.__C.Context_cut == 'static':
                uttItemstr = self.tokenizer.sep_token.join(prompt_dia_texts)
                # utts_ids = self.tokenizer.encode(uttItemstr)
            # token_type_id2=[]


           
            # 'emotion'''emotion_neu','cause','AnnoEmo_cause_pair','AnnoEmo_cause_pair_neu','emo_cause_pair','emo_cause_pair_neu','all'
            if self.__C.TASK_TYPE == 'emotion' or self.__C.TASK_TYPE == 'emotion_neu' or self.__C.TASK_TYPE == 'cause':
                Dia_questions = Dia_Cause_questions  if self.__C.TASK_TYPE == 'cause' else Dia_Emotion_questions
                for i,item in enumerate(Dia_questions):
                    q =item['question_neu'] if self.__C.TASK_TYPE == 'emotion_neu' else item['question'] 
                    a = item['answer_neu'] if self.__C.TASK_TYPE == 'emotion_neu' else item['answer'] 
                    utterance_ID = item['Cause_utterance_ID'] if self.__C.TASK_TYPE == 'cause' else item['Emotion_utterance_ID']

                    if self.__C.Context_cut == 'realtime':
                        new_pdt=prompt_dia_texts[:utterance_ID]
                        uttItemstr = self.tokenizer.sep_token.join(new_pdt)
                    elif self.__C.Context_cut == 'ct_cut':
                        new_pdt=prompt_dia_texts[:utterance_ID+self.__C.context_len]
                        uttItemstr = self.tokenizer.sep_token.join(new_pdt)
                    elif self.__C.Context_cut == 'static':
                        pass
                    else:
                        raise ValueError("Context_cut not found")

                    # full_question = uttItemstr+'</s>'+q
                    prompt_ids=self.tokenizer.encode(uttItemstr,q)
                
                    token_type_id=[0]*len(prompt_ids)
                    att_mask = [1]*len(prompt_ids)
                 
                    label =[self.common_data.Emotion2index[a] if  self.__C.TASK_TYPE == 'emotion' else self.common_data.YesNo2index[a]]
                    # q_ids = self.tokenizer.encode(q)
                    # a_ids =self.tokenizer.encode(a)
                    # stop()
                    # assert len(a_ids)==2
                    # assert a_ids[0] in self.common_data.emotionids or a_ids[0] in self.common_data.YesNoids
                    # qs_ids =q_ids+a_ids
                    # qs_label=[-100]*len(q_ids)+[a_ids[0]]+[-100]
                    # if 'train' not in self.split_name_list:
                    #     qs_ids=q_ids
                    #     qs_label=[-100]*len(q_ids)+[a_ids[0]]+[-100]

                    # prompt_ids = utts_ids + qs_ids

                    # token_type_id=[0]*(len(utts_ids))+[1]*len(qs_ids)
                    # att_mask = [1]*len(prompt_ids)
                    # label=[-100]*(len(utts_ids))+qs_label
                   
                    utts_video_feat = [vi for _,vi in video_feats_dia.items()]
                    qs_video_feat = video_feats_dia[str(utterance_ID)]
                    prompt_video_feat = utts_video_feat+[qs_video_feat]
                    utts_audio_feat = [vi for _,vi in audio_feats_dia.items()]
                    qs_audio_feat = audio_feats_dia[str(utterance_ID)]
                    prompt_audio_feat = utts_audio_feat+[qs_audio_feat]
                    

                    self.utt_prompt_ids.append(prompt_ids)
                    self.utt_prompt_attention_masks.append(att_mask)
                    self.token_type_ids.append(token_type_id)
                    self.sample_video_feats.append(prompt_video_feat)
                    self.sample_audio_feats.append(prompt_audio_feat)
                    self.gpt2_labels.append(label)
                    key_name = f'dia_{dia_key}_utt_{utterance_ID}_index_{i}'
                    self.key_names.append(key_name)   
            
            elif self.__C.TASK_TYPE == 'AnnoEmo_Annocause_pair' or self.__C.TASK_TYPE == 'AnnoEmo_Annocause_pair_neu':
                for i, item in enumerate(Dia_Emotion_Cause_questions):
                    Emotion = item['Emotion'] if self.__C.TASK_TYPE == 'AnnoEmo_Annocause_pair' else item['Emotion_neu']
                    Cause = item['cause_label'] if 'cause_label' in item else None
                    
                    if self.__C.RUN_MODE=='train_test':
                        type_info =True
                    else:
                        type_info = Emotion!='neutral' and Cause
                    if type_info:
                        qr=item['question'] 
                        # q=qr.format(Emotion,Cause)
                        q=qr.format('','')
                        a = item['answer'] 
                        Emotion_utterance_ID = item['Emotion_utterance_ID'] 
                        Cause_utterance_ID = item['Cause_utterance_ID'] 

                        prompt_ids=self.tokenizer.encode(uttItemstr,q)
                        token_type_id=[0]*len(prompt_ids)
                        att_mask = [1]*len(prompt_ids)
                        label =[ self.common_data.YesNo2index[a]]
                        # q_ids = self.tokenizer.encode(q)
                        # a_ids =self.tokenizer.encode(a)

                        # assert len(a_ids)==2
                        # assert a_ids[0] in self.common_data.emotionids or a_ids[0] in self.common_data.YesNoids
                        # qs_ids =q_ids+a_ids
                        # qs_label=[-100]*len(q_ids)+[a_ids[0]]+[-100]
                        # if 'train' not in self.split_name_list:
                        #     qs_ids=q_ids
                        #     qs_label=[-100]*len(q_ids)+[a_ids[0]]+[-100]

                        # prompt_ids = utts_ids + qs_ids
                        # token_type_id=[0]*(len(utts_ids))+[1]*len(qs_ids)
                        # att_mask = [1]*len(prompt_ids)
                        # label=[-100]*(len(utts_ids))+qs_label
                       

                        utts_video_feat = [vi for _,vi in video_feats_dia.items()]
                        qs_video_feat = [video_feats_dia[str(Cause_utterance_ID)]]+[video_feats_dia[str(Emotion_utterance_ID)]]
                        prompt_video_feat = utts_video_feat+qs_video_feat
                        utts_audio_feat = [vi for _,vi in audio_feats_dia.items()]
                        qs_audio_feat = [audio_feats_dia[str(Cause_utterance_ID)]]+[audio_feats_dia[str(Emotion_utterance_ID)]]
                        prompt_audio_feat = utts_audio_feat+qs_audio_feat
                        

                        self.utt_prompt_ids.append(prompt_ids)
                        self.utt_prompt_attention_masks.append(att_mask)
                        self.token_type_ids.append(token_type_id)
                        self.sample_video_feats.append(prompt_video_feat)
                        self.sample_audio_feats.append(prompt_audio_feat)
                        self.gpt2_labels.append(label)
                        key_name = f'dia_{dia_key}_uttEmo_{Emotion_utterance_ID}_uttCas_{Cause_utterance_ID}_index_{i}'
                        self.key_names.append(key_name) 
           
            elif self.__C.TASK_TYPE == 'AnnoEmo_precause_pair' or self.__C.TASK_TYPE == 'AnnoEmo_precause_pair_neu':
              
                for i, item in enumerate(Dia_Emotion_Cause_questions):
                    Emotion = item['Emotion'] if self.__C.TASK_TYPE == 'AnnoEmo_cause_pair' else item['Emotion_neu']
                    if self.__C.RUN_MODE=='train_test':
                        type_info =True
                    else:
                        type_info = Emotion!='neutral'
                    if type_info:
                        qr=item['question'] 
                        q=qr.format(Emotion,'')
                        # q=qr.format('','')
                        a = item['answer'] 
                        Emotion_utterance_ID = item['Emotion_utterance_ID'] 
                        Cause_utterance_ID = item['Cause_utterance_ID'] 
                        
                        if self.__C.Context_cut == 'realtime':
                            new_pdt=prompt_dia_texts[:Emotion_utterance_ID]
                            uttItemstr = self.tokenizer.sep_token.join(new_pdt)
                        if Emotion_utterance_ID<Cause_utterance_ID:
                            continue
                        elif self.__C.Context_cut == 'static':
                            pass
                        else:
                            raise ValueError("Context_cut not found")

                        # full_question = uttItemstr+'</s>'+q
                        prompt_ids=self.tokenizer.encode(uttItemstr,q)
                    
                        token_type_id=[0]*len(prompt_ids)
                        att_mask = [1]*len(prompt_ids)
                    
                        label =[self.common_data.Emotion2index[a] if  self.__C.TASK_TYPE == 'emotion' else self.common_data.YesNo2index[a]]
                        # q_ids = self.tokenizer.encode(q)
                        # a_ids =self.tokenizer.encode(a)
                        # assert len(a_ids)==2
                        # assert a_ids[0] in self.common_data.emotionids or a_ids[0] in self.common_data.YesNoids
                        # qs_ids =q_ids+a_ids
                        # qs_label=[-100]*len(q_ids)+[a_ids[0]]+[-100]
                        # if 'train' not in self.split_name_list:
                        #     qs_ids=q_ids
                        #     qs_label=[-100]*len(q_ids)+[a_ids[0]]+[-100]

                        # prompt_ids = utts_ids + qs_ids
                        # token_type_id=[0]*(len(utts_ids))+[1]*len(qs_ids)
                        # att_mask = [1]*len(prompt_ids)
                        # label=[-100]*(len(utts_ids))+qs_label

                        utts_video_feat = [vi for _,vi in video_feats_dia.items()]
                        qs_video_feat = [video_feats_dia[str(Cause_utterance_ID)]]+[video_feats_dia[str(Emotion_utterance_ID)]]
                        prompt_video_feat = utts_video_feat+qs_video_feat
                        utts_audio_feat = [vi for _,vi in audio_feats_dia.items()]
                        qs_audio_feat = [audio_feats_dia[str(Cause_utterance_ID)]]+[audio_feats_dia[str(Emotion_utterance_ID)]]
                        prompt_audio_feat = utts_audio_feat+qs_audio_feat
                        

                        self.utt_prompt_ids.append(prompt_ids)
                        self.utt_prompt_attention_masks.append(att_mask)
                        self.token_type_ids.append(token_type_id)
                        self.sample_video_feats.append(prompt_video_feat)
                        self.sample_audio_feats.append(prompt_audio_feat)
                        self.gpt2_labels.append(label)
                        key_name = f'dia_{dia_key}_uttEmo_{Emotion_utterance_ID}_uttCas_{Cause_utterance_ID}_index_{i}'
                        self.key_names.append(key_name) 
             
            elif self.__C.TASK_TYPE == 'preEmo_precause_pair':
                 for i, item in enumerate(Dia_Emotion_Cause_questions):
                    Emotion = item['Emotion'] if self.__C.TASK_TYPE == 'AnnoEmo_Annocause_pair' else item['Emotion_neu']
                    Cause = item['cause_label'] if 'cause_label' in item else None
                    qr=item['question'] 
                    q=qr.format('','')
                    a = item['answer']
                    Emotion_utterance_ID = item['Emotion_utterance_ID'] 
                    Cause_utterance_ID = item['Cause_utterance_ID'] 
                    
                    if self.__C.Context_cut == 'realtime':
                        new_pdt=prompt_dia_texts[:Emotion_utterance_ID]
                        uttItemstr = self.tokenizer.sep_token.join(new_pdt)
                        if Emotion_utterance_ID<Cause_utterance_ID:
                            continue
                    elif self.__C.Context_cut == 'static':
                        pass
                    else:
                        raise ValueError("Context_cut not found")

                    # full_question = uttItemstr+'</s>'+q
                    prompt_ids=self.tokenizer.encode(uttItemstr,q)
                
                    token_type_id=[0]*len(prompt_ids)
                    att_mask = [1]*len(prompt_ids)
                 
                    label =[self.common_data.Emotion2index[a] if  self.__C.TASK_TYPE == 'emotion' else self.common_data.YesNo2index[a]]
                    
                    

                    # q_ids = self.tokenizer.encode(q)
                    # a_ids =self.tokenizer.encode(a)
                    # assert len(a_ids)==2
                    # assert a_ids[0] in self.common_data.emotionids or a_ids[0] in self.common_data.YesNoids
                    # qs_ids =q_ids+a_ids
                    # qs_label=[-100]*len(q_ids)+[a_ids[0]]+[-100]
                    # if 'train' not in self.split_name_list:
                    #     qs_ids=q_ids
                    #     qs_label=[-100]*len(q_ids)+[a_ids[0]]+[-100]

                    # prompt_ids = utts_ids + qs_ids
                    # token_type_id=[0]*(len(utts_ids))+[1]*len(qs_ids)
                    # att_mask = [1]*len(prompt_ids)
                    # label=[-100]*(len(utts_ids))+qs_label
                    
                   
                    utts_video_feat = [vi for _,vi in video_feats_dia.items()]
                    qs_video_feat = [video_feats_dia[str(Cause_utterance_ID)]]+[video_feats_dia[str(Emotion_utterance_ID)]]
                    prompt_video_feat = utts_video_feat+qs_video_feat
                    utts_audio_feat = [vi for _,vi in audio_feats_dia.items()]
                    qs_audio_feat = [audio_feats_dia[str(Cause_utterance_ID)]]+[audio_feats_dia[str(Emotion_utterance_ID)]]
                    prompt_audio_feat = utts_audio_feat+qs_audio_feat
                    

                    self.utt_prompt_ids.append(prompt_ids)
                    self.utt_prompt_attention_masks.append(att_mask)
                    self.token_type_ids.append(token_type_id)
                    self.sample_video_feats.append(prompt_video_feat)
                    self.sample_audio_feats.append(prompt_audio_feat)
                    self.gpt2_labels.append(label)
                    key_name = f'dia_{dia_key}_uttEmo_{Emotion_utterance_ID}_uttCas_{Cause_utterance_ID}_index_{i}'
                    self.key_names.append(key_name) 
      
        # self.test_for_prediciton()
     
        max_video_len = max([len(item) for item in self.sample_video_feats])
        max_audio_len = max([len(item) for item in self.sample_audio_feats])
        assert max_video_len == max_audio_len
        self.max_av_len = max_video_len
        max_token_len = max([len(item) for item in self.utt_prompt_ids])
        self.max_prompt_len  = max_token_len # if 'train' in self.split_name_list else max_token_len+2
        print('max_video_len--',max_video_len)
        print('max_token_len--',self.max_prompt_len)
        token_len_distribution = list(set([len(item) for item in self.utt_prompt_ids]))

        


    def test_for_prediciton(self,):
        if self.__C.TASK_TYPE == 'preEmo_precause_pair' or self.__C.TASK_TYPE == 'preEmo_precause_pair_neu':
            assert'test' in self.split_name_list
            preEmo_test_file = self.preEmo_test_file
            precause_test_file = self.precause_test_file
            # is the dia,utt a emotional one
            IsEmotionDict = {} 
            IsCauseDict = {} 
            if preEmo_test_file and precause_test_file:
                with open(self.preEmo_test_file, "r",encoding='utf8') as fp1:
                    preEmo_test_results = json.load(fp1)
                with open(self.precause_test_file, "r",encoding='utf8') as fp2:
                    precause_test_results = json.load(fp2)
                assert len(preEmo_test_results) == len(precause_test_results)
                for emo_result, cause_result in zip(preEmo_test_results,precause_test_results):
                    emo_keyname = emo_result['keyname']
                    emo_prob_label = emo_result['prob_label']
                    cause_keyname = cause_result['keyname']
                    cause_prob_label = cause_result['prob_label']
                    if emo_prob_label in ['Yes','neutral']:
                        Emotion = 'neutral'
                    elif emo_prob_label in ['No']:
                         Emotion = 'not neutral'
                    else:
                        Emotion=emo_prob_label
                    if cause_prob_label in ['Yes']:
                        Cause= 'causal'
                    else:
                        Cause = 'not causal'
                    assert emo_keyname ==  cause_keyname
                    _,dia_key,_,utterance_ID,_,index=emo_keyname.split('_')
                    if dia_key in IsEmotionDict:
                        IsEmotionDict[dia_key][utterance_ID] = Emotion
                    else:
                        IsEmotionDict[dia_key]={}
                        IsEmotionDict[dia_key][utterance_ID] = Emotion

                    if dia_key in IsCauseDict:
                        IsCauseDict[dia_key][utterance_ID] = Cause
                    else:
                        IsCauseDict[dia_key]={}
                        IsCauseDict[dia_key][utterance_ID] = Cause


                        
            for dia_key,dialogue in self.dialogues.items():
                Dia_Emotion_questions = self.Emotion_quetions[dia_key]
                Dia_Cause_questions = self.Cause_questions[dia_key]
                Dia_Emotion_Cause_questions = self.Emotion_Cause_questions[dia_key]
                prompt_dia_texts = self.prompt_texts[dia_key]
            
                video_feats_dia = self.video_feats[dia_key]
                audio_feats_dia = self.audio_feats[dia_key]

                Dia_IsEmotionDict = IsEmotionDict[dia_key] 
                Dia_IsCauseDict = IsCauseDict[dia_key]

                uttItemstr = self.tokenizer.sep_token.join(prompt_dia_texts)
                utts_ids = self.tokenizer.encode(uttItemstr)
                small_size=False
                for i, item in enumerate(Dia_Emotion_Cause_questions):
                    
                    Emotion_utterance_ID = item['Emotion_utterance_ID'] 
                    Cause_utterance_ID = item['Cause_utterance_ID'] 
                    try:
                        Emotion = Dia_IsEmotionDict[str(Emotion_utterance_ID)]
                        Cause = Dia_IsCauseDict[str(Cause_utterance_ID)]
                        small_size = True
                    except:
                        break
                    # Emotion = item['Emotion'] if self.__C.TASK_TYPE == 'AnnoEmo_Annocause_pair' else item['Emotion_neu']
                    # Cause = item['cause_label'] if 'cause_label' in item else None
                    
                    if Emotion!='neutral' and Cause == 'causal':
                       
                        qr=item['question'] 
                        a = item['answer'] 
                        q=qr.format(Emotion,Cause)
                        q_ids = self.tokenizer.encode(q)
                        a_ids =self.tokenizer.encode(a)
                        assert len(a_ids)==2
                        assert a_ids[0] in self.common_data.emotionids or a_ids[0] in self.common_data.YesNoids
                        qs_ids =q_ids+a_ids
                        qs_label=[-100]*len(q_ids)+[a_ids[0]]+[-100]
                        if 'train' not in self.split_name_list:
                            qs_ids=q_ids
                            qs_label=[-100]*len(q_ids)+[a_ids[0]]+[-100]

                        prompt_ids = utts_ids + qs_ids
                        token_type_id=[0]*(len(utts_ids))+[1]*len(qs_ids)
                        att_mask = [1]*len(prompt_ids)
                        label=[-100]*(len(utts_ids))+qs_label
                       

                        utts_video_feat = [vi for _,vi in video_feats_dia.items()]
                        qs_video_feat = [video_feats_dia[str(Cause_utterance_ID)]]+[video_feats_dia[str(Emotion_utterance_ID)]]
                        prompt_video_feat = utts_video_feat+qs_video_feat
                        utts_audio_feat = [vi for _,vi in audio_feats_dia.items()]
                        qs_audio_feat = [audio_feats_dia[str(Cause_utterance_ID)]]+[audio_feats_dia[str(Emotion_utterance_ID)]]
                        prompt_audio_feat = utts_audio_feat+qs_audio_feat
                        

                        self.utt_prompt_ids.append(prompt_ids)
                        self.utt_prompt_attention_masks.append(att_mask)
                        self.token_type_ids.append(token_type_id)
                        self.sample_video_feats.append(prompt_video_feat)
                        self.sample_audio_feats.append(prompt_audio_feat)
                        self.gpt2_labels.append(label)
                        key_name = f'dia_{dia_key}_uttEmo_{Emotion_utterance_ID}_uttCas{Cause_utterance_ID}_index_{i}'
                        self.key_names.append(key_name) 
           
                if small_size:
                    break
                
    
         

    def pad_data(self,):
        self.pad_utt_prompt_ids = []
        self.pad_utt_prompt_attention_masks=[]
        self.pad_token_type_ids = []
        self.pad_video_feats=[]
        self.pad_audio_feats=[]
        self.pad_gpt2_labels = []
      
        for utt_prompt_id,utt_prompt_attention_mask,token_type_id,video_feat,audio_feat,gpt2_label \
            in zip(self.utt_prompt_ids,self.utt_prompt_attention_masks,self.token_type_ids,self.sample_video_feats,self.sample_audio_feats,self.gpt2_labels):
            if  self.tokenizer.padding_side=='right':
                pad_token_id = self.tokenizer.pad_token_id
             
                pad_utt_prompt_id= utt_prompt_id + [pad_token_id]*(self.max_prompt_len-len(utt_prompt_id))
                pad_utt_prompt_attention_mask= utt_prompt_attention_mask +[0]*(self.max_prompt_len-len(utt_prompt_attention_mask)) 
                pad_token_type_id = token_type_id +[0]*(self.max_prompt_len-len(token_type_id))
            
                pad_video_feat=video_feat+[0]*(self.max_av_len-len(video_feat))
                pad_audio_feat=audio_feat+[0]*(self.max_av_len-len(audio_feat))
                # video_feat_ts= torch.tensor(np.concatenate([item[np.newaxis,:] for item in video_feat],0))
                # audio_feat_ts = torch.tensor(np.concatenate([item[np.newaxis,:] for item in audio_feat],0))
                # # print(self.max_video_len,video_feat_ts.shape[1])
                # pad_video_feat = torch.zeros(self.max_av_len,video_feat_ts.shape[1])
                # pad_audio_feat = torch.zeros(self.max_av_len,audio_feat_ts.shape[1])
                # pad_video_feat[:video_feat_ts.shape[0]]=video_feat_ts
                # pad_audio_feat[:audio_feat_ts.shape[0]]=audio_feat_ts

                pad_gpt2_label= gpt2_label
                # pad_gpt2_label= [-100]*(self.max_prompt_len-len(gpt2_label)) + gpt2_label
          
            self.pad_utt_prompt_ids.append(pad_utt_prompt_id)
            self.pad_utt_prompt_attention_masks.append(pad_utt_prompt_attention_mask)
            self.pad_token_type_ids.append(pad_token_type_id)
            self.pad_video_feats.append(pad_video_feat)
            self.pad_audio_feats.append(pad_audio_feat)
            self.pad_gpt2_labels.append(pad_gpt2_label)

            # if len(pad_utt_prompt_id) != self.max_prompt_len or len(pad_utt_prompt_attention_mask) !=self.max_prompt_len\
            #     or len(pad_utt_prompt_attention_mask)!= self.max_prompt_len or len(pad_token_type_id)!=self.max_prompt_len\
            #         or len(pad_gpt2_label) != self.max_prompt_len:


    def get_Cause_prompt(self,vid=None,Emotion_name=None):
        key_name = self.key_names[vid]
        _,dia_key,_,Emotion_utterance_ID,_,Cause_utterance_ID,_,index = key_name.split('_')
        Dia_Emotion_Cause_questions = self.Emotion_Cause_questions[dia_key]
        prompt_dia_texts = self.prompt_texts[dia_key]
        uttItemstr = self.tokenizer.sep_token.join(prompt_dia_texts)
        video_feats_dia = self.video_feats[dia_key]
        audio_feats_dia = self.audio_feats[dia_key]
        utts_ids = self.tokenizer.encode(uttItemstr)
        item = Dia_Emotion_Cause_questions[index]

        q_pref =item['question_prefix'] 
        q_append =item['question_appendix']
        if Emotion_name =='No':
            Emotion='not neutral'
        else:
            Emotion = Emotion_name
        # Emotion = item['Emotion'] if self.__C.TASK_TYPE == 'AnnoEmo_cause_pair' else item['Emotion_neu']
        q=q_pref+Emotion+q_append
        a = item['answer'] 
        Emotion_utterance_ID = item['Emotion_utterance_ID'] 
        Cause_utterance_ID = item['Cause_utterance_ID'] 
        q_ids = self.tokenizer.encode(q)
        a_ids =self.tokenizer.encode(a)
        assert len(a_ids)==2
        assert a_ids[0] in self.common_data.emotionids or a_ids[0] in self.YesNoids
        qs_ids =q_ids+a_ids
        qs_label=[-100]*len(q_ids)+a_ids
        if 'train' not in self.split_name_list:
            qs_ids=q_ids
            qs_label=[-100]*len(q_ids)+a_ids

        prompt_ids = utts_ids + qs_ids
        token_type_id=[0]*(len(utts_ids))+[1]*len(qs_ids)
        att_mask = [1]*len(prompt_ids)
        label=[-100]*(len(utts_ids))+qs_label

        utts_video_feat = [vi for _,vi in video_feats_dia.items()]
        qs_video_feat = [video_feats_dia[Cause_utterance_ID]]+[video_feats_dia[Emotion_utterance_ID]]
        prompt_video_feat = utts_video_feat+qs_video_feat
        utts_audio_feat = [vi for _,vi in audio_feats_dia.items()]
        qs_audio_feat = [audio_feats_dia[Cause_utterance_ID]]+[audio_feats_dia[Emotion_utterance_ID]]
        prompt_audio_feat = utts_audio_feat+qs_audio_feat

        video_blank_idx = self.tokenizer.encode('<videoblankpos>')[0]
        audio_blank_idx = self.tokenizer.encode('<audioblankpos>')[0]
        video_blank_id = torch.tensor([0 if id!=video_blank_idx else 1 for id in prompt_ids])
        audio_blank_id = torch.tensor([0 if id!=audio_blank_idx else 1 for id in prompt_ids])


   
        return prompt_ids, att_mask,token_type_id,prompt_video_feat,prompt_audio_feat,label,video_blank_id,audio_blank_id





    def __getitem__(self, idx):
    # def getitem(self, idx):
        # get question in token ids, image in features,
        # and answer in binary-label vector

        __C = self.__C

        pad_utt_prompt_id =self.pad_utt_prompt_ids[idx]
        pad_utt_prompt_attention_mask =self.pad_utt_prompt_attention_masks[idx]
        token_type_id= self.pad_token_type_ids[idx]
        gpt2_label = self.pad_gpt2_labels[idx]
        video_featIDs = self.pad_video_feats[idx]
        audio_featIDs = self.pad_audio_feats[idx]
        video_feat = self.video_embedding[video_featIDs,:]
        audio_feat = self.audio_embedding[audio_featIDs,:]
        video_feat= torch.tensor(video_feat).float()
        audio_feat=torch.tensor(audio_feat).float()

        # print(video_featIDs,audio_featIDs)
        # print(video_feat.shape,audio_feat.shape)
        # assert 0
        


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

        video_blank_idx = self.tokenizer.encode('<videoblankpos>')[1:-1][0]
        audio_blank_idx = self.tokenizer.encode('<audioblankpos>')[1:-1][0]
     
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


