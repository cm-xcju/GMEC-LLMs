# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Runner that handles the prompting process
# ------------------------------------------------------------------------------ #

from ipdb import set_trace as stop
from configs.task_cfgs import Cfgs
from .utils.data_utils import Qid2Data
from .utils.fancy_pbar import progress, info_column
from pathlib import Path
import yaml
from copy import deepcopy
from datetime import datetime
import argparse
import os
import sys
# sys.path.append(os.getcwd())
from evaluation.ans_punct import prep_ans

import pickle
import json
import time
import math
import random
random.seed(24) #10
import re
# from dotenv import load_dotenv, dotenv_values
# load_dotenv('.env')
# env_configs = dotenv_values()
import httpx as _httpx
# from openai import OpenAI
import openai
# Once you add your API key below, make sure to not share it with anyone! The API key should remain private.



class Runner:
    def __init__(self, __C, evaluater):
        self.__C = __C
        self.evaluater = evaluater
        
        # proxies = {'http://': 'http://127.0.0.1:7890', 'https://': 'http://127.0.0.1:7890'}
        # _http_client = _httpx.Client(proxies=proxies)
        # self.client = OpenAI(
        #     api_key=self.__C,
        #     # base_url="https://api.openai.com/v1/",
        #     http_client=_http_client,
        #     organization=None,

            
        # )
        # client.OPENAI_API_KEY
        openai.api_key = __C.OPENAI_KEY
        openai.api_base = "https://openkey.cloud/v1"
        openai.proxy = {"https": "http://127.0.0.1:7890",
                        "http": "http://127.0.0.1:7890"}
        # openai.proxy = {"https": "http://127.0.0.1:1081",
        #                 "http": "http://127.0.0.1:1081"}

    def gpt3_infer(self, prompt_text, _retry=0):
        # print(prompt_text)
        # exponential backoff
        # if _retry > 0:
        #     print('retrying...')
        #     st = 2 ** _retry
        #     time.sleep(st)

        if self.__C.DEBUG:
            # print(prompt_text)
            time.sleep(0.05)
            return 0, 0
    
        try:

            # print('calling gpt3...')
            response = openai.ChatCompletion.create(
                model=self.__C.MODEL,
                messages=[
                    {"role": "system", "content": self.__C.PROMPT_HEAD},
                    {"role": "user", "content": prompt_text},
                    ],
                temperature=self.__C.TEMPERATURE,
                max_tokens=self.__C.MAX_TOKENS,
                # stop=["\n", "<|endoftext|>"],
                # timeout=20,
            )
            print(response)
            # print('gpt3 called.')
            
        except Exception as e:
            print(type(e), e)
            if str(e) == 'You exceeded your current quota, please check your plan and billing details.':
                exit(1)
            if "maximum context length is 4097 tokens" in str(e):
                new_prompt_text = '===\nContext:'+'===\nContext:'.join(prompt_text.split('===\nContext:')[2:])
                
            return self.gpt3_infer(new_prompt_text, _retry + 1)

        response_txt = response.choices[0].message.content
        total_tokens = response['usage']['total_tokens']
        # print(response_txt)
        # plist = []
        # for ii in range(len(response['choices'][0]['logprobs']['tokens'])):
        #     if response['choices'][0]['logprobs']['tokens'][ii] in ["\n", "<|endoftext|>"]:
        #         break
        #     plist.append(response['choices'][0]['logprobs']['token_logprobs'][ii])
        # prob = math.exp(sum(plist))

        return response_txt, total_tokens

    def sample_make(self, qs_context, question, caption, cands, ans=None):
        line_prefix = self.__C.LINE_PREFIX
        
        if not self.__C.noContext:
            prompt_text = line_prefix + f'Context: \n{qs_context}\n'
        else:
            prompt_text=''
        

        # new_qs = re.sub(r"\d+,Speaker *:","",question)
        # begpos=question.find("utterance")
        # end_pos = question.find(":")
        # new_qs = question[:begpos]+'utterance "'+question[end_pos+2:-14]+'"'+question[-14:]
      
        if self.__C.TASK_TYPE in ['emotion']:
            new_qs ='The emotion of'+ ' "'+question.split("The emotion of ")[1].split(' is ? ')[0]+'" is ?'
        elif self.__C.TASK_TYPE in ['preEmo_precause_pair']:
            new_qs='The'+ ' "'+question[5:].split(' is the cause of the  ')[0]+'" is the cause of the "' +question[5:].split(' is the cause of the  ')[1][:-3]+'"?'

      
        
        if not self.__C.noContext:
            if self.__C.useCaption!='None':
                prompt_text += line_prefix + f'Image Caption: {caption}\n'
          
            prompt_text += line_prefix + f'Question: {new_qs}\n'
        else:
            if self.__C.useCaption !='None':
                prompt_text += line_prefix + f'Context: {caption}\n'
            prompt_text += line_prefix + f'Question: {new_qs}\n'
        cands_with_conf = [
            f'{cand["answer"]} ({cand["confidence"]:.2f})' for cand in cands]
        cands = ', '.join(cands_with_conf)
        prompt_text += line_prefix + f'Candidates: {cands}\n'
        prompt_text += line_prefix + 'Answer:'
        if ans is not None:
            prompt_text += f' {ans}'
        return prompt_text

    def get_context(self, example_qids):
        # making context text for one testing input
        # prompt_text = self.__C.PROMPT_HEAD
        prompt_text = ''
        examples = []
        for key in example_qids:
            ques = self.trainset.get_question(key)
            qs_context = self.trainset.get_qs_context(key)
         
            caption = self.trainset.get_caption(key)
            cands = self.trainset.get_candidate(key)
            # here not should use this
            # gt_ans = self.trainset.get_true_answer(key)
            gt_ans = self.trainset.get_most_answer(key)
            examples.append((ques, caption, cands, gt_ans))
            prompt_text += self.sample_make(qs_context,
                                            ques, caption, cands, ans=gt_ans)
            prompt_text += '\n\n'
        return prompt_text

    def save_json_data(self, data, tgt_path):
        with open(tgt_path, "w", encoding='utf8') as fp:
            fp.write(json.dumps(data, indent=4, ensure_ascii=False))

    def sample_small_size(self, dataset, size):
        key_list = dataset.qid_to_data.keys()

        key_slice = random.sample(key_list, size)

        dataset.qid_to_data = {k: dataset.qid_to_data[k] for k in key_slice}
        dataset.data_size = size

    def run(self):
        # where logs will be saved
        Path(self.__C.LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(self.__C.LOG_PATH, 'w') as f:
            f.write(str(self.__C) + '\n')
        # where results will be saved
        Path(self.__C.RESULT_DIR).mkdir(parents=True, exist_ok=True)

        self.cache = {}
        self.cache_file_path = os.path.join(
            self.__C.RESULT_DIR,
            'cache.json'
        )
        # if self.__C.RESUME:
        if os.path.exists(self.cache_file_path):
            self.cache = json.load(open(self.cache_file_path, 'r'))
      

        print('Note that the accuracies printed before final evaluation (the last printed one) are rough, just for checking if the process is normal!!!\n')
        self.trainset = Qid2Data(
            self.__C,
            self.__C.TRAIN_SPLITS,
            True
        )
        self.valset = Qid2Data(
            self.__C,
            self.__C.TEST_SPLITS,
            self.__C.EVAL_NOW,
            json.load(open(self.__C.EXAMPLES_PATH, 'r'))
        )
       
        # if 'aok' in self.__C.TASK:
        #     from evaluation.aokvqa_evaluate import AOKEvaluater as Evaluater
        # else:
        #     from evaluation.okvqa_evaluate import OKEvaluater as Evaluater
        # evaluater = Evaluater(
        #     self.valset.annotation_path,
        #     self.valset.question_path
        # )

        infer_times = self.__C.T_INFER
        N_inctx = self.__C.N_EXAMPLES
        all_total_tokens = 0
        no_match_num = 0
        print()
        rand_num =  len(self.valset.qid_to_data.keys())
        
        self.sample_small_size(self.valset, rand_num)
        # filename="outputs\\results\keys_test\emonu_BL_400_e1f3_noC.json"
        # test_keys=json.load(open(filename,'r',encoding='utf-8'))
        # for qid in progress.track(self.valset.dia_utt_qs_data.keys(), description="Working...  "):
        total_prompts_num = 0
        total_noprompts_num = 0
        infer_nums = 0
        for qid in self.valset.qid_to_data.keys():
            infer_nums+=1
        # for qid in test_keys:
            if qid in self.cache:
                cache_item = self.cache[qid]
                answer = cache_item['answer']
                most_answer = cache_item['most_answer']
                true_ans = cache_item['true_ans']
                self.evaluater.chatgpt_add(qid, answer, most_answer, true_ans)
                continue
            
            question = self.valset.get_question(qid)
            qs_context = self.valset.get_qs_context(qid)
            caption = self.valset.get_caption(qid)
            candidate = self.valset.get_candidate(qid)
            true_ans = self.valset.get_true_answer(qid)
            most_answer = self.valset.get_most_answer(qid)

            if candidate[0]['confidence']< self.__C.confidence_threshold:
                total_prompts_num+=1
            else:
                total_noprompts_num+=1
            
            # continue


            prompt_query = self.sample_make(
                qs_context, question, caption, candidate)

            # print(prompt_query)
            example_qids = self.valset.get_similar_qids(
                qid, k=infer_times * N_inctx)
            random.shuffle(example_qids)

            # no example
            # example_qids = []

            prompt_info_list = []
            ans_pool = {}
            # multi-times infer
            # Please answer the question according to the context and candidate answers. Each candidate answer is associated with a confidence score within a bracket. The true answer must be included in the candidate answers
            # Please answer the question according to the context and candidate answers. Each candidate answer is associated with a confidence score within a bracket. The true answer may not be included in the candidate answers.
            for t in range(infer_times):
                # print(f'Infer {t}...')
            
                prompt_in_ctx = self.get_context(
                    example_qids[(N_inctx * t):(N_inctx * t + N_inctx)])
                prompt_text = prompt_in_ctx + prompt_query
                print(self.__C.PROMPT_HEAD+'\n\n'+prompt_text)
              
                jump_flag=False
                if self.__C.TASK_TYPE in ['preEmo_precause_pair']:
                    _,dia_id,_,EmoId,_,CasId,*_ = qid.split('_')
                    if int(EmoId)-int(CasId) >4:
                        jump_flag=True
                        
                   
                if candidate[0]['confidence']<self.__C.confidence_threshold and not jump_flag:
                    gen_text, total_tokens = self.gpt3_infer(prompt_text)
                else:
                    gen_text=candidate[0]['answer']
                    total_tokens=0
                all_total_tokens += total_tokens

                
                # answer_r = self.evaluater.prep_ans(gen_text)
                answer_r = prep_ans(gen_text)
                if self.__C.TASK_TYPE in ['emotion']:
                    if 'neutral' in answer_r:
                        ans='neutral'
                    elif 'anger' in answer_r:
                        ans='anger'
                    elif 'surprise' in answer_r:
                        ans='surprise'
                    elif 'sadness' in answer_r or 'sad' in answer_r:
                        ans='sadness'
                    elif 'joy' in answer_r:
                        ans='joy'
                    elif 'disgust' in answer_r:
                        ans='disgust'
                    elif 'fear' in answer_r:
                        ans='fear'
                    else:
                        ans = most_answer
                        no_match_num += 1
                   
                else:
                    if 'no' in answer_r:
                        ans = 'No'
                    elif 'yes' in answer_r:
                        ans = 'Yes'
                    else:
                        ans = most_answer
                        no_match_num += 1
                

                    # raise ValueError('can not find the output')

                # if ans != '':
                #     ans_pool[ans] = ans_pool.get(ans, 0.) + gen_prob
                ans_pool[ans]=ans_pool.get(ans,0)+1
                prompt_info = {
                    'prompt': prompt_text,
                    'gpt_answer': gen_text,
                    'answer': ans,
                }
                prompt_info_list.append(prompt_info)
                # time.sleep(self.__C.SLEEP_PER_INFER)

            # vote
         
            answer=sorted(ans_pool.items(),key=lambda x:x[1],reverse=True)[0][0]
            
            # answer = self.valset.get_topk_answers(qid)[0]['answer']
            # if len(ans_pool) == 0:
            #     answer = self.valset.get_topk_answers(qid, 1)[0]['answer']
            # else:
            #     answer = sorted(ans_pool.items(), key=lambda x: x[1], reverse=True)[0][0]

            self.evaluater.chatgpt_add(qid, answer, most_answer, true_ans)
            self.cache[qid] = {
                'question_id': qid,
                'answer': answer,
                'most_answer': most_answer,
                'true_ans': true_ans,
                'prompt_info': prompt_info_list
            }
            if infer_nums%10==0:
                self.save_json_data(self.cache, self.cache_file_path)
            # json.dump(self.cache, open(self.cache_file_path, 'w'))
            print(
                f'all_total_tokens: {all_total_tokens},  no_match_num: {no_match_num}')
            # ll = len(self.cache)
            # if self.__C.EVAL_NOW and not self.__C.DEBUG:
            #     if ll > 21 and ll % 10 == 0:
            #         rt_accuracy = self.valset.rt_evaluate(self.cache.values())
            #         info_column.info = f'Acc: {rt_accuracy}'
     
        self.evaluater.prompt_save(self.__C.RESULT_PATH)
        if self.__C.EVAL_NOW:
            with open(self.__C.LOG_PATH, 'a+') as logfile:
                self.evaluater.prompt_evaluation(logfile,self.__C)
                print(
                    f'total_tokens:{all_total_tokens},  no_match_num: {no_match_num}  rand_num: {rand_num}' + '\n', file=logfile)
                
                print(f'total_prompts_num:{total_prompts_num}  total_noprompts_num: {total_noprompts_num}', file=logfile)


def prompt_login_args(parser):
    parser.add_argument('--debug', dest='DEBUG',
                        help='debug mode', action='store_true')
    parser.add_argument('--resume', dest='RESUME',
                        help='resume previous run', action='store_true')
    parser.add_argument('--task', dest='TASK',
                        help='task name', type=str, required=True)
    parser.add_argument('--version', dest='VERSION',
                        help='version name', type=str, required=True)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file',
                        type=str, default='configs/prompt.yml')
    parser.add_argument('--examples_path', dest='EXAMPLES_PATH',
                        help='answer-aware example file path, default: "assets/answer_aware_examples_for_ok.json"', type=str, default=None)
    parser.add_argument('--candidates_path', dest='CANDIDATES_PATH',
                        help='candidates file path, default: "assets/candidates_for_ok.json"', type=str, default=None)
    parser.add_argument('--captions_path', dest='CAPTIONS_PATH',
                        help='captions file path, default: "assets/captions_for_ok.json"', type=str, default=None)
    parser.add_argument('--openai_key', dest='OPENAI_KEY',
                        help='openai api key', type=str, default=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Heuristics-enhanced Prompting')
    prompt_login_args(parser)
    args = parser.parse_args()
    __C = Cfgs(args)
    with open(args.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    __C.override_from_dict(yaml_dict)
    print(__C)

    runner = Runner(__C)
    runner.run()
