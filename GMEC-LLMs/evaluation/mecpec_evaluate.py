import json
# from evaluation.mec_utils.mec import MEC
# from evaluation.mec_utils.mecEval import MECEval
# from .ans_punct import prep_ans as _prep_ans
import argparse
from pdb import set_trace as stop
import numpy as np
import torch
class mecEvaluater:
    def __init__(self, __C):
        self.result_file = []
        self.result_model_file = []
        self.result_prompt_file = []
        self.result_path = None
        # when one emotion have more than one id, only use the first id for probalition calcualtion 
        # self.Emotion2Idxs=
        # self.Emotion_idxs_one=[ids[0] for key, ids in self.Emotion2Idxs.items()]
        # self.Emotion_idxs_all=[]
        # for key, ids in self.Emotion2Idxs.items():
        #     self.Emotion_idxs_all+=ids
        # self.Emotion_names=[key for key,_ in self.Emotion2Idxs.items()]
    
    def clear(self,):
        self.result_file=[]

    def add(self, vid, logits_lm,res=None):
     
        distribution, prob_label_id,prob_label, gpt2_label_id,true_label,prob_raw_id,prob_raw_label = self.get_logit(vid,logits_lm)
        res_id, generate_label,res,raw_label = self.get_label(res)
        keyname = self.get_keyname(vid)
        self.result_file.append({
            'data_id': vid,
            'keyname':keyname,
            'prob': distribution,
            'prob_label_id':prob_label_id,
            'prob_label':prob_label,
            'prob_raw_id':prob_raw_id,
            'prob_raw_label':prob_raw_label,
            'true_label_id':gpt2_label_id,
            'true_label':true_label,
            'label_generate_id':res_id,
            'label_generate':generate_label,
            'generate_id':res,
            'generate_decode':raw_label,

        })
        
    def save(self, result_path: str):
        self.result_path = result_path
        with open(self.result_path, "w",encoding='utf8') as fp:
            fp.write(json.dumps(self.result_file, indent=4, ensure_ascii=False))
        # json.dump(self.result_file, open(self.result_path, 'w'))

    def init_2(self,common_data=None,dataset=None,task_type=None):
        self.Emotion2Idxs=common_data.Emotion2Idxs
     
        self.Idxs2Emotion={value:key for key,value in common_data.Emotion2Idxs.items()}
        self.emotionids=common_data.emotionids
        self.YesNo2Idxs=common_data.YesNo2Idxs
        self.Idxs2YesNo ={value:key for key,value in common_data.YesNo2Idxs.items()}
        self.YesNoids=common_data.YesNoids
        self.Emotion2index=common_data.Emotion2index
        self.Index2Emotion = {value:key for key,value in common_data.Emotion2index.items()}
        self.YesNo2index=common_data.YesNo2index
        self.Index2YesNo = {value:key for key,value in common_data.YesNo2index.items()}
        self.data_type = dataset.split_name_list[0]
        self.task_type = task_type
        self.common_data=common_data
        self.dataset=dataset
        self.tokenizer=self.common_data.tokenizer
        self.Cause2Idxs={}
        self.Cause_idxs_one=[]
        for i in range(30):
            cas_id = self.tokenizer.encode(str(i+1))[1:-1]
            assert len(cas_id) == 1
            self.Cause2Idxs[str(i+1)]= cas_id[0]
            self.Cause_idxs_one +=cas_id
    def get_keyname(self,vid):
        assert vid < len(self.dataset)
        keyname = self.dataset.key_names[vid]
        return keyname
    # def prep_ans(self,answers):
    #     return _prep_ans(answers)

    def get_logit(self,vid,logits_lm):
        # addd the out of logits_lm
      
        prob_raw_id = None
        prob_raw_label =None
        gpt2_label_id = self.dataset.gpt2_labels[vid][0]
        if self.task_type == 'emotion':
            # distribution_tensor = logits_lm[self.emotionids] #.softmax(-1)
            distribution = logits_lm.numpy().tolist()
            prob_label_id= torch.argmax(logits_lm).numpy().tolist()
            # prob_label_id= self.common_data.Index2emotion[pos]
            prob_label= self.Index2Emotion[prob_label_id]
            true_label =self.Index2Emotion[gpt2_label_id]
        elif self.task_type == 'emotion_neu' or self.task_type == 'cause'\
            or self.task_type == 'AnnoEmo_Annocause_pair' or self.task_type == 'AnnoEmo_Annocause_pair_neu'\
            or self.task_type == 'AnnoEmo_precause_pair' or self.task_type == 'AnnoEmo_precause_pair_neu'\
            or self.task_type == 'preEmo_precause_pair' :
               
                distribution = logits_lm.numpy().tolist()
                prob_label_id= torch.argmax(logits_lm).numpy().tolist()
                # prob_label_id=self.YesNoids[pos]
                prob_label = self.Index2YesNo[prob_label_id]
                true_label =self.Index2YesNo[gpt2_label_id]
        else:
            
            raise ValueError('task_type not found')
        
        return distribution, prob_label_id,prob_label, gpt2_label_id,true_label,prob_raw_id,prob_raw_label

    
     
    def get_label(self,res):
        if res==None:
            return None,None,None,None
        if self.task_type == 'emotion':
            if res in self.emotionids:
                res_id = res
                generate_label = self.Idxs2Emotion[res]
            else:
                res_id = self.Emotion2Idxs['neutral']
                generate_label='neutral'
        elif self.task_type == 'emotion_neu':
            if res in self.YesNoids:
                res_id = res
                generate_label = self.Idxs2YesNo[res]
            else:
                res_id = self.YesNo2Idxs['Yes']
                generate_label='Yes'
        elif self.task_type == 'cause' or self.task_type == 'AnnoEmo_cause_pair' or self.task_type == 'AnnoEmo_cause_pair_neu'\
            or self.task_type == 'emo_cause_pair' or self.task_type == 'emo_cause_pair_neu'\
            or self.task_type == 'AnnoEmo_Annocause_pair' or self.task_type == 'AnnoEmo_Annocause_pair_neu':
            if res in self.YesNoids:
                res_id = res
                generate_label = self.Idxs2YesNo[res]
            else:
                res_id = self.YesNo2Idxs['No']
                generate_label='No'
        raw_label = self.tokenizer.decode([res])
        return res_id, generate_label,res,raw_label
    
    # for gpt3 evaluation
    def chatgpt_add(self,qid, answer,most_answer,true_ans):
        self.result_prompt_file.append({
            'keyname':qid,
            'prob_label':answer,
            'true_label':true_ans,
           
        })
        self.result_model_file.append({
            'keyname':qid,
            'prob_label':most_answer,
            'true_label':true_ans,
        })
    def prompt_evaluation(self,logfile=None,__C=None):
        self.task_type=__C.TASK_TYPE
        self.Emotion2Idxs = {
            'neutral': 29797,
            'anger': 2564,
            'surprise': 50263,
            'sadness': 50265,
            'joy': 2633,
            'disgust': 50264,
            'fear': 50266,
        }
        self.emotionids = [item for _, item in self.Emotion2Idxs.items()]

        self.YesNo2Idxs = {
            'Yes': 5297,
            'No': 2949,
        }
        self.YesNoids = [item for _, item in self.YesNo2Idxs.items()]

        print('evalute the prompt output' + '\n', file=logfile)
        self.result_file=self.result_prompt_file
        self.evaluate(logfile)
        
        print('evalute the raw output' + '\n', file=logfile)
        self.result_file=self.result_model_file
        self.evaluate(logfile)
    def prompt_save(self,result_path: str):
        self.result_path = result_path
        model_result_path = result_path[:-5]+'_model'+result_path[-5:]
        prompt_result_path = result_path[:-5]+'_prompt'+result_path[-5:]
        with open(model_result_path, "w",encoding='utf8') as fp:
            fp.write(json.dumps(self.result_model_file, indent=4, ensure_ascii=False))
        with open(prompt_result_path, "w",encoding='utf8') as fp:
            fp.write(json.dumps(self.result_prompt_file, indent=4, ensure_ascii=False))

    def evaluate(self, logfile=None):
        # assert self.result_path is not None, "Please save the result file first."
        nums_text=None
    
        if self.task_type == 'emotion':
            conf_mat = np.zeros([7,7])
            Emotion2cateid = {key: i for i,(key,value) in enumerate(self.Emotion2Idxs.items())}
            for result in self.result_file:
               
                prob_label = result['prob_label']
                prob_id = Emotion2cateid[prob_label]
                true_label = result['true_label']
                true_id = Emotion2cateid[true_label]
                conf_mat[true_id][prob_id]+=1
            p = np.diagonal( conf_mat / np.reshape(np.sum(conf_mat, axis = 0) + 1e-8, [1,7]) )
            r = np.diagonal( conf_mat / np.reshape(np.sum(conf_mat, axis = 1) + 1e-8, [7,1]) )
            f = 2*p*r/(p+r+1e-8)
            weight0 = np.sum(conf_mat, axis = 1) / np.sum(conf_mat)
            w_avg_f = np.sum(f * weight0)
            w_avg_r = np.sum(r * weight0)
            w_avg_p = np.sum(p * weight0)

            # report the weighted average F1 score of the four main 
            # emotion categories except Disgust and Fear.
            # 不考虑占比较小的disgust/fear
            idx = [2,3,4,5]
            weight1 = weight0[idx]
            weight = weight1 / np.sum(weight1)
            
            w_avg_p_part = np.sum(p[idx] * weight)
            w_avg_r_part = np.sum(r[idx] * weight)
            w_avg_f_part = np.sum(f[idx] * weight) # 4个情绪的加权f1
            # 'np_p':p,'npr':r,'npf':f,
            scores =  {'prec':round(w_avg_p,4),'recall':round(w_avg_r,4),'f1':round(w_avg_f,4), \
                    'w_avg_p_part':w_avg_p_part,'w_avg_r_part':w_avg_r_part,'w_avg_f_part':w_avg_f_part}
        elif self.task_type=='emotion_neu' :
            pred_num, acc_num, true_num = 0, 0, 0
            acc_num_all = 0
            total_num = len(self.result_file)
            YesNo2cateid = {key: i for i,(key,value) in enumerate(self.YesNo2Idxs.items())}
            # yes 0, No 1
            for result in self.result_file:
                prob_label = result['prob_label']
                prob_id = YesNo2cateid[prob_label]
                true_label = result['true_label']
                true_id = YesNo2cateid[true_label]
                if prob_id==1:
                    pred_num+=1
                if true_id==1:
                    true_num+=1
                if prob_id==1 and true_id==1:
                    acc_num+=1
                #  for acc
                if prob_id == true_id:
                    acc_num_all += 1
            accuracy = acc_num_all / total_num
            p, r = acc_num/(pred_num+1e-8), acc_num/(true_num+1e-8)
            f = 2*p*r/(p+r+1e-8)
            nums_text = (f'true labels {true_num}, pred_labels {pred_num}, acc_num {acc_num}')
            scores = {'prec':round(p,4), 'recall':round(r,4), 'f1':round(f,4),'accuracy':round(accuracy,4)}
 
        elif  self.task_type=='cause' \
            or self.task_type == 'AnnoEmo_Annocause_pair' or self.task_type == 'AnnoEmo_Annocause_pair_neu'\
            or self.task_type == 'AnnoEmo_precause_pair' or self.task_type == 'AnnoEmo_precause_pair_neu'\
            or self.task_type == 'preEmo_precause_pair':
            pred_num, acc_num, true_num = 0, 0, 0
            acc_num_all = 0
            total_num = len(self.result_file)
            YesNo2cateid = {key: i for i,(key,value) in enumerate(self.YesNo2Idxs.items())}
            # yes 0, No 1
           
            for result in self.result_file:
                prob_label = result['prob_label']
                prob_id = YesNo2cateid[prob_label]
                true_label = result['true_label']
                true_id = YesNo2cateid[true_label]
                if prob_id==0:
                    pred_num+=1
                if true_id==0:
                    true_num+=1
                if prob_id==0 and true_id==0:
                    acc_num+=1
                 # for acc
                if prob_id == true_id:
                    acc_num_all += 1
            accuracy = acc_num_all / total_num
            p, r = acc_num/(pred_num+1e-8), acc_num/(true_num+1e-8)
            f = 2*p*r/(p+r+1e-8)
            scores = {'prec':round(p,4), 'recall':round(r,4), 'f1':round(f,4),'accuracy':round(accuracy,4)}
            nums_text = (f'true labels {true_num}, pred_labels {pred_num}, acc_num {acc_num}')
        

        else:
            scores= None

        # elif self.task_type == 'emo_cause_pair' :
        #     pred_num, acc_num, true_num = 0, 0, 0
        #     Emotion2cateid = {key: i for i,(key,value) in enumerate(self.Emotion2Idxs.items())}
        #     YesNo2cateid = {key: i for i,(key,value) in enumerate(self.YesNo2Idxs.items())}
        #     conf_mat = np.zeros([7,7])
        #     for result in self.result_file:
        #         prob_label = result['prob_label']
        #         prob_id = Emotion2cateid[prob_label]
        #         true_label = result['true_label']
        #         true_id = Emotion2cateid[true_label]

        #         if 'Cause_prob_label' in result:
        #             Cause_prob_label = result['Cause_prob_label']
        #             cause_prob_id = YesNo2cateid[Cause_prob_label]
        #             Cause_true_label = result['Cause_true_label']
        #             cause_true_id = YesNo2cateid[Cause_true_label]
        #         if prob_label !='neutral':
        #             pred_num+=1
        #         if true_label !='neutral':
        #             true_num+=1
        #         if prob_label == true_label and true_label!='neutral':
        #             assert 'Cause_prob_label' in result
        #             # if Cause_prob_label == 
        #             pass

        # elif  self.task_type == 'emo_cause_pair_neu':
        #     pass
        



        # eval_str = _evaluate(self.annotation_path, self.question_path, self.result_path)
        print()
        # print(eval_str)
        if logfile is not None:
            print(str(scores) + '\n', file=logfile)
           
            if nums_text:
                print(str(nums_text) + '\n', file=logfile)
        return scores



def _evaluate_for_preEmo_precause_pair():# for emotion_
    Emotion_Cause_questions_test_path = '../question_files/Emotion_Cause_questions_test.json'
    Emotion_result_file_path = '../outputs/results/trainin_emonu_5e5_optim2_8/result_20230609112821_emotion_neu_17_test.json' 
    Cause_result_file_path = '../outputs/results/trainin_cau_5e5_optim2_8/result_20230610023236_cause_3_test.json'
    # AnnoEmotion_AnnoCause_result_path = '../outputs/results/trainin_aeac_5e5_optim2_8/result_20230610174046_AnnoEmo_Annocause_pair_8_test.json'
    AnnoEmotion_AnnoCause_result_path = '../outputs/results/trainin_aeac_5e5_optim2_8/result_20230617233131_AnnoEmo_Annocause_pair_neu_onlytest.json'
    emotion_cate='neu'
    # preEmo_precause_pair_path = 

    with open(Emotion_Cause_questions_test_path, "r",encoding='utf8') as fp1:
        Emotion_Cause_questions = json.load(fp1)
    with open(Emotion_result_file_path, "r",encoding='utf8') as fp1:
        Emotion_results = json.load(fp1)
    with open(Cause_result_file_path, "r",encoding='utf8') as fp1:
        Cause_results = json.load(fp1)
    with open(AnnoEmotion_AnnoCause_result_path, "r",encoding='utf8') as fp1:
        AnnoEmotion_AnnoCause_results = json.load(fp1)
    # pair_id_all = [dia_id,emoId,CasId,true_emotion ]
    # pair_id = [dia_id,emoId,CasId,true_emotion ] emoId in emotion_list and CasId in cause list
    # 1 get the pair_id_all
    pair_id_all=[]
    pair_id=[]

    for diaId,item in Emotion_Cause_questions.items():
        for emo_cas_p in item:
            emoId=emo_cas_p['Emotion_utterance_ID']
            casId=emo_cas_p['Cause_utterance_ID']
            Emotion=emo_cas_p['Emotion']
            Emotion_neu=emo_cas_p['Emotion_neu']
            answer=emo_cas_p['answer'].replace('<|endoftext|>','').strip()
            if answer =='Yes':
                assert Emotion !='neutral'
                if emotion_cate in ['neu']:
                    pair_id_all.append([diaId,str(emoId),str(casId),'No'])
                else:
                    pair_id_all.append([diaId,str(emoId),str(casId),Emotion])

        
    pair_emo_pre=[]
    pair_cas_pre=[]
    pair_emo_true=[]
    pair_cas_true=[]

    for emoRes,casRes in zip(Emotion_results,Cause_results):
        emo_keyname = emoRes['keyname']
        
        _, diaId, _, emoId,*_ =emo_keyname.split('_')
        cas_keyname = casRes['keyname']
        _, casdiaId, _, casId,*_=cas_keyname.split('_')
        assert diaId == casdiaId and emoId==casId
        pro_emo=emoRes['prob_label']
        true_emo = emoRes['true_label']
        pro_cas=casRes['prob_label']
        true_cas = casRes['true_label']
        if pro_emo not in ['Yes','neutral']:
            pair_emo_pre.append([diaId,emoId,pro_emo])
        if true_emo not in ['Yes','neutral']:
            pair_emo_true.append([diaId,emoId,true_emo])
      
        
        if pro_cas == 'Yes':
            pair_cas_pre.append([diaId,casId])
        if true_cas == 'Yes':
            pair_cas_true.append([diaId,casId])


    pair_id_pro=[]
    for emop in pair_emo_pre:
        diaId,emoid,emo=emop
        for i in range(35):
            
            if [diaId,str(i)] in pair_cas_pre:
                pair_id_pro.append([diaId,emoid,str(i),emo])
            
    pair_id_true=[]
    for emop in pair_emo_true:
        diaId,emoid,emo=emop
        for i in range(35):
            
            if [diaId,str(i)] in pair_cas_true:
                pair_id_true.append([diaId,emoid,str(i),emo])
            

 



    pair_emo_pre_wo=[item[:2]for item in pair_emo_pre]
    pred_pairs=[]
    for aeac in AnnoEmotion_AnnoCause_results:
        keyname = aeac['keyname']
        prob_label = aeac['prob_label']

        _,diaId,_,emoId,_,casId,*_=keyname.split('_')
        # casId=casstr.replace('uttCas','')
        if not casId.isdigit():
            stop()
            raise ValueError('its not a digit')
        if prob_label =='Yes':
            if [diaId,emoId] in pair_emo_pre_wo and [diaId,casId] in pair_cas_pre:
                index = pair_emo_pre_wo.index([diaId,emoId])
                emo = pair_emo_pre[index][2]
                pred_pairs.append([diaId,emoId,casId,emo])


    # for raw_ pair_id_filtered


    ### evaluation
    def cal_prf(pair_id_all, pair_id):
        acc_num, true_num, pred_num = 0, len(pair_id_all), len(pair_id)

        for p in pair_id:
            if p in pair_id_all:
                # print(p)
                acc_num += 1
               
        p, r = acc_num/(pred_num+1e-8), acc_num/(true_num+1e-8)
        f1 = 2*p*r/(p+r+1e-8)

        return [p, r, f1]
   
    p,r,f1=cal_prf(pair_id_all,pred_pairs)
    ec_p,ec_r,ec_f1=cal_prf(pair_id_all,pair_id_pro)
    emof = cal_prf(pair_emo_true,pair_emo_pre)
    casf = cal_prf(pair_cas_true,pair_cas_pre)
    print(f'emotion: {emof}, casf: {casf}')
    return {'p':round(p,4),'r':round(r,4),'f1':round(f1,4),'ec_p':round(ec_p,4),'ec_r':round(ec_r,4),'ec_f1':round(ec_f1,4)}
    

def _evaluate_for_emo_pepc():# for emotion_pepc
    def get_emo_result(emo_results,emo_keyname):
        for emo in emo_results:
            keyname = emo['keyname']
            if emo_keyname==keyname:
                return emo
        raise ValueError('emoid not found')
    prefix='../outputs/results/'
    pepc_result_paths=[
        
        "bartLmn_pepc_clip_3_TAV/result_20230721103301_preEmo_precause_pair_4_test.json",
        "bartLmn_pepc_clip_realtime_3_TAV/result_20230724192454_preEmo_precause_pair_3_test.json",

    ]
    emo_result_paths=[
        "bartLmn_emo_clip_1/result_20230711173710_emotion_15_test.json",
        "bartLmn_emo_clip_realtime_4_TAV/result_20230802110042_emotion_13_test.json",

    ]
    idx=1
    pepc_result_path=prefix+pepc_result_paths[idx]
    emo_result_path=prefix+emo_result_paths[idx]
    with open(pepc_result_path, "r",encoding='utf8') as fp1:
        pepc_results = json.load(fp1)
    with open(emo_result_path, "r",encoding='utf8') as fp1:
        emo_results = json.load(fp1)
    
    pred_num, acc_num, true_num = [[0]*7 for i in range(3)]
    acc_num_all = 0
    total_num = len(pepc_results)
    YesNo2Idxs ={
            'Yes':5297,
            'No':2949,
        }
    YesNo2cateid = {key: i for i,(key,value) in enumerate(YesNo2Idxs.items())}
    Emotion2Idxs={
            'neutral': 29797,
            'anger':2564,
            'surprise':50263,
            'sadness':50265,
            'joy':2633,
            'disgust':50264,
            'fear':50266,
        }
    Emotion2cateid = {key: i for i,(key,value) in enumerate(Emotion2Idxs.items())}
    # conf_mat = np.zeros([7,7])
    for pepc in pepc_results:
        pepc_keyname = pepc['keyname']
        _, diaId, _, emoId,_,casId,*_ =pepc_keyname.split('_')
        prob_label = pepc['prob_label']
        true_label = pepc['true_label']
        emo_keyname=f'dia_{diaId}_utt_{emoId}_index_{int(emoId)-1}'
        emo_result= get_emo_result(emo_results,emo_keyname)
        emo_prob_label = emo_result['prob_label']
        emo_true_label = emo_result['true_label']

        if emo_prob_label!='neutral' and prob_label=='Yes' and emo_true_label == emo_prob_label and prob_label==true_label: 
            acc_num[Emotion2cateid[emo_true_label]]+=1
        if emo_prob_label!='neutral' and prob_label=='Yes':
            pred_num[Emotion2cateid[emo_prob_label]]+=1
        
        if emo_true_label!='neutral' and true_label=='Yes':
            true_num[Emotion2cateid[emo_true_label]]+=1
        
        

    p = [round(a/(p+1e-8),4) for a, p in zip(acc_num,pred_num)] 
    r = [round(a/(t+1e-8),4) for a, t in zip(acc_num,true_num)]
    f = [round(2*p1*r1/(p1+r1+1e-8),4) for p1, r1 in zip(p,r)]
    weight4=[t/sum(true_num[1:5])for t in true_num[1:5]]
    f4=sum([fw*w for fw,w in zip(f[1:5],weight4) ])
    weight6=[t/sum(true_num[1:])for t in true_num[1:]]
    f6=sum([fw*w for fw,w in zip(f[1:],weight6) ])


    
    # w4_idx=[1,2,3,4]
    # w4_acc_num=sum([acc_num[i] for i in w4_idx])
    # w4_true_num=sum([true_num[i] for i in w4_idx])
    # p4, r4 = w4_acc_num/(w4_true_num+1e-8), w4_acc_num/(w4_true_num+1e-8)
    # f4 = 2*p4*r4/(p4+r4+1e-8)
    # w6_idx=[1,2,3,4,5,6]
    # w6_acc_num=sum([acc_num[i] for i in w6_idx])
    # w6_true_num=sum([true_num[i] for i in w6_idx])
    # p6, r6 = w6_acc_num/(w6_true_num+1e-8), w6_acc_num/(w6_true_num+1e-8)
    # f6 = 2*p6*r6/(p6+r6+1e-8)

    result_dict={
        'Emotion2cateid':Emotion2cateid,
        # 'p':p[1:],
        # 'r':r[1:],
        'f':f[1:],
        'f4':round(f4,4),
        'f6':round(f6,4),

    }
    print(result_dict)

def evaluate_for_prompt_emo_pepc():
    prompt_pepc_model_result_paths = ['outputs/results/prompt_bartLmn_pepc_clip_realtime_3_TAV_e20i1_MinGPT4_2/result_20240111184043_model.json',
                                      'outputs/results/prompt_bartLmn_pepc_clip_realtime_3_TAV_e20i1_3/result_20240112003438_model.json'
                                      ]
    prompt_pepc_prompt_result_paths = ['outputs/results/prompt_bartLmn_pepc_clip_realtime_3_TAV_e20i1_MinGPT4_2/result_20240111184043_prompt.json',
                                       'outputs/results/prompt_bartLmn_pepc_clip_realtime_3_TAV_e20i1_3/result_20240112003438_prompt.json'
                                       ]
    prompt_emo_model_result_paths = ['outputs/results/prompt_bartLmn_emo_clip_realtime_3_TAV_e20i1_MinGPT4_2/result_20240112230013_model.json',
                                     'outputs/results/prompt_bartLmn_emo_clip_realtime_3_TAV_e20i1/result_20240111233048_model.json'
                                     ]
    prompt_emo_prompt_result_paths = ['outputs/results/prompt_bartLmn_emo_clip_realtime_3_TAV_e20i1_MinGPT4_2/result_20240112230013_prompt.json',
                                      'outputs/results/prompt_bartLmn_emo_clip_realtime_3_TAV_e20i1/result_20240111233048_prompt.json'
                                      ]
    test_id = 1
    with open(prompt_pepc_model_result_paths[test_id], "r",encoding='utf8') as fp1:
        prompt_pepc_model_results = json.load(fp1)
    with open(prompt_pepc_prompt_result_paths[test_id], "r",encoding='utf8') as fp1:
        prompt_pepc_prompt_results = json.load(fp1)
    with open(prompt_emo_model_result_paths[test_id], "r",encoding='utf8') as fp1:
        prompt_emo_model_results = json.load(fp1)
    with open(prompt_emo_prompt_result_paths[test_id], "r",encoding='utf8') as fp1:
        prompt_emo_prompt_results = json.load(fp1)
    
    def evaluate_result(prompt_pepc_model_results,prompt_emo_model_results):
        pred_num, acc_num, true_num = [[0]*7 for i in range(3)]
        acc_num_all = 0
        total_num = len(prompt_pepc_model_results)
        Emotion2Idxs={
            'neutral': 29797,
            'anger':2564,
            'surprise':50263,
            'sadness':50265,
            'joy':2633,
            'disgust':50264,
            'fear':50266,
        }
        Emotion2cateid = {key: i for i,(key,value) in enumerate(Emotion2Idxs.items())}
        for result in prompt_pepc_model_results:
            pepc_prob_label = result['prob_label']
            pepc_true_label = result['true_label']
            pepc_utt_key = result['keyname']
            _, diaId, _, emoId,_,casId,*_ =pepc_utt_key.split('_')
            emo_utt_key = f'dia_{diaId}_utt_{emoId}_index_{int(emoId)-1}'
            emo_prob_label=None
            emo_true_label=None
            for result in prompt_emo_model_results:
                if result['keyname']==emo_utt_key:
                    emo_prob_label = result['prob_label']
                    emo_true_label = result['true_label']
                    break
            
          
            if not emo_prob_label or not emo_true_label:
                raise ValueError('emo_prob_label or emo_true_label not found')

            if emo_prob_label!='neutral' and pepc_prob_label =='Yes' and emo_true_label == emo_prob_label and pepc_prob_label==pepc_true_label: 
                acc_num[Emotion2cateid[emo_true_label]]+=1
            if emo_prob_label!='neutral' and pepc_prob_label=='Yes':
                pred_num[Emotion2cateid[emo_prob_label]]+=1
            
            if emo_true_label!='neutral' and pepc_true_label=='Yes':
                true_num[Emotion2cateid[emo_true_label]]+=1
                
            
        p = [round(a/(p+1e-8),4) for a, p in zip(acc_num,pred_num)] 
        r = [round(a/(t+1e-8),4) for a, t in zip(acc_num,true_num)]
        f = [round(2*p1*r1/(p1+r1+1e-8),4) for p1, r1 in zip(p,r)]
        weight4=[t/sum(true_num[1:5])for t in true_num[1:5]]
        f4=sum([fw*w for fw,w in zip(f[1:5],weight4) ])
        weight6=[t/sum(true_num[1:])for t in true_num[1:]]
        f6=sum([fw*w for fw,w in zip(f[1:],weight6) ])
        result_dict={
        'Emotion2cateid':Emotion2cateid,
        # 'p':p[1:],
        # 'r':r[1:],
        'f':f[1:],
        'f4':round(f4,4),
        'f6':round(f6,4),

        }

        return result_dict


    model_result_dict = evaluate_result(prompt_pepc_model_results,prompt_emo_model_results)
    prompt_result_dict = evaluate_result(prompt_pepc_prompt_results,prompt_emo_prompt_results)

    print(prompt_result_dict)
    print(model_result_dict)

    return '  '


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Evaluate all the test result file.')
    # parser.add_argument('--annotation_path', type=str, required=True)
    # parser.add_argument('--question_path', type=str, required=True)
    # parser.add_argument('--result_path', type=str, required=True)
    # args = parser.parse_args()
    # result_str = _evaluate(args.annotation_path, args.question_path, args.result_path)
    # result_str =_evaluate_for_preEmo_precause_pair()

    # select: emo_pepc (MECPE-C) task with GMEC  or prompt_emo_pepc(MECPE-C) task with LLMs
    # result_str =_evaluate_for_emo_pepc()
    result_str =evaluate_for_prompt_emo_pepc()
    print(result_str)


  