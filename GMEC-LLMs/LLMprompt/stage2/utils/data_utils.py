# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: dataset utils for stage2
# ------------------------------------------------------------------------------ #

import json
from typing import Dict
import pickle
from collections import Counter
from ipdb import set_trace as stop
# following two score is rough, and only for print accuracies during inferring.


def ok_score(gt_answers):
    gt_answers = [a['answer'] for a in gt_answers]
    ans2cnt = Counter(gt_answers)
    # sort
    ans2cnt = sorted(ans2cnt.items(), key=lambda x: x[1], reverse=True)
    ans2score = {}
    for ans, cnt in ans2cnt:
        # ans2score[ans] = min(1.0, cnt / 3.0)
        if cnt == 1:
            ans2score[ans] = 0.3
        elif cnt == 2:
            ans2score[ans] = 0.6
        elif cnt == 3:
            ans2score[ans] = 0.9
        else:
            ans2score[ans] = 1.0
    return ans2score


def aok_score(gt_answers):
    gt_answers = [a for a in gt_answers]
    ans2cnt = Counter(gt_answers)
    # sort
    ans2cnt = sorted(ans2cnt.items(), key=lambda x: x[1], reverse=True)
    ans2score = {}
    for ans, cnt in ans2cnt:
        # ans2score[ans] = min(1.0, cnt / 3.0)
        if cnt == 1:
            ans2score[ans] = 1 / 3.
        elif cnt == 2:
            ans2score[ans] = 2 / 3.
        else:
            ans2score[ans] = 1.
    return ans2score


class Qid2Data(Dict):
    def __init__(self, __C, splits, annotated=False, similar_examples=None):
        super().__init__()

        self.__C = __C
        self.annotated = annotated
        if self.__C.TASK_TYPE in ['emotion', 'emotion_neu']:
            ques_path = 'emo'+splits[0]
            split_path = self.__C.QUESTION_PATH[ques_path]

            self.question_dict = json.load(
                open(split_path, 'r', encoding='utf-8'))
        elif self.__C.TASK_TYPE in ['preEmo_precause_pair']:
            ques_path = 'emocause'+splits[0]
            split_path = self.__C.QUESTION_PATH[ques_path]
            self.question_dict = json.load(
                open(split_path, 'r', encoding='utf-8'))


        # ques_set = []
        # for split in splits:
        #     split_path = self.__C.QUESTION_PATH[split]
        #     _ques_set = json.load(open(split_path, 'r'))
        #     if 'questions' in _ques_set:
        #         _ques_set = _ques_set['questions']
        #     ques_set += _ques_set
        # qid_to_ques = {q['question_id']: q for q in ques_set}

        # if annotated:
        #     anno_set = []
        #     for split in splits:
        #         split_path = self.__C.ANSWER_PATH[split]
        #         _anno_set = json.load(open(split_path, 'r'))
        #         if 'annotations' in _anno_set:
        #             _anno_set = _anno_set['annotations']
        #         anno_set += _anno_set
        #     qid_to_anno = {a['question_id']: a for a in anno_set}

        qid_to_topk = json.load(open(__C.CANDIDATES_PATH))
        # qid_to_topk = {t['question_id']: t for t in topk}
        new_UTTS_PATH = __C.UTTS_PATH[:-5]+f'_{splits[0]}'+__C.UTTS_PATH[-5:]

        prompt_texts = json.load(open(new_UTTS_PATH, 'r', encoding='utf-8'))

        iid_to_capt = json.load(open(__C.CAPTIONS_PATH, 'r', encoding='utf-8'))

        # _score = aok_score if 'aok' in __C.TASK else ok_score

        # ques_set = ques_set['questions']
        # anno_set = anno_set['annotations']
        dia_utt_qs_data = {}
        new_prompt_texts_data={}
        for dia_id in prompt_texts.keys():
            dia_utt_qs_data[dia_id] = {}
            utts = prompt_texts[dia_id]

            repl_utts = [item.replace('<video>', '').replace('<audio>', '').replace('<text>', '').replace(
                '<videoblankpos>', '').replace('<audioblankpos>', '').replace('<sep>', '') for item in utts]
            new_prompt_texts_data[dia_id]=repl_utts
            dia_utt_qs_data[dia_id]['Context'] = {}# '\n'.join(repl_utts)
            dia_utt_qs_data[dia_id]['Questions'] = {}
            dia_utt_qs_data[dia_id]['Captions'] = {}
            dia_utt_qs_data[dia_id]['Candidates'] = {}
            dia_utt_qs_data[dia_id]['Examples'] = {}
            dia_utt_qs_data[dia_id]['Labels'] = {}
            dia_utt_qs_data[dia_id]['True_ans'] = {}

        # only for emotion and cause extraction
        if self.__C.TASK_TYPE in ['emotion', 'emotion_neu']:
            for dia_id in self.question_dict.keys():
                question_list = self.question_dict[dia_id]
                for i, qs_item in enumerate(question_list):
                    raw_qs = qs_item['question_neu'] if self.__C.TASK_TYPE in [
                        'emotion_neu'] else qs_item['question']
                    question = raw_qs.replace('<text>', '').replace('<video>', '').replace('<audio>', '').replace(
                        '<videoblankpos>', '').replace('<audioblankpos>', '').replace('<sep>', '').replace('<|endoftext|>', '')
                    dia_utt_qs_data[dia_id]['Questions'][str(i+1)] = question
                    # _,dia_key,_,utterance_ID,_,index=emo_keyname.split('_')
                    # dia1357utt1
                    utt_key = qs_item['utt_key']
                    uttID = utt_key.split('utt')[-1]
                    assert eval(uttID) == i+1
                    if self.__C.TASK_TYPE in ['emotion']:
                        label = qs_item['emotion']
                        true_ans = qs_item['answer'].replace('<|endoftext|>', '')
                    elif self.__C.TASK_TYPE in ['emotion_neu']:
                        label = qs_item['Emotion_neu']
                        true_ans = qs_item['answer_neu'].replace(
                            '<|endoftext|>', '')
                    elif self.__C.TASK_TYPE in ['cause']:
                        label = qs_item['cause']
                        true_ans = qs_item['answer'].replace('<|endoftext|>', '')
                    else:
                        raise ValueError('not found any label in data_utils')
                    # for context 
                    
                    best_utt = max(int(uttID)-1-2,0)
                    end_utt = min(int(uttID)+4,len(new_prompt_texts_data[dia_id]))

                    new_repl_utts=None
                    if best_utt>=end_utt:
                        stop()
                       
                    if self.__C.Context_cut == 'realtime':
                        new_repl_utts = new_prompt_texts_data[dia_id][best_utt:int(uttID)]
                    elif self.__C.Context_cut == 'static':
                        new_repl_utts = new_prompt_texts_data[dia_id][best_utt:end_utt]
                    
                   
                    dia_utt_qs_data[dia_id]['Context'][uttID]='\n'.join(new_repl_utts)
                    # for candiate_path
                    key_name = f'dia_{dia_id}_utt_{uttID}_index_{i}'
                    dia_utt_qs_data[dia_id]['Candidates'][uttID] = qid_to_topk[key_name]

                    # dia21utt4_m image path
                    imguttId = f'dia{dia_id}utt{uttID}_m'
                    dia_utt_qs_data[dia_id]['Captions'][uttID] = iid_to_capt[imguttId]
                    dia_utt_qs_data[dia_id]['Labels'][uttID] = label
                    dia_utt_qs_data[dia_id]['True_ans'][uttID] = true_ans

                    if similar_examples:
                        dia_utt_qs_data[dia_id]['Examples'] = similar_examples[key_name]
        # only for emotion and cause extraction
        if self.__C.TASK_TYPE in ['preEmo_precause_pair']:
            for dia_id in self.question_dict.keys():
                question_list = self.question_dict[dia_id]
                for i, qs_item in enumerate(question_list):
                    raw_qs = qs_item['question']
                    question = raw_qs.replace('<text>', '').replace('<video>', '').replace('<audio>', '').replace(
                        '<videoblankpos>', '').replace('<audioblankpos>', '').replace('<sep>', '').replace('<|endoftext|>', '')
                    question = question.format('','')
                    
                    # _,dia_key,_,utterance_ID,_,index=emo_keyname.split('_')
                    # dia1357utt1
                    Emotion_utterance_ID = qs_item['Emotion_utterance_ID']
                    Cause_utterance_ID = qs_item['Cause_utterance_ID']
                    utt_key=f'dia{dia_id}_uttEmo{Emotion_utterance_ID}_uttCas{Cause_utterance_ID}'
                    # utt_key = qs_item['utt_key']
                    # uttID = utt_key.split('utt')[-1]
                    # assert eval(uttID) == i+1
                    
                    # for context 
                    label =  qs_item['cause']
                    true_ans = qs_item['answer'].replace('<|endoftext|>', '')

                    begin_utt = min(max(int(Emotion_utterance_ID)-1-2,0),int(Cause_utterance_ID)-1)
                    end_utt = min(int(Emotion_utterance_ID)+4,len(new_prompt_texts_data[dia_id]))
                    new_repl_utts=None
                    if begin_utt>=end_utt:
                        stop()
                    if self.__C.Context_cut == 'realtime':
                        if int(Emotion_utterance_ID)<int(Cause_utterance_ID):
                            continue
                        
                        new_repl_utts = new_prompt_texts_data[dia_id][begin_utt:int(Emotion_utterance_ID)]
                    elif self.__C.Context_cut == 'static':
                        new_repl_utts = new_prompt_texts_data[dia_id][begin_utt:end_utt]
                    
                    key_name = f'dia_{dia_id}_uttEmo_{Emotion_utterance_ID}_uttCas_{Cause_utterance_ID}_index_{i}'
                    
                   
                
                    new_key = f'uttEmo_{Emotion_utterance_ID}_uttCas_{Cause_utterance_ID}_index_{i}'
                    dia_utt_qs_data[dia_id]['Context'][new_key]='\n'.join(new_repl_utts)
                    dia_utt_qs_data[dia_id]['Questions'][new_key] = question
                    # for candiate_path
                    dia_utt_qs_data[dia_id]['Candidates'][new_key] = qid_to_topk[key_name]
                    # dia21utt4_m image path
                    imguttId = f'dia{dia_id}utt{Emotion_utterance_ID}_m'
                    dia_utt_qs_data[dia_id]['Captions'][new_key] = iid_to_capt[imguttId]
                    dia_utt_qs_data[dia_id]['Labels'][new_key] = label
                    dia_utt_qs_data[dia_id]['True_ans'][new_key] = true_ans

                    if similar_examples:
                        dia_utt_qs_data[dia_id]['Examples'] = similar_examples[key_name]
       
        # len(dia_utt_qs_data['926']['Examples'])
        self.dia_utt_qs_data = dia_utt_qs_data
        qid_to_data = {}
        for diaID, item in dia_utt_qs_data.items():
            Context = item['Context']
            Questions = item['Questions']
            Captions = item['Captions']
            Candidates = item['Candidates']
            Examples = item['Examples']
            Labels = item['Labels']
            True_ans = item['True_ans']

            assert len(Captions) == len(Candidates) and len(Labels) == len(
                Captions) and len(Questions) == len(Captions)
          
            if self.__C.TASK_TYPE in ['emotion']:
                for i in range(len(Captions)):
                    uttID = i+1
                    utt_key = f'dia_{diaID}_utt_{uttID}_index_{i}'
                    uttIDstr = str(uttID)

                    most_answer = self.getmostans(Candidates[uttIDstr])
                    qid_to_data[utt_key] = {
                        'question_id': utt_key,
                        'Context': Context[uttIDstr],
                        'Caption': Captions[uttIDstr],
                        'Question': Questions[uttIDstr],
                        'most_answer': most_answer,
                        'Candidate': Candidates[uttIDstr],
                        'Label': Labels[uttIDstr],
                        'True_ans': True_ans[uttIDstr],
                    }

                    if similar_examples:
                        qid_to_data[utt_key]['similar_qids'] = similar_examples[utt_key]

                        # check if all items have similar_qids
                        for qid, item in self.items():
                            if 'similar_qids' not in item:
                                raise ValueError(
                                    f'qid {qid} does not have similar_qids')
            elif self.__C.TASK_TYPE in ['preEmo_precause_pair']:
                 for new_key in Captions.keys():
                    # key_name = f'dia_{dia_id}_uttEmo_{Emotion_utterance_ID}_uttCas_{Cause_utterance_ID}_index_{i}'
                    # uttID = i+1
                    utt_key = f'dia_{diaID}_{new_key}'
                    # uttIDstr = str(uttID)
                   
                    most_answer = self.getmostans(Candidates[new_key])
                    qid_to_data[utt_key] = {
                        'question_id': utt_key,
                        'Context': Context[new_key],
                        'Caption': Captions[new_key],
                        'Question': Questions[new_key],
                        'most_answer': most_answer,
                        'Candidate': Candidates[new_key],
                        'Label': Labels[new_key],
                        'True_ans': True_ans[new_key],
                    }

                    if similar_examples:
                        qid_to_data[utt_key]['similar_qids'] = similar_examples[utt_key]

                        # check if all items have similar_qids
                        for qid, item in self.items():
                            if 'similar_qids' not in item:
                                raise ValueError(
                                    f'qid {qid} does not have similar_qids')

        self.qid_to_data = qid_to_data
     

    def __getitem__(self, __key):
        return self.qid_to_data[__key]

    def get_caption(self, qid):
        caption = self.qid_to_data[qid]['Caption']
        # if with_tag:
        #     tags = self.get_tags(qid, k_tags)
        #     caption += ' ' + ', '.join(tags) + '.'
        return caption

    def get_question(self, qid):
        return self[qid]['Question']

    def get_qs_context(self, qid):
        return self[qid]['Context']

    def get_gt_answers(self, qid):
        return self[qid]['Candidate']

    def get_candidate(self, qid):
        cand = self[qid]['Candidate']
        cand_s = sorted(cand, key=lambda x: x['confidence'], reverse=True)
        return cand_s

    def get_most_answer(self, qid):
        return self[qid]['most_answer']

    def get_label(self, qid):
        return self[qid]['Label']

    def get_true_answer(self, qid):
        return self[qid]['True_ans']

    def get_topk_answers(self, qid, k=None):
        if k is None:
            return self[qid]['Candidate']
        else:
            return self[qid]['Candidate'][:k]

    def get_similar_qids(self, qid, k=None):
        similar_qids = self[qid]['similar_qids']
        if k is not None:
            similar_qids = similar_qids[:k]
        return similar_qids

    def getmostans(self, candidate):

        value_list = [item['confidence'] for item in candidate]
        max_v = max(value_list)
        idx = value_list.index(max_v)
        return candidate[idx]['answer']

    def evaluate_by_threshold(self, ans_set, threshold=1.0):
        total_score = 0.0
        for item in ans_set:
            qid = item['question_id']
            topk_answers = self.get_topk_answers(qid)
            top1_confid = topk_answers[0]['confidence']
            if top1_confid > threshold:
                answer = topk_answers[0]['answer']
            else:
                answer = item['answer']
            gt_answers = self.get_gt_answers(qid)
            if answer in gt_answers:
                total_score += gt_answers[answer]
        return total_score / len(ans_set)

    def topk_accuracy(self, k=1, sub_set=None):
        total_score = 0.0
        if sub_set is not None:
            qids = sub_set
        else:
            qids = list(self.qid_to_data.keys())
        for qid in qids:
            topk_answers = self.get_topk_answers(qid)[:k]
            gt_answers = self.get_gt_answers(qid)
            score_list = [gt_answers.get(a['answer'], 0.0)
                          for a in topk_answers]
            total_score += max(score_list)
        return total_score / len(qids)

    def rt_evaluate(self, answer_set):
        if not self.annotated:
            return ''
        score1 = self.evaluate_by_threshold(answer_set, 1.0) * 100
        score2 = self.evaluate_by_threshold(answer_set, 0.0) * 100
        score_string = f'{score2:.2f}->{score1:.2f}'
        return score_string
