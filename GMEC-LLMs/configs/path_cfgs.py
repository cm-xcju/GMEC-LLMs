# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: set const paths and dirs
# ------------------------------------------------------------------------------ #

import os

class PATH:
    def __init__(self):

        self.LOG_ROOT = 'outputs/logs/'
        self.CKPT_ROOT = 'outputs/ckpts/'
        self.RESULTS_ROOT = 'outputs/results/'
        self.DATASET_ROOT = '../../datasets/MECPE-main'
        self.QUESTIOIN_ROOT = './question_files/'
        self.MECPE_ROOT = '../../datasets/MELD-ECPE'
        self.ASSETS_ROOT = 'assets/'
        self.MELD_ROOT = '../../datasets/MELD.Raw'

        self.IMAGE_FEATURE_DIR={
            # 'feature':self.MECPE_ROOT + '/MELDECPE/video_embedding_4096.npy',
            'feature':self.MECPE_ROOT + '/MELDECPE/video_embedding_CLIPmean_4096.npy',
            'v2id':self.MECPE_ROOT + '/MELDECPE/video_id_mapping.npy',
        }
        self.AUDIO_FEATURE_DIR={
            'feature':self.MECPE_ROOT + '/MELDECPE/audio_embedding_6373.npy',
        }
        self.TEXT_DIR={
            'train':self.MECPE_ROOT + '/MELDECPE/mecpe_train.json',
            'dev':self.MECPE_ROOT + '/MELDECPE/mecpe_dev.json',
            'test':self.MECPE_ROOT + '/MELDECPE/mecpe_test.json',
        }

        self.IMAGE_DIR={
            'mecpe':self.MECPE_ROOT + '/Images/',
           
        }

        self.QUESTION_PATH = {
            'emotrain': self.QUESTIOIN_ROOT + 'Emotion_quetions_train.json',
            'emodev': self.QUESTIOIN_ROOT + 'Emotion_quetions_dev.json',
            'emotest': self.QUESTIOIN_ROOT + 'Emotion_quetions_test.json',
            'causetrain': self.QUESTIOIN_ROOT + 'Cause_questions_train.json',
            'causedev': self.QUESTIOIN_ROOT + 'Cause_questions_dev.json',
            'causetest': self.QUESTIOIN_ROOT + 'Cause_questions_test.json',
            'emocausetrain': self.QUESTIOIN_ROOT + 'Emotion_Cause_questions_train.json',
            'emocausedev': self.QUESTIOIN_ROOT + 'Emotion_Cause_questions_dev.json',
            'emocausetest': self.QUESTIOIN_ROOT + 'Emotion_Cause_questions_test.json',
        
        }


    
        # }


