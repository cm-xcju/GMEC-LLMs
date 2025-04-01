# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: The goal of this file is to define the mapping from task and data
# mode to dataset splits.
# ------------------------------------------------------------------------------ #

class DictSafe(dict):

    def __init__(self, data={}):
        dict.__init__(self, data)
        for key, value in data.items():
            if isinstance(value, dict):
                self[key] = DictSafe(value)

    def __getitem__(self, key):
        return self.get(key, [])


TASK_TO_SPLIT = {
    'mec': {
        'pretrain': {
            'train_split': ['train'],
            'dev_split': ['dev'],
            'test_split': ['test'],
        
        },
        'finetune': {
            'train_split': ['train'],
            'dev_split': ['dev'],
            'test_split': ['test'],
        },
        'train': {
            'train_split': ['train'],
            'dev_split': ['dev'],
            'test_split': ['test'],
        }
    },
    
}
TASK_TO_SPLIT = DictSafe(TASK_TO_SPLIT)

SPLIT_TO_IMGS = {
    'mec':'mecpe'
    # 'v2train': 'train2014',
    # 'v2val': 'val2014',
    # 'v2valvg_no_ok': 'val2014',
    # 'vg': 'val2014',
    # 'oktrain': 'train2014',
    # 'oktest': 'val2014',
    # 'aoktrain': 'train2017',
    # 'aokval': 'val2017',
    # 'aoktest': 'test2017',
}


if __name__ == '__main__':
    print(TASK_TO_SPLIT['mecpe']['test']['train_split'])