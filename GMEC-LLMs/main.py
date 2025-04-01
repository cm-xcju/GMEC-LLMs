import argparse
import yaml
import torch

from evaluation.mecpec_evaluate import mecEvaluater
from configs.task_cfgs import Cfgs
from LLMprompt import get_args, get_runner
from pdb import set_trace as stop
# parse cfgs and args
args = get_args()
__C = Cfgs(args)

with open(args.cfg_file, 'r') as f:
    yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
__C.override_from_dict(yaml_dict)
print(__C)

# build runner
if  __C.DATA_TAG == 'mec':
     evaluater = mecEvaluater(
        __C
    )

runner = get_runner(__C, evaluater)

# run
runner.run()
