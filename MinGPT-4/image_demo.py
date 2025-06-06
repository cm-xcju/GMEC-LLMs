import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from pdb import set_trace as stop
import glob
import json
from datetime import datetime
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')


# chat_state.append_message(chat_state.roles[0], msg)
# chat_state.messages=[
#     [
#         'human',''.join([value for role, value in chat_state.messages])

#     ]
# ]


# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return chat_state, img_list

def upload_img(gr_img, chat_state):
    if gr_img is None:
        return
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(gr_img, chat_state, img_list)
    return  chat_state, img_list

def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    return chatbot, chat_state, img_list


def get_describe(gr_img):
    img_list=[]
    chat_state=CONV_VISION.copy()
    # for gr_img in img_list_raw:
    llm_message1 = chat.upload_img(gr_img, chat_state, img_list)

    msg='what is the emotion and cause of this image?'
    chatbot, chat_state = gradio_ask(msg,[],chat_state)


    llm_message = chat.answer(conv=chat_state,
                                img_list=img_list,
                                num_beams=5,
                                temperature=1.0,
                                max_new_tokens=100,
                                max_length=2000)[0]
    return llm_message

target_path = '../datasets/MELD-ECPE/Captions/13B-mingpt4.json'
image_path = '../datasets/MELD-ECPE/Images'
if not os.path.exists(target_path):
    mingpt4_captions = {}
else:
    with open(target_path, "r",encoding='utf8') as fp:
        mingpt4_captions = json.load(fp)
image_names = glob.glob(image_path+'/*.jpg')
for i,image_file in enumerate(image_names):
    key_name = image_file.split('/')[-1][:-4]
    if key_name not in mingpt4_captions.keys():
        describe = get_describe(image_file)
        mingpt4_captions[key_name]=describe
    if i%50==0:
        print(f'item {i} nowTime: {datetime.now():%Y-%m-%d %H:%M:%S}\n', end='          ')
        with open(target_path, "w",encoding='utf8') as fp:
                fp.write(json.dumps(mingpt4_captions, indent=4, ensure_ascii=False))
with open(target_path, "w",encoding='utf8') as fp:
        fp.write(json.dumps(mingpt4_captions, indent=4, ensure_ascii=False))




# title = """<h1 align="center">Demo of MiniGPT-4</h1>"""
# description = """<h3>This is the demo of MiniGPT-4. Upload your images and start chatting!</h3>"""
# article = """<p><a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p><p><a href='https://github.com/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p><a href='https://raw.githubusercontent.com/Vision-CAIR/MiniGPT-4/main/MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></p>
# """

# #TODO show examples below

# with gr.Blocks() as demo:
#     gr.Markdown(title)
#     gr.Markdown(description)
#     gr.Markdown(article)

#     with gr.Row():
#         with gr.Column(scale=0.5):
#             image = gr.Image(type="pil")
#             upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
#             clear = gr.Button("Restart")
            
#             num_beams = gr.Slider(
#                 minimum=1,
#                 maximum=10,
#                 value=1,
#                 step=1,
#                 interactive=True,
#                 label="beam search numbers)",
#             )
            
#             temperature = gr.Slider(
#                 minimum=0.1,
#                 maximum=2.0,
#                 value=1.0,
#                 step=0.1,
#                 interactive=True,
#                 label="Temperature",
#             )

#         with gr.Column():
#             chat_state = gr.State()
#             img_list = gr.State()
#             chatbot = gr.Chatbot(label='MiniGPT-4')
#             text_input = gr.Textbox(label='User', placeholder='Please upload your image first', interactive=False)
    
#     upload_button.click(upload_img, [image, text_input, chat_state], [image, text_input, upload_button, chat_state, img_list])
    
#     text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
#         gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
#     )
#     clear.click(gradio_reset, [chat_state, img_list], [chatbot, image, text_input, upload_button, chat_state, img_list], queue=False)

# demo.launch(share=True, enable_queue=True)
