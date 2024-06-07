import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from PIL import Image

from transformers import StoppingCriteriaList

from .base_model import VLM_BaseModel

import sys
sys.path.append("./models/MiniGPT-4")
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *



conv_dict = {'pretrain_vicuna_7b': CONV_VISION_Vicuna0,
             'pretrain_vicuna_13b': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}


config_path = {
    'minigpt4_vicuna-7b': './models/MiniGPT-4/eval_configs/minigpt4_vicuna_7b_eval.yaml',
    'minigpt4_vicuna-13b': './models/MiniGPT-4/eval_configs/minigpt4_vicuna_13b_eval.yaml',
    'minigpt4_llama_2': './models/MiniGPT-4/eval_configs/minigpt4_llama2_eval.yaml',
    }



class MiniGPT4(VLM_BaseModel):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        self.config_path = config_path[model_name]
        cfg = Config(self.config_path)
        model_config = cfg.model_cfg
        self.CONV_VISION = conv_dict[model_config.model_type]
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to("cuda")
        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.stop_words_ids = [[835], [2277, 29937]]
        self.stop_words_ids = [torch.tensor(ids).to("cuda") for ids in self.stop_words_ids]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=self.stop_words_ids)])
        self.chat = Chat(model, self.vis_processor, device="cuda", stopping_criteria=self.stopping_criteria)

    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """

        if len(images) == 0:
            raise ValueError('No image is provided.')
        if len(images) > 1:
            return '[Skipped]: Currently only support single image.'
        
        # Init chat state
        CONV_VISION = self.CONV_VISION
        chat_state = CONV_VISION.copy()
        img_list = []

        # download image image
        image_path = images[0]
        img = Image.open(image_path).convert("RGB")
        
        # upload Image
        self.chat.upload_img(img, chat_state, img_list)
        self.chat.encode_img(img_list)
        instruction = instruction[0] if type(instruction)==list else instruction
        instruction = self.inst_pre + instruction + self.inst_suff
        # ask
        self.chat.ask(instruction, chat_state)

        # answer
        out, out_token = self.chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=self.config.get('num_beams', 5),
            temperature=self.config.get('temperature', 1.0),
            max_new_tokens=self.config.get('max_new_tokens', 300),
            top_p=self.config.get('top_p', 0.9),
            length_penalty=self.config.get('length_penalty', 1.0), 
        )
        out = out.strip()
        return out


class MiniGPT4_Vicuna_7B(MiniGPT4):
    def __init__(self, **kwargs):
        model_name = 'minigpt4_vicuna-7b'
        super().__init__(model_name=model_name, **kwargs)

class MiniGPT4_Vicuna_13B(MiniGPT4):
    def __init__(self, **kwargs):
        model_name = 'minigpt4_vicuna-13b'
        super().__init__(model_name=model_name, **kwargs)

class MiniGPT4_LLaMA_2(MiniGPT4):
    def __init__(self, **kwargs):
        model_name = 'minigpt4_llama_2'
        super().__init__(model_name=model_name, **kwargs)



if __name__ == "__main__":
    from utils import setup_seeds
    from dataset.test_cases import VLM_TEST_CASES

    setup_seeds()
    model = MiniGPT4_Vicuna_7B()

    for test_case in VLM_TEST_CASES:
        pred = model.generate(
            instruction=test_case['instruction'],
            images=test_case['images'],
        )
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Images:\t{test_case["images"]}')
        print(f'Answer:\t{pred}')
        print('-'*20)

    del model
    torch.cuda.empty_cache()
    model = MiniGPT4_Vicuna_13B()

    for test_case in VLM_TEST_CASES:
        pred = model.generate(
            instruction=test_case['instruction'],
            images=test_case['images'],
        )
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Images:\t{test_case["images"]}')
        print(f'Answer:\t{pred}')
        print('-'*20)


    del model
    torch.cuda.empty_cache()
    model = MiniGPT4_LLaMA_2()

    for test_case in VLM_TEST_CASES:
        pred = model.generate(
            instruction=test_case['instruction'],
            images=test_case['images'],
        )
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Images:\t{test_case["images"]}')
        print(f'Answer:\t{pred}')
        print('-'*20)