import argparse
import os
import random

import re
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
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)



config_path = './models/MiniGPT-4/eval_configs/minigptv2_eval.yaml'


class MiniGPT_v2(VLM_BaseModel):
    def __init__(self, config_path=config_path, model_name='minigpt_v2', **kwargs):
        super().__init__(model_name, **kwargs)
        self.config_path = config_path
        cfg = Config(self.config_path)
        model_config = cfg.model_cfg
        self.CONV_VISION = CONV_VISION
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda')

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.chat = Chat(model, self.vis_processor, device='cuda')

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
        out = self.chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=self.config.get('num_beams', 5),
            temperature=self.config.get('temperature', 1.0),
            max_new_tokens=self.config.get('max_new_tokens', 500),
            top_p=self.config.get('top_p', 0.9),
            length_penalty=self.config.get('length_penalty', 1.0), 
        )[0]
        out = out.strip()
        return out




if __name__ == "__main__":
    from utils import setup_seeds
    from dataset.test_cases import VLM_TEST_CASES

    setup_seeds()
    model = MiniGPT_v2()

    for test_case in VLM_TEST_CASES:
        pred = model.generate(
            instruction=test_case['instruction'],
            images=test_case['images'],
        )
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Images:\t{test_case["images"]}')
        print(f'Answer:\t{pred}')
        print('-'*20)

  
    model.inst_pre = "[vqa] "
    for test_case in VLM_TEST_CASES:
        pred = model.generate(
            instruction=test_case['instruction'],
            images=test_case['images'],
        )
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Images:\t{test_case["images"]}')
        print(f'Answer:\t{pred}')
        print('-'*20)