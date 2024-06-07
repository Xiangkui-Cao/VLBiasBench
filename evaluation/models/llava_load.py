import os
import torch
import requests

from PIL import Image
from io import BytesIO

import sys
sys.path.append("./models/LLaVA")
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model import *
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


from .base_model import VLM_BaseModel


model_path = {
    'llava-1.5-7b': './models/LLaVA/checkpoints/llava-v1.5-7b',
    'llava-1.5-13b': './models/LLaVA/checkpoints/llava-v1.5-13b',
    }


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def get_model(model_path, config):
    """Model Provider with tokenizer and processor.
    """
    # in case of using a pretrained model with only a MLP projector weights
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, config.get("load_8bit", False), config.get("load_4bit", False), device='cuda')
    return model, tokenizer, image_processor


class LLaVA_1_5(VLM_BaseModel):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        self.model_path = model_path[model_name]

        self.model, self.tokenizer, self.image_processor = get_model(self.model_path, self.config)
        self.model.eval()
        self.mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        for i in self.model.named_parameters():
            print(f"{i[0]} -> {i[1].dtype}")


        if "v0" in self.model_path.lower():
            conv_mode = "v0"
        elif "v1" in self.model_path.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_path.lower():
            conv_mode = "mpt_multimodal"
        elif "llama-2" in self.model_path.lower():
            conv_mode = "llava_llama_2"
        else:
            conv_mode = "v0"
        self.conv_mode = conv_mode

        ## Generation configs
        self.do_sample = self.config.get('do_sample', True)
        self.temperature = self.config.get('temperature', 0.2)
        self.num_beams = self.config.get('num_beams', 5)
        self.max_new_tokens = self.config.get('max_new_tokens', 1024)
        self.top_p = self.config.get('top_p', 0.9)
        self.top_k = self.config.get('top_k', 50)
        self.length_penalty = self.config.get('length_penalty', 1.0),

    def generate(self, instruction, images):

        assert len(images) == 1

        image_path = images[0]
        
        query = self.inst_pre + instruction + self.inst_suff
        
        if self.mm_use_im_start_end:
            query = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + query
        else:
            query = DEFAULT_IMAGE_TOKEN + '\n' + query

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to('cuda')

        image = Image.open(image_path)
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)


        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to('cuda'),
                do_sample=self.do_sample,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                top_p=self.top_p,
                top_k=self.top_k,
                length_penalty=self.length_penalty,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs



class LLaVA_1_5_7B(LLaVA_1_5):
    def __init__(self, **kwargs):
        model_name = 'llava-1.5-7b'
        super().__init__(model_name=model_name, **kwargs)

class LLaVA_1_5_13B(LLaVA_1_5):
    def __init__(self, **kwargs):
        model_name = 'llava-1.5-13b'
        super().__init__(model_name=model_name, **kwargs)




if __name__ == "__main__":
    from utils import setup_seeds
    from dataset.test_cases import VLM_TEST_CASES


    setup_seeds()
    model = LLaVA_1_5_7B()

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
    model = LLaVA_1_5_13B()

    for test_case in VLM_TEST_CASES:
        pred = model.generate(
            instruction=test_case['instruction'],
            images=test_case['images'],
        )
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Images:\t{test_case["images"]}')
        print(f'Answer:\t{pred}')
        print('-'*20)
