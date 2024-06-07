from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from .base_model import VLM_BaseModel
import sys

model_path = {
    'blip2_flan-t5-xl': './models/LAVIS/checkpoints/blip2-flan-t5-xl',
    'blip2-opt-2.7b': './models/LAVIS/checkpoints/blip2-opt-2.7b',
    'blip2-opt-6.7b': './models/LAVIS/checkpoints/blip2-opt-6.7b',
    }


class BLIP2(VLM_BaseModel):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        self.model_path = model_path[model_name]
        self.dtype = torch.float16 if self.config.get('load_float16', False) else torch.float32
        self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=self.dtype).to('cuda')
        self.processor = Blip2Processor.from_pretrained(self.model_path)
        self.inst_pre = 'Question: ' + self.inst_pre
        self.inst_suff = self.inst_suff + ' Answer:'
        

    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        image_path = images[0]  # instruct_blip2 only supports single image
        raw_image = Image.open(image_path).convert("RGB")
        instruction = instruction[0] if type(instruction)==list else instruction
        instruction = self.inst_pre + instruction + self.inst_suff
        inputs = self.processor(images=raw_image, text=instruction, return_tensors="pt").to('cuda', self.dtype)
        outputs = self.model.generate(
                **inputs,
                do_sample=True,
                num_beams=self.config.get('num_beams', 5),
                max_new_tokens=self.config.get('max_new_tokens', 256),
                min_length=1,
                top_p=self.config.get('top_p', 0.9),
                top_k=self.config.get('top_k', 50),
                repetition_penalty=1.5,
                length_penalty=self.config.get('length_penalty', 1.0),
                temperature=self.config.get('temperature', 1.0),
        )
        outputs[outputs == 0] = 2
        out = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return out


class BLIP2_Flan_T5_XL(BLIP2):
    def __init__(self, **kwargs):
        model_name = 'blip2_flan-t5-xl'
        super().__init__(model_name=model_name, **kwargs)

class BLIP2_OPT_3B(BLIP2):
    def __init__(self, **kwargs):
        model_name = 'blip2-opt-2.7b'
        super().__init__(model_name=model_name, **kwargs)

class BLIP2_OPT_7B(BLIP2):
    def __init__(self, **kwargs):
        model_name = 'blip2-opt-6.7b'
        super().__init__(model_name=model_name, **kwargs)




if __name__ == "__main__":
    from utils import setup_seeds
    from dataset.test_cases import VLM_TEST_CASES

    setup_seeds()
    model = BLIP2_Flan_T5_XL()

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
    model = BLIP2_OPT_3B()

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
    model = BLIP2_OPT_7B()

    for test_case in VLM_TEST_CASES:
        pred = model.generate(
            instruction=test_case['instruction'],
            images=test_case['images'],
        )
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Images:\t{test_case["images"]}')
        print(f'Answer:\t{pred}')
        print('-'*20)