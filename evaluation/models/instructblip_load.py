"""
Refer to https://huggingface.co/Salesforce/instructblip-vicuna-13b
"""
import os
from PIL import Image
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

from .base_model import VLM_BaseModel

model_path = {
    'instructblip_vicuna-7b': './models/LAVIS/checkpoints/instructblip-vicuna-7b',
    'instructblip_vicuna-13b': './models/LAVIS/checkpoints/instructblip-vicuna-13b',
    'instructblip_flan-t5-xl': './models/LAVIS/checkpoints/instructblip-flan-t5-xl',
    'instructblip_flan-t5-xxl': './models/LAVIS/checkpoints/instructblip-flan-t5-xxl',
    }


class InstructBLIP(VLM_BaseModel):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        self.model_path = model_path[model_name]
        self.dtype = torch.float16 if self.config.get('load_float16', False) else torch.float32

        if 'xxl' in model_name or '13b' in model_name:
            with init_empty_weights():
                model = InstructBlipForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                )
            device_map = infer_auto_device_map(model, max_memory={0:'16GiB',1:'16GiB',}, no_split_module_classes=['Block', 'T5Block', 'LlamaDecoderLayer'])
            device_map["language_model.lm_head"] = 0
            self.model = load_checkpoint_and_dispatch(model, self.model_path, device_map=device_map).eval()
        else:
            self.model = InstructBlipForConditionalGeneration.from_pretrained(self.model_path, load_in_8bit=self.config.get('load_in_8bit', False), device_map="cuda", torch_dtype=self.dtype)

        self.processor = InstructBlipProcessor.from_pretrained(self.model_path)


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
                top_k=self.config.get('top_k', 0.9),
                repetition_penalty=1.5,
                length_penalty=self.config.get('length_penalty', 1.0),
                temperature=self.config.get('temperature', 1.0),
        )
        outputs[outputs == 0] = 2
        out = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return out


class InstructBLIP_Vicuna_7B(InstructBLIP):
    def __init__(self, **kwargs):
        model_name = 'instructblip_vicuna-7b'
        super().__init__(model_name=model_name, **kwargs)

class InstructBLIP_Vicuna_13B(InstructBLIP):
    def __init__(self, **kwargs):
        model_name = 'instructblip_vicuna-13b'
        super().__init__(model_name=model_name, **kwargs)

class InstructBLIP_Flan_T5_XL(InstructBLIP):
    def __init__(self, **kwargs):
        model_name = 'instructblip_flan-t5-xl'
        super().__init__(model_name=model_name, **kwargs)

class InstructBLIP_Flan_T5_XXL(InstructBLIP):
    def __init__(self, **kwargs):
        model_name = 'instructblip_flan-t5-xxl'
        super().__init__(model_name=model_name, **kwargs)


if __name__ == "__main__":
    from utils import setup_seeds
    from dataset.test_cases import VLM_TEST_CASES

    setup_seeds()
    model = InstructBLIP_Vicuna_7B()

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
    model = InstructBLIP_Vicuna_13B()

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
    model = InstructBLIP_Flan_T5_XL()

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
    model = InstructBLIP_Flan_T5_XXL()

    for test_case in VLM_TEST_CASES:
        pred = model.generate(
            instruction=test_case['instruction'],
            images=test_case['images'],
        )
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Images:\t{test_case["images"]}')
        print(f'Answer:\t{pred}')
        print('-'*20)