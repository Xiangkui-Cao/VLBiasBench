from PIL import Image 
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_model import VLM_BaseModel


model_path = './models/Qwen-VL/checkpoints/Qwen-VL'


class Qwen_VL(VLM_BaseModel):
    def __init__(self, model_path=model_path, model_name='qwen-vl', **kwargs):
        super().__init__(model_name, **kwargs)
        self.inst_suff = self.inst_suff + ' Answer:'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()


    def generate(self, instruction, images, interleaved=False):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """

        instruction = self.inst_pre + instruction + self.inst_suff
        query = self.tokenizer.from_list_format([
            {'image': images[0]}, # Either a local path or an url
            {'text': instruction},
        ])
        inputs = self.tokenizer(query, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            pred = self.model.generate(
                **inputs,
                do_sample=True,
                num_beams=self.config.get('num_beams', 5),
                max_new_tokens=self.config.get('max_new_tokens', 256),
                min_length=1,
                top_p=self.config.get('top_p', 0.9),
                top_k=self.config.get('top_k', 50),
                length_penalty=self.config.get('length_penalty', 1.0),
                temperature=self.config.get('temperature', 1.0),
            )
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        response = response[response.find("Answer:")+len("Answer:"):]
        response = response[:response.find("<|endoftext|>")]
        return response



if __name__ == "__main__":
    from utils import setup_seeds
    from dataset.test_cases import VLM_TEST_CASES

    setup_seeds()
    model = Qwen_VL()
    for test_case in VLM_TEST_CASES:
        pred = model.generate(
            instruction=test_case['instruction'],
            images=test_case['images'],
        )
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Images:\t{test_case["images"]}')
        print(f'Answer:\t{pred}')
        print('-'*20)