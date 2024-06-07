import torch
import transformers
from PIL import Image
import sys

sys.path.append("./models/Otter/src")
from otter_ai import OtterForConditionalGeneration

from .base_model import VLM_BaseModel


model_path = './models/Otter/checkpoints/OTTER-Image-MPT7B'

class Otter(VLM_BaseModel):
    def __init__(self, model_path=model_path, model_name='otter', **kwargs):
        super().__init__(model_name, **kwargs)
        self.inst_pre = "<image>User: " + self.inst_pre
        self.inst_suff = self.inst_suff + " GPT:<answer>"
        precision = {}
        if self.config['load_bit'] == "bf16":
            precision["torch_dtype"] = torch.bfloat16
        elif self.config['load_bit'] == "fp16":
            precision["torch_dtype"] = torch.float16
        elif self.config['load_bit'] == "fp32":
            precision["torch_dtype"] = torch.float32
        self.model = OtterForConditionalGeneration.from_pretrained(model_path, device_map="sequential", **precision)
        self.model.text_tokenizer.padding_side = "left"
        self.tokenizer = self.model.text_tokenizer
        self.image_processor = transformers.CLIPImageProcessor()
        self.model.eval()


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
        

        image_path = images[0]
        image = Image.open(image_path).convert("RGB")

        if isinstance(image, Image.Image):
            if image.size == (224, 224) and not any(image.getdata()):  # Check if image is blank 224x224 image
                vision_x = torch.zeros(1, 1, 1, 3, 224, 224, dtype=next(self.model.parameters()).dtype)
            else:
                vision_x = self.image_processor.preprocess([image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        else:
            raise ValueError("Invalid input data. Expected PIL Image.")


        instruction = self.inst_pre + instruction + self.inst_suff
        
        lang_x = self.model.text_tokenizer([instruction], return_tensors="pt",)

        model_dtype = next(self.model.parameters()).dtype

        vision_x = vision_x.to(dtype=model_dtype)
        lang_x_input_ids = lang_x["input_ids"]
        lang_x_attention_mask = lang_x["attention_mask"]

        generated_text = self.model.generate(
            do_sample=True,
            vision_x=vision_x.to(self.model.device),
            lang_x=lang_x_input_ids.to(self.model.device),
            attention_mask=lang_x_attention_mask.to(self.model.device),
            max_new_tokens=self.config.get('max_new_tokens', 512),
            num_beams=self.config.get('num_beams', 3),
            no_repeat_ngram_size=3,
            temperature=self.config.get('temperature', 1.0),
            top_p=self.config.get('top_p', 0.9),
            top_k=self.config.get('top_k', 50),
            length_penalty=self.config.get('length_penalty', 1.0), 
            pad_token_id=self.tokenizer.eos_token_id,
        )
        parsed_output = self.tokenizer.decode(generated_text[0]).split("<answer>")[-1].lstrip().rstrip().split("<|endofchunk|>")[0].lstrip().rstrip().lstrip('"').rstrip('"')
        return parsed_output



if __name__ == "__main__":
    from utils import setup_seeds
    from dataset.test_cases import VLM_TEST_CASES

    setup_seeds()
    model = Otter()

    for test_case in VLM_TEST_CASES:
        pred = model.generate(
            instruction=test_case['instruction'],
            images=test_case['images'],
        )
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Images:\t{test_case["images"]}')
        print(f'Answer:\t{pred}')
        print('-'*20)