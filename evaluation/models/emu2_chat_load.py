from PIL import Image 
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch


from .base_model import VLM_BaseModel


model_path = './models/Emu/Emu2/checkpoints/Emu2-Chat'


class Emu2_Chat(VLM_BaseModel):
    def __init__(self, model_path=model_path, model_name='emu2-chat', **kwargs):
        super().__init__(model_name, **kwargs)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True)  

        device_map = infer_auto_device_map(model, max_memory={0:'18GiB',1:'18GiB',2:'18GiB',3:'18GiB',}, no_split_module_classes=['Block','LlamaDecoderLayer'])  
        # input and output logits should be on same device
        device_map["model.decoder.lm.lm_head"] = 0

        self.model = load_checkpoint_and_dispatch(model, model_path, device_map=device_map).eval()

    def generate(self, instruction, images, interleaved=False):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """

        # `[<IMG_PLH>]` is the image placeholder which will be replaced by image embeddings. 
        # the number of `[<IMG_PLH>]` should be equal to the number of input images
        
        query = self.inst_pre + instruction + self.inst_suff
        if not interleaved:
            query = "[<IMG_PLH>]" + query
            
        assert query.count("[<IMG_PLH>]") == len(images)            

        images = [Image.open(image).convert('RGB') for image in images]

        inputs = self.model.build_input_ids(
            text=[query],
            tokenizer=self.tokenizer,
            image=images

        )

        with torch.no_grad():
            outputs = self.model.generate(
                do_sample=True,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image=inputs["image"].to(torch.bfloat16),
                max_new_tokens=self.config.get("max_new_tokens", 64),
                num_beams=self.config.get("num_beams", 1),
                temperature=self.config.get("temperature", 1.0),
                top_k=self.config.get("top_k", 50),
                top_p=self.config.get("top_p", 1.0),
                length_penalty=self.config.get("length_penalty", -1),
                low_memory=self.config.get("low_memory", False),
            )

        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return output_text



if __name__ == "__main__":
    from utils import setup_seeds
    from dataset.test_cases import VLM_TEST_CASES

    setup_seeds()
    model = Emu2_Chat()

    for test_case in VLM_TEST_CASES:
        pred = model.generate(
            instruction=test_case['instruction'],
            images=test_case['images'],
        )
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Images:\t{test_case["images"]}')
        print(f'Answer:\t{pred}')
        print('-'*20)
