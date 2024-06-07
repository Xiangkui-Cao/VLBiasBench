from PIL import Image 
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from diffusers import DiffusionPipeline


from .base_model import T2I_BaseModel


model_path = './models/Emu/Emu2/checkpoints/Emu2-Gen'


class Emu2_Gen(T2I_BaseModel):
    def __init__(self, model_path=model_path, model_name='emu2-gen', **kwargs):
        super().__init__(model_name, **kwargs)
        
        tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer")

        with init_empty_weights():
            multimodal_encoder = AutoModelForCausalLM.from_pretrained(
                f"{model_path}/multimodal_encoder",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                variant="bf16"
            )

        device_map = infer_auto_device_map(multimodal_encoder, max_memory={1:'20GiB',2:'20GiB',3:'20GiB',4:'20GiB',}, no_split_module_classes=['Block','LlamaDecoderLayer'])  
        # input and output logits should be on same device
        device_map["model.decoder.lm.lm_head"] = 0

        multimodal_encoder = load_checkpoint_and_dispatch(multimodal_encoder, f"{model_path}/multimodal_encoder", device_map=device_map).eval()
        self.pipe = DiffusionPipeline.from_pretrained(
            model_path,
            custom_pipeline=model_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="bf16",
            multimodal_encoder=multimodal_encoder,
            tokenizer=tokenizer,
        )
        self.pipe.unet.to("cuda:0")
        self.pipe.vae.to("cuda:0")
        self.pipe.safety_checker.to("cuda:0")
        # for name, para in self.pipe.unet.named_parameters():
        #     print(name, para.device)
        # for name, para in self.pipe.vae.named_parameters():
        #     print(name, para.device)
        # for name, para in self.pipe.multimodal_encoder.named_parameters():
        #     print(name, para.device)
        # for name, para in self.pipe.safety_checker.named_parameters():
        #     print(name, para.device)


    def generate(self, instruction, images=[], **kwargs):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (PIL.Image) a generated image in type of PIL.Image
        """
        prompt = []
        for image in images:
            prompt.append(Image.open(image).convert('RGB'))

        instruction = self.inst_pre + instruction + self.inst_suff
        prompt.append(instruction)

        ret = self.pipe(
            inputs = prompt,
            height = self.config.get("height", 1024),
            width = self.config.get("width", 1024),
            num_inference_steps = self.config.get("num_inference_steps", 50),
            guidance_scale = self.config.get("guidance_scale", 3.),
        ).image
        return ret



if __name__ == "__main__":
    from utils import setup_seeds
    from dataset.test_cases import T2I_TEST_CASES, TI2I_TEST_CASES

    setup_seeds()
    model = Emu2_Gen()

    for ix, test_case in enumerate(T2I_TEST_CASES):
        img = model.generate(
            instruction=test_case['instruction'],
        )
        save_path = "./outputs/temp/gen_{}.png".format(ix)
        img.save(save_path)
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Save to:\t{save_path}')
        print('-'*20)


    for ix, test_case in enumerate(TI2I_TEST_CASES):
        img = model.generate(
            instruction=test_case['instruction'],
            images=test_case['images'],
        )
        save_path = "./outputs/temp/edit_{}.png".format(ix)
        img.save(save_path)
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Save to:\t{save_path}')
        print('-'*20)



# import cv2
# from diffusers import DiffusionPipeline
# import numpy as np
# from PIL import Image
# import requests
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # For the first time of using,
# # you need to download the huggingface repo "BAAI/Emu2-GEN" to local first
# path = "path to local BAAI/Emu2-GEN"

# multimodal_encoder = AutoModelForCausalLM.from_pretrained(
#     f"{path}/multimodal_encoder",
#     trust_remote_code=True,
#     torch_dtype=torch.bfloat16,
#     use_safetensors=True,
#     variant="bf16"
# )
# tokenizer = AutoTokenizer.from_pretrained(f"{path}/tokenizer")

# pipe = DiffusionPipeline.from_pretrained(
#     path,
#     custom_pipeline="pipeline_emu2_gen",
#     torch_dtype=torch.bfloat16,
#     use_safetensors=True,
#     variant="bf16",
#     multimodal_encoder=multimodal_encoder,
#     tokenizer=tokenizer,
# )

# # For the non-first time of using, you can init the pipeline directly
# pipe = DiffusionPipeline.from_pretrained(
#     path,
#     custom_pipeline="pipeline_emu2_gen",
#     torch_dtype=torch.bfloat16,
#     use_safetensors=True,
#     variant="bf16",
# )

# pipe.to("cuda")

# # text-to-image
# prompt = "impressionist painting of an astronaut in a jungle"
# ret = pipe(prompt)
# ret.image.save("astronaut.png")

# # image editing
# image = Image.open(requests.get('https://github.com/baaivision/Emu/Emu2/examples/dog.jpg?raw=true',stream=True).raw).convert('RGB')
# prompt = [image, "wearing a red hat on the beach."]
# ret = pipe(prompt)
# ret.image.save("dog_hat_beach.png")

# # grounding generation
# def draw_box(left, top, right, bottom):
#     mask = np.zeros((448, 448, 3), dtype=np.uint8)
#     mask = cv2.rectangle(mask, (left, top), (right, bottom), (255, 255, 255), 3)
#     mask = Image.fromarray(mask)
#     return mask

# dog1 = Image.open(requests.get('https://github.com/baaivision/Emu/Emu2/examples/dog1.jpg?raw=true',stream=True).raw).convert('RGB')
# dog2 = Image.open(requests.get('https://github.com/baaivision/Emu/Emu2/examples/dog2.jpg?raw=true',stream=True).raw).convert('RGB')
# dog3 = Image.open(requests.get('https://github.com/baaivision/Emu/Emu2/examples/dog3.jpg?raw=true',stream=True).raw).convert('RGB')
# dog1_mask = draw_box( 22,  14, 224, 224)
# dog2_mask = draw_box(224,  10, 448, 224)
# dog3_mask = draw_box(120, 264, 320, 438)

# prompt = [
#     "<grounding>",
#     "An oil painting of three dogs,",
#     "<phrase>the first dog</phrase>"
#     "<object>",
#     dog1_mask,
#     "</object>",
#     dog1,
#     "<phrase>the second dog</phrase>"
#     "<object>",
#     dog2_mask,
#     "</object>",
#     dog2,
#     "<phrase>the third dog</phrase>"
#     "<object>",
#     dog3_mask,
#     "</object>",
#     dog3,
# ]
# ret = pipe(prompt)
# ret.image.save("three_dogs.png")

# # Autoencoding
# # to enable the autoencoding mode, you can only input exactly one image as prompt
# # if you want the model to generate an image,
# # please input extra empty text "" besides the image, e.g.
# #   autoencoding mode: prompt = image or [image]
# #   generation mode: prompt = ["", image] or [image, ""]
# prompt = Image.open("./examples/doodle.jpg").convert("RGB")
# ret = pipe(prompt)
# ret.image.save("doodle_ae.png")
