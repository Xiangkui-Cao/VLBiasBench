import sys
from PIL import Image 
import torch 

from .base_model import VLM_BaseModel

from mmengine import Config
sys.path.append("./models/shikra")
from mllm.dataset.process_function import PlainBoxFormatter
from mllm.dataset.builder import prepare_interactive
from mllm.utils import draw_bounding_boxes
from mllm.models.builder.build_shikra import load_pretrained_shikra


model_path = './models/shikra/checkpoints/shikra-7b'

model_args = Config(dict(
    type='shikra',
    version='v1',

    # checkpoint config
    cache_dir=None,
    model_name_or_path=model_path,
    vision_tower='./models/shikra/checkpoints/clip-vit-large-patch14',
    pretrain_mm_mlp_adapter=None,

    # model config
    mm_vision_select_layer=-2,
    model_max_length=2048,

    # finetune config
    freeze_backbone=False,
    tune_mm_mlp_adapter=False,
    freeze_mm_mlp_adapter=False,

    # data process config
    is_multimodal=True,
    sep_image_conv_front=False,
    image_token_len=256,
    mm_use_im_start_end=True,

    target_processor=dict(
        boxes=dict(type='PlainBoxFormatter'),
    ),

    process_func_args=dict(
        conv=dict(type='ShikraConvProcess'),
        target=dict(type='BoxFormatProcess'),
        text=dict(type='ShikraTextProcess'),
        image=dict(type='ShikraImageProcessor'),
    ),

    conv_args=dict(
        conv_template='vicuna_v1.1',
        transforms=dict(type='Expand2square'),
        tokenize_kwargs=dict(truncation_size=None),
    ),

    gen_kwargs_set_pad_token_id=True,
    gen_kwargs_set_bos_token_id=True,
    gen_kwargs_set_eos_token_id=True,
))
training_args = Config(dict(
    bf16=False,
    fp16=True,
    device='cuda',
    fsdp=None,
))

class Shikra_7B(VLM_BaseModel):
    def __init__(self, model_path=model_path, model_name='shikra-7b', **kwargs):
        super().__init__(model_name, **kwargs)

        self.model, preprocessor = load_pretrained_shikra(model_args, training_args, **dict())
        self.model.to(dtype=torch.float16, device=torch.device('cuda'))
        preprocessor['target'] = {'boxes': PlainBoxFormatter()}
        self.tokenizer = preprocessor['text']
        self.gen_kwargs = dict(
            use_cache=True,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.config.get("max_new_tokens", 512),
            top_p=self.config.get("top_p", 1.0),
            temperature=self.config.get("temperature", 1.0),
        )
        self.ds = prepare_interactive(model_args, preprocessor)
        self.mode = self.config.get('mode', 'Advanced')

        
        self.task_template = {
                # "SpotCap": "Provide a comprehensive description of the image <image> and specify the positions of any mentioned objects in square brackets.",
                # "Cap": "Summarize the content of the photo <image>.",
                # "GCoT": "With the help of the image <image>, can you clarify my question '<question>'? Also, explain the reasoning behind your answer, and don't forget to label the bounding boxes of the involved objects using square brackets.",
                "VQA": "For this image <image>, I want a simple and direct answer to my question: <question>",
                # "REC": "Can you point out <expr> in the image <image> and provide the coordinates of its location?",
                # "REG": "For the given image <image>, can you provide a unique description of the area <boxes>?",
                # "GC": "Can you give me a description of the region <boxes> in image <image>?",
                "Advanced": "<question>",
            }


    def generate(self, instruction, images, interleaved=False):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        template = self.task_template[self.mode]
        instruction = template.replace("<question>", instruction)
        instruction = self.inst_pre + instruction + self.inst_suff
        image = Image.open(images[0])

        self.ds.set_image(image)
        self.ds.set_message(role=self.ds.roles[0], message=instruction, boxes=None, boxes_seq=None)

        model_inputs = self.ds.to_model_input()
        model_inputs['images'] = model_inputs['images'].to(torch.float16)
        input_ids = model_inputs['input_ids']

        with torch.inference_mode():
            with torch.autocast(dtype=torch.float16, device_type='cuda'):
                output_ids = self.model.generate(**model_inputs, **self.gen_kwargs)
        input_token_len = input_ids.shape[-1]
        response = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        return response



if __name__ == "__main__":
    from utils import setup_seeds
    from dataset.test_cases import VLM_TEST_CASES

    setup_seeds()
    model = Shikra_7B()
    model.mode = 'Advanced'
    for test_case in VLM_TEST_CASES:
        pred = model.generate(
            instruction=test_case['instruction'],
            images=test_case['images'],
        )
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Images:\t{test_case["images"]}')
        print(f'Answer:\t{pred}')
        print('-'*20)


    model.mode = 'VQA'
    for test_case in VLM_TEST_CASES:
        pred = model.generate(
            instruction=test_case['instruction'],
            images=test_case['images'],
        )
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Images:\t{test_case["images"]}')
        print(f'Answer:\t{pred}')
        print('-'*20)