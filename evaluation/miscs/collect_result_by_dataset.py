import os
import zipfile
import sys
sys.path.append("..")
from utils import get_result_dir
dataset_name = "vlbold-r"
# model_list = "minigpt4_vicuna-7b minigpt4_vicuna-13b minigpt4_llama_2 minigpt_v2 minigpt_v2_vqa minigpt_v2_grounding blip2_flan-t5-xl blip2-opt-3b blip2-opt-7b instructblip_vicuna-7b instructblip_vicuna-13b instructblip_flan-t5-xl instructblip_flan-t5-xxl llava_1.5-7b llava_1.5-13b otter shikra-7b shikra-7b_vqa internlm-xcomposer-vl-7b qwen-vl emu2-chat" # gaive

model_list = "minigpt4_vicuna-13b blip2_flan-t5-xl  llava_1.5-13b otter" # emu2-chat"


# model_list = "minigpt4_vicuna-7b minigpt4_vicuna-13b minigpt4_llama_2 minigpt_v2 minigpt_v2_vqa blip2_flan-t5-xl blip2-opt-3b blip2-opt-7b instructblip_vicuna-7b instructblip_vicuna-13b instructblip_flan-t5-xl instructblip_flan-t5-xxl llava_1.5-7b llava_1.5-13b otter shikra-7b shikra-7b_vqa internlm-xcomposer-vl-7b qwen-vl emu2-chat"


# model_list = "minigpt4_vicuna-7b minigpt4_vicuna-13b minigpt4_llama_2 minigpt_v2 minigpt_v2_grounding blip2_flan-t5-xl blip2-opt-3b blip2-opt-7b instructblip_vicuna-7b instructblip_vicuna-13b instructblip_flan-t5-xl instructblip_flan-t5-xxl llava_1.5-7b llava_1.5-13b otter shikra-7b internlm-xcomposer-vl-7b qwen-vl emu2-chat"


output_dir = "../outputs"
zip_filename = dataset_name + ".zip"

    
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for model_name in model_list.split():
        model_dir = get_result_dir(model_name, dataset_name, output_root=output_dir)[0]
        for root, dirs, files in os.walk(model_dir):
            rename_root = '/'.join(root.split('/')[-2:])
            for file in files:
                zipf.write(os.path.join(root, file), os.path.join(rename_root, file))


