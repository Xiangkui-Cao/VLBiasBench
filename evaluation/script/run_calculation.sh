cd ..
set -e
source /data/yuanzheng/anaconda3/etc/profile.d/conda.sh

MODEL_LIST="minigpt4_vicuna-13b blip2_flan-t5-xl llava_1.5-13b otter" # emu2-chat"
# MODEL_LIST="minigpt4_vicuna-7b minigpt4_vicuna-13b minigpt4_llama_2 minigpt_v2 minigpt_v2_vqa blip2_flan-t5-xl blip2-opt-3b blip2-opt-7b instructblip_vicuna-7b instructblip_vicuna-13b instructblip_flan-t5-xl instructblip_flan-t5-xxl llava_1.5-7b llava_1.5-13b otter shikra-7b shikra-7b_vqa internlm-xcomposer-vl-7b qwen-vl emu2-chat"
# MODEL_LIST="minigpt4_vicuna-7b minigpt4_vicuna-13b minigpt4_llama_2 minigpt_v2 minigpt_v2_grounding blip2_flan-t5-xl blip2-opt-3b blip2-opt-7b instructblip_vicuna-7b instructblip_vicuna-13b instructblip_flan-t5-xl instructblip_flan-t5-xxl llava_1.5-7b llava_1.5-13b otter shikra-7b internlm-xcomposer-vl-7b qwen-vl" # emu2-chat"
# MODEL_LIST="blip2-opt-3b"

DATASET_LIST="vlbold-r" # pope amber-g amber-d vlbold-ri vlbold_xl-pi vlbold_xl-g vlbold_xl-ri mathvista mmbench sciqa ptqa


conda activate base
CUDA_VISIBLE_DEVICES=5 python run_calculation.py --model_list $MODEL_LIST --dataset_list $DATASET_LIST

