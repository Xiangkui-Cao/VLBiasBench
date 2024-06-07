cd ..
set -e
source /data/yuanzheng/anaconda3/etc/profile.d/conda.sh

DATASET_LIST="vlbold_xl-pi"

# conda activate emu2
# CUDA_VISIBLE_DEVICES=4,5,6,7 python run_evaluation.py --model_name 'emu2-chat' --dataset_list $DATASET_LIST

# conda activate llava
# CUDA_VISIBLE_DEVICES=4,5 python run_evaluation.py --model_name 'llava_1.5-13b' --dataset_list $DATASET_LIST

# conda activate otter
# CUDA_VISIBLE_DEVICES=4,5 python run_evaluation.py --model_name 'otter' --dataset_list $DATASET_LIST

conda activate minigptv
# CUDA_VISIBLE_DEVICES=4 python run_evaluation.py --model_name 'minigpt4_vicuna-7b' --dataset_list $DATASET_LIST 
# CUDA_VISIBLE_DEVICES=4 python run_evaluation.py --model_name 'minigpt4_vicuna-13b' --dataset_list $DATASET_LIST 
# CUDA_VISIBLE_DEVICES=4 python run_evaluation.py --model_name 'minigpt4_llama_2' --dataset_list $DATASET_LIST 
CUDA_VISIBLE_DEVICES=6 python run_evaluation.py --model_name 'minigpt_v2' --dataset_list $DATASET_LIST 
CUDA_VISIBLE_DEVICES=6 python run_evaluation.py --model_name 'minigpt_v2' --dataset_list $DATASET_LIST --cfg_options "{\"inst_pre\": \"[grounding] \"}" --comment "grounding"

# conda activate lavis
# CUDA_VISIBLE_DEVICES=4 python run_evaluation.py --model_name 'blip2_flan-t5-xl' --dataset_list $DATASET_LIST 
# CUDA_VISIBLE_DEVICES=4 python run_evaluation.py --model_name 'blip2-opt-3b' --dataset_list $DATASET_LIST 
# CUDA_VISIBLE_DEVICES=4 python run_evaluation.py --model_name 'blip2-opt-7b' --dataset_list $DATASET_LIST 
# CUDA_VISIBLE_DEVICES=4 python run_evaluation.py --model_name 'instructblip_vicuna-7b' --dataset_list $DATASET_LIST 
# CUDA_VISIBLE_DEVICES=4 python run_evaluation.py --model_name 'instructblip_flan-t5-xl' --dataset_list $DATASET_LIST 
# ## CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --model_name 'instructblip_flan-t5-xxl' --dataset_list $DATASET_LIST

# conda activate llava
# CUDA_VISIBLE_DEVICES=4 python run_evaluation.py --model_name 'llava_1.5-7b' --dataset_list $DATASET_LIST
# CUDA_VISIBLE_DEVICES=2,3 python run_evaluation.py --model_name 'llava_1.5-13b' --dataset_list $DATASET_LIST

# conda activate otter
# CUDA_VISIBLE_DEVICES=2,3 python run_evaluation.py --model_name 'otter' --dataset_list $DATASET_LIST


