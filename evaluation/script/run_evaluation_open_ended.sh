cd ..
set -e
source /mnt/sdb1/caoxiangkui/anaconda3/etc/profile.d/conda.sh

DATASET_LIST="open_ended_dataset"

conda activate emu2
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_evaluation.py --model_name 'emu2-chat' --dataset_list $DATASET_LIST

conda activate internlm 
CUDA_VISIBLE_DEVICES=0,1 python run_evaluation.py --model_name 'internlm-xcomposer-vl-7b' --dataset_list $DATASET_LIST


conda activate llava
CUDA_VISIBLE_DEVICES=0,1 python run_evaluation.py --model_name 'llava_1.5-13b' --dataset_list $DATASET_LIST


conda activate lavis
CUDA_VISIBLE_DEVICES=0,1 python run_evaluation.py --model_name 'instructblip_vicuna-13b' --dataset_list $DATASET_LIST 
CUDA_VISIBLE_DEVICES=0,1 python run_evaluation.py --model_name 'instructblip_flan-t5-xxl' --dataset_list $DATASET_LIST

conda activate llava
CUDA_VISIBLE_DEVICES=0,1 python run_evaluation.py --model_name 'llava_1.5-13b' --dataset_list $DATASET_LIST

conda activate otter
CUDA_VISIBLE_DEVICES=0,1 python run_evaluation.py --model_name 'otter' --dataset_list $DATASET_LIST



conda activate qwen-vl
CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --model_name 'qwen-vl' --dataset_list $DATASET_LIST


conda activate shikra
CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --model_name 'shikra-7b' --dataset_list $DATASET_LIST
CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --model_name 'shikra-7b' --dataset_list $DATASET_LIST --cfg_options "{\"mode\": \"VQA\"}" --comment "vqa"

conda activate minigptv
CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --model_name 'minigpt4_vicuna-7b' --dataset_list $DATASET_LIST
CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --model_name 'minigpt4_vicuna-13b' --dataset_list $DATASET_LIST
CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --model_name 'minigpt4_llama_2' --dataset_list $DATASET_LIST
CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --model_name 'minigpt_v2' --dataset_list $DATASET_LIST
CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --model_name 'minigpt_v2' --dataset_list $DATASET_LIST --cfg_options "{\"inst_pre\": \"[vqa] \"}" --comment "vqa"

conda activate lavis
CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --model_name 'blip2_flan-t5-xl' --dataset_list $DATASET_LIST 
CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --model_name 'blip2-opt-3b' --dataset_list $DATASET_LIST 
CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --model_name 'blip2-opt-7b' --dataset_list $DATASET_LIST 
CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --model_name 'instructblip_vicuna-7b' --dataset_list $DATASET_LIST 
CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --model_name 'instructblip_flan-t5-xl' --dataset_list $DATASET_LIST 

conda activate llava
CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --model_name 'llava_1.5-7b' --dataset_list $DATASET_LIST

