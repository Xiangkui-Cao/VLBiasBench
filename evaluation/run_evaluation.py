
import os
import argparse
from datetime import datetime
import json

from utils import setup_seeds, DictAction
from models.load_model import load_model
from dataset.load_dataset import load_dataset
import logging


valid_models = ['minigpt4_vicuna-7b', 'minigpt4_vicuna-13b', 'minigpt4_llama_2', 'minigpt_v2', 'blip2_flan-t5-xl', 'blip2-opt-3b', 'blip2-opt-7b', 'instructblip_vicuna-7b',  'instructblip_vicuna-13b', 'instructblip_flan-t5-xl', 'instructblip_flan-t5-xxl', 'llava_1.5-7b', 'llava_1.5-13b', 'otter', 'emu2-chat', 'qwen-vl', 'shikra-7b', 'internlm-xcomposer-vl-7b']
valid_datasets = ['close_ended_dataset', 'open_ended_dataset']
parser = argparse.ArgumentParser(description='Example argparse script.')
parser.add_argument('--model_name', choices=valid_models, help='A list of valid models.')
parser.add_argument('--dataset_list', nargs='+', choices=valid_datasets, help='A list of valid datasets.')
parser.add_argument('--cfg_options', type=json.loads,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
parser.add_argument('--comment', type=str, default="", help='Extra flag append to the model name in output dir.')
parser.add_argument('--seed', type=int, default=42, help='The random seed.')
args = parser.parse_args()



setup_seeds(seed=args.seed)
model = load_model(args.model_name, cfg_options=args.cfg_options)
for dataset_name in args.dataset_list:
    dataset = load_dataset(dataset_name)

    output_dir = os.path.join("./outputs", args.model_name if not args.comment else "_".join([args.model_name, args.comment]), dataset_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'config.json'), 'w') as json_file:
        json.dump(model.config, json_file, indent=4)
        json.dump(dataset.config, json_file, indent=4)

    logger = logging.getLogger(args.model_name + "|" + dataset_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.txt'), mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(''))
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(name)s - %(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    result_json = []
    for idx, test_case in enumerate(dataset):
        if "otter" in args.model_name and idx <= 80509:
            continue
        try:
            pred = model.generate(
                instruction=test_case['instruction'],
                images=test_case['images'],
            )
        except Exception as error:
            pred = ""
            logger.error(f'An exception occurred: {error}')
        # pred = model.generate(
                # instruction=test_case["instruction"],
                # images=test_case["images"],
                #)
            
        logger.info(f'ID:\t{idx}')
        logger.info(f'Instruction:\t{test_case["instruction"]}')
        logger.info(f'Images:\t{test_case["images"]}')
        logger.info(f'Answer:\t{pred}')
        logger.info('-' * 60)

        result_case = {
            'id': idx if test_case.get('id') is None else test_case['id'],
            'instruction': test_case['instruction'],
            'in_images': test_case['images'],
            'answer': pred,
        }
        result_json.append(result_case)

    with open(os.path.join(output_dir, 'result.json'), 'w') as json_file:
        json.dump(result_json, json_file, indent=4)

    stream_handler.close()
    file_handler.close()
