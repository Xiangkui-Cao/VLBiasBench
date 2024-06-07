from PIL import Image
import json
import os
import textwrap

# Assuming json_data contains multiple model responses for each ID
# And each model's response is in a separate JSON with the same ID
# For demonstration, I'll create a mock version of the JSON data for multiple models


model_responses = {
    'minigpt4_vicuna-7b': 'vlm_test_20240118-035411',
    'minigpt4_vicuna-13b': 'vlm_test_20240118-035934',
    'minigpt4_llama_2': 'vlm_test_20240118-040516',
    'minigpt_v2': 'vlm_test_20240118-040928',
    'minigpt_v2_vqa': 'vlm_test_20240118-042017',
    'minigpt_v2_grounding': 'vlm_test_20240118-041257',
    'blip2_flan-t5-xl': 'vlm_test_20240118-042131',
    'blip2-opt-3b': 'vlm_test_20240118-042241',
    'blip2-opt-7b': 'vlm_test_20240118-042519',
    'instructblip_vicuna-7b': 'vlm_test_20240118-042706',
    'instructblip_flan-t5-xl': 'vlm_test_20240118-042746',
    # 'instructblip_flan-t5-xxl': 'vlm_test_20231227084504',
    'llava_1.5-7b': 'vlm_test_20240118-042829',
    'llava_1.5-13b': 'vlm_test_20240118-042925',
    'otter': 'vlm_test_20240118-043107',
    'emu2-chat': 'vlm_test_20240118-043632',
}

id_to_plot = 1  # Replace with the actual ID you want to plot


# Create a plot for the specified ID
def collect_answer(model_responses, id):
    # Set up the plot grid
    num_models = len(model_responses)
    
    # Load and display the image
    model_name = next(iter(model_responses))
    json_file = json.load(open(f'../outputs/{model_name}/{model_responses[model_name]}/result.json'))
    item = next(item for item in json_file if item['id'] == id)
    image_path, instruction = item['in_images'][0], item['instruction']

    print(image_path)
    print(f"[Instruction]\n{instruction}\n")


    # Display the model responses
    for i, (model_name, response_file) in enumerate(model_responses.items()):
        json_file = json.load(open(f'../outputs/{model_name}/{response_file}/result.json'))
        item = next(item for item in json_file if item['id'] == id)
        response_text = item['answer']
        # response_text = textwrap.fill(response_text, width=80)  # Set width to desired value

        print(f"[{model_name} Response]\n{response_text}\n")

# Generate the comparison plot for a specific ID
collect_answer(model_responses, id_to_plot)

