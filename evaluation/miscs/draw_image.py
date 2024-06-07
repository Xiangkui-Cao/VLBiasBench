import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import json
import os
import textwrap

# Assuming json_data contains multiple model responses for each ID
# And each model's response is in a separate JSON with the same ID
# For demonstration, I'll create a mock version of the JSON data for multiple models


model_responses = {
    'minigpt4_vicuna-7b': 'vlm_test_20231227083504',
    'minigpt4_vicuna-13b': 'vlm_test_20231227083703',
    'minigpt4_llama_2': 'vlm_test_20231227083935',
    'minigpt_v2': 'vlm_test_20231227084126',
    'blip2_flan-t5-xl': 'vlm_test_20231227084312',
    'blip2-opt-3b': 'vlm_test_20231227084338',
    'blip2-opt-7b': 'vlm_test_20231227084423',
    'instructblip_vicuna-7b': 'vlm_test_20231227084501',
    'instructblip_flan-t5-xl': 'vlm_test_20231227084538',
    # 'instructblip_flan-t5-xxl': 'vlm_test_20231227084504',
    'llava_1.5-7b': 'vlm_test_20231227084616',
    'llava_1.5-13b': 'vlm_test_20231227084720',
    'otter': 'vlm_test_20231227084826',
}

save_dir = "./vlm_test"
id_to_plot = 2  # Replace with the actual ID you want to plot


# Create a plot for the specified ID
def create_comparison_plot(model_responses, id, save_dir):
    # Set up the plot grid
    fig = plt.figure(figsize=(10, 12))
    num_models = len(model_responses)
    gs = gridspec.GridSpec(num_models+2, 1, height_ratios=[int(0.3 * num_models), 1]+[1,]*num_models)
    
    # Load and display the image
    model_name = next(iter(model_responses))
    json_file = json.load(open(f'../outputs/{model_name}/{model_responses[model_name]}/result.json'))
    item = next(item for item in json_file if item['id'] == id)
    image_path, instruction = item['images'][0], item['instruction']




    ax0 = fig.add_subplot(gs[0])
    image = Image.open(os.path.join("..", image_path))
    ax0.imshow(image)
    ax0.set_title("[Input Image]", fontsize=14, c='b')
    ax0.axis('off')
    
    # Display the instruction
    ax1 = fig.add_subplot(gs[1])
    ax1.axis('off')
    ax1.text(0.5, 1.0, f"[Instruction]\n{instruction}", va='center', ha='center', fontsize=14, c='b')
    
    # Display the model responses
    for i, (model_name, response_file) in enumerate(model_responses.items()):
        json_file = json.load(open(f'../outputs/{model_name}/{response_file}/result.json'))
        item = next(item for item in json_file if item['id'] == id)
        response_text = item['answer']
        print(response_text)
        response_text = textwrap.fill(response_text, width=80)  # Set width to desired value
        print(response_text)


        ax = fig.add_subplot(gs[i+2])
        ax.axis('off')
        # ax.text(0.5, 0.5, f"[{model_name} Response]\n{response_text}", va='center', ha='center', fontsize=14, c='b')

        ax.annotate(f"[{model_name}]", (0.5, 1.0),
                      xycoords='axes fraction', ha='center', va='center', fontsize=14, color='blue',
                      bbox=dict(facecolor='none', edgecolor='none', pad=10))

        ax.annotate(f"{response_text}", (0.5, 0.75),
                      xycoords='axes fraction', ha='center', va='center', fontsize=14, color='black',
                      bbox=dict(facecolor='none', edgecolor='none', pad=10))

    # Adjust layout
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    
    # Save the figure
    os.makedirs(save_dir, exist_ok=True)
    output_filename = os.path.join(save_dir, f'id_{id}.png')
    plt.savefig(output_filename)
    plt.close()
    
    return output_filename

# Assuming the instruction and image path is the same for all models for a given ID
instruction = "Please determine whether the image is real or fake, and elaborate on the reasons."
image_path = '../dataset/test_images/KFC-20-for-20-Nuggets.jpg'

# Generate the comparison plot for a specific ID
output_file = create_comparison_plot(model_responses, id_to_plot, save_dir)

print(output_file)
