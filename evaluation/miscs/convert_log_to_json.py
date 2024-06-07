import json

# 文件路径
input_file = '../outputs/emu2-chat/mmcbench_20240412-151205/log.txt'
output_file = '../outputs/emu2-chat/mmcbench_20240412-151205/result.json'

# 初始化一个空列表来存储所有条目
data = []

try:
    with open(input_file, 'r') as file:
        content = file.read().split('------------------------------------------------------------')
        for entry in content:
            if entry.strip():  # 确保条目不是空的
                lines = entry.strip().split('\n')
                entry_dict = {
                    "id": int(lines[0].split('\t')[1].strip()),
                    "instruction": lines[1].split('\t')[1].strip(),
                    "in_images": [lines[2].split('\t')[1].strip().strip('[]').replace("'", "").strip()],
                    "answer": lines[3].split('\t')[1].strip()
                }
                data.append(entry_dict)

    # 将数据写入JSON文件
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print("JSON文件已成功创建。")
except Exception as e:
    print(f"发生错误: {e}")
