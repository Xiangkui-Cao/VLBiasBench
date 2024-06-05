# VLBiasBench
VLBiasBench： A large-scale dataset composed of high-quality synthetic images aimed at evaluating social biases in LVLMs{ font-family: Arial, sans-serif; margin: 0; padding: 0; display: flex; flex-direction: column; align-items: center; line-height: 1.6; /\* 改善文本的可读性 \*/ } /\* 其他样式不变，这里只展示修改部分 \*/ .image-container { text-align: center; /\* 确保图片和图片标题居中 \*/ } /\* 合并.image-container img的样式 \*/ /\* 保持其他图片的样式不变 \*/ /\* 保持其他图片的样式不变 \*/ /\*.image-container img {\*/ /\* width:95%;\*/ /\* display: block; !\* img默认是inline元素，这里改为block以便使用margin auto居中 \*!\*/ /\* margin-left: auto;\*/ /\* margin-right: auto;\*/ /\*}\*/ /\*!\* 移除了针对 .overview-img 和 .subtasks-img 设置宽度的规则 \*!\*/ /\*!\* 特别为第二张图设置一个更小的宽度，确保这个规则在样式表中的位置 \*!\*/ /\*.subtasks-img {\*/ /\* width: 40%; !\* 设置第二张图宽度为40% \*!\*/ /\*}\*/ /\* 确保.container内的所有元素都能得到适当的居中处理 \*/ .container { text-align: left; /\* 这将影响到容器内的行内元素和文本内容 \*/ margin-top: 20px; width: 80%; margin-left: auto; margin-right: auto; /\* 水平居中 \*/ } .center { text-align: center; } .author-info sup { font-size: smaller; } .author-info, .institution, .email-contact { display: flex; justify-content: center; } .email-contact { margin-bottom: 20px; } /\* 新增或修改的图片样式 \*/ .overview-img { width: 95%; height: auto; display: block; margin-left: auto; margin-right: auto; } .subtasks-img { width: 40%; height: auto; display: block; margin-left: auto; margin-right: auto; } /\*.overview-img, .subtasks-img {\*/ /\* width: 96%;\*/ /\* height: auto;\*/ /\* display: block;\*/ /\* margin-left: auto;\*/ /\* margin-right: auto;\*/ /\*}\*/ table { /\*表格位于页面居中位置\*/ margin-left: auto; margin-right: auto; width: 50%; border-collapse: collapse; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); /\* 添加阴影效果 \*/ } table, th, td { border: 1px solid black; } th, td { padding: 8px; text-align: left; } th { background-color: #f2f2f2; } tr:nth-child(even) { background-color: #f9f9f9; /\* 为偶数行添加条纹背景 \*/ } a { color: #007bff; /\* 设置链接颜色 \*/ text-decoration: none; /\* 移除下划线 \*/ } a:hover { text-decoration: underline; /\* 鼠标悬停时添加下划线 \*/ } @media (max-width: 768px) { .container, .image-container img { width: 95%; /\* 在小屏设备上调整容器和图片宽度 \*/ } }

![](figure/icon0.webp)

## 🎨Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs

🔗The link to our project is [\[here\]](https://github.com/Robin-WZQ/LADS)

### Overview🔍

![Overview of the automatic pipeline in Dysca](./figure/dysca_framework.svg)

**Figure 1. Overview of the automatic pipeline in Dysca for generating VQAs, cleaning VQAs and evaluating LVLMs.**

  

![The available subtasks of our Dysca](./figure/subtasks.svg)

**Figure 2. The available subtasks of our Dysca.**

**Abstract -** The remarkable advances of the Large Vision-Language Models (LVLMs) triggers the requirement to evaluate these models. However, most evaluation benchmark conduct questions by selecting images from previous benchmark rather than using in-the-wild images, resulting in a potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic images and clean scenarios, leaving the multi-stylized images and complex scenarios unexplored. In response to these challenges, we propose Dysca, a dynamic and scalable benchmark for evaluating LVLMs by leveraging synthesis images. Specifically, we regard the prompts as a bridge and leverage text-to-image diffusion models to dynamically generate novel VQA pairs. We consider 57 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Print Attack, Adversarial Attack, and Corrupted) and 3 question types (i.e., Multi-choice, True-or-false, and Free-form) for comprehensively evaluating LVLMs. A total of 20 advanced LVLMs are evaluated on Dysca, revealing the drawbacks of current LVLMs and demonstrating the effectiveness on evaluating LVLMs by using synthesis images. Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks, scenarios and models.

### Key statistics of Dysca📊

| Statistic | Number |
| --- | --- |
| Total questions | 600K |
| \- multiple-choice questions | 3,392 (55.2%) |
| \- Free-form questions | 2,749 (44.8%) |
| \- True-or-false | 5,261 (85.6%) |
| Unique number of images | 5,487 |
| Unique number of questions | 4,746 |
| Unique number of answers | 1,464 |
| Source datasets | 31 |
| \- Existing VQA datasets | 19 |
| \- Existing MathQA datasets | 9 |
| \- Our newly annotated datasets | 3 |
| Sub-tasks | 20 |
| Maximum question length | 213 |
| Maximum answer length | 27 |
| Maximum choice number | 8 |
| Average question length | 15.6 |
| Average answer length | 1.2 |
| Average choice number | 3.4 |

### Examples of Dysca📸

### Evaluation Results🏆

#### Clean Scenario

| Rank | Model | Score |
| --- | --- | --- |
| 🥇 | A |  |
| 🥈 | B |  |
| 🥉 | C |  |
| 4 | D |  |
| 5 | E |  |

### Guidelines🧭

#### The folder "category"

The category folder contains all of Dysca's source material. It contains the following:

                `category ├── People │   ├── Age.txt │   ├── Emotion.txt │   ├── Gender.txt │   ├── Race.txt ├── Actions.txt ├── Profession.txt ├── Celebrity.txt ├── Animal.txt ├── Plant.txt ├── Food.txt ├── Object.txt ├── Landmarks.txt ├── Clothes.txt ├── Movie.txt ├── TV shows.txt ├── Anime.txt ├── Color.txt ├── Background.txt ├── Styles.json ├── ocr_text.json ├── text.txt`
                    
    

#### Generating Prompts, Questions and Answers

The `./code/prompt_question_answer.py` is used to generate the source data for the dataset, i.e., all the prompts, questions, and answers used to generate the images. Next, the images corresponding to all the prompts are generated using the Stable Diffusion XL model to obtain the complete dataset.

Specifically, in the main function's parameters:

+   "tasks" parameter can choose one from "recognition", "OCR".
+   "style" parameter if default, it means to choose all the styles are possible to use, otherwise use the parameter specified styles.
+   "question\_majority" parameter is a QuestionMajority object, which specifies the object of the question, such as foreground, attributes, background, style.
+   "question\_types" specifies which question types are selected.
+   "prompt\_num" specifies the upper limit of generated prompts.
+   "save\_dir" refers to the folder where the results are saved.

### Related projects🔗

+   [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)
+   [EMU2](https://github.com/baaivision/Emu)
+   [InstructBLIP](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip)
+   [LLaVA-1.5](https://github.com/haotian-liu/LLaVA)
+   [miniGPT4](https://github.com/Vision-CAIR/MiniGPT-4)
+   [miniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4)
+   [Otter](https://github.com/Vision-CAIR/MiniGPT-4)
+   [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
+   [Shikra](https://github.com/shikras/shikra)
+   [InternLM-XComposer](https://github.com/InternLM/InternLM-XComposer)

### Cite our work📝

        `coming soon...`
        
    

### Acknowledgement

This section will include acknowledgements...
