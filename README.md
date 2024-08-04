# 藏文心理健康支持对话数据集(Tibetan_Mental)与大模型(Tibetan_Mental_Chat)

## 一、项目简介
**Tibetan_Mental_Chat(藏文心理健康对话机器人)** 是一个开源的藏文心理健康支持对话的大语言模型项目，支持QLoRA和全量参数微调Baichuan2、CodeLLaMA、LLaMA2、LLaMA、Qwen、Baichuan、ChatGLM2、InternLM、Ziya、Bloom、XVERSE等开源模型。


🔔 本项目主要内容如下：
- 📗 支持全量参数指令微调、QLoRA低成本高效指令微调、其中QLoRA是我们主推的一种高效的训练方式。
- 📗 支持绝大部分主流的开源大模型，如Baichuan2、CodeLLaMA、LLaMA2、LLaMA、Qwen、Baichuan、ChatGLM2、InternLM、Ziya、Bloom、XVERSE等。
- 📗 支持lora与base model进行权重合并，推理更便捷。

## 二、系统Demo
![面向藏语的心理健康咨询大模型平台](https://github.com/Shajiu/Tibetan_Mental_Health_Chat/blob/main/demo.png#pic_center=600x200)



## 三、安装环境
在requirements.txt下固定了几个主要的python包的版本，执行如下脚本即可。

**注意：Baichuan2需要安装pytorch 2.0。除Baichuan2以外，其他模型的训练，我们均在torch==1.13上进行训练。**
```bash
pip install requirements.txt
```

## 四、模型列表

🔔 使用本项目的训练代码，以及上述训练数据，我们训练并开源了以下模型。

藏文模型：

| 模型                                                                             | 基座模型                                | Max Length |
|--------------------------------------------------------------------------------|-------------------------------------|------------|
| [Tibetan_Baichuan2_7B_Mental_Health](https://huggingface.co/shajiu/Tibetan_Baichuan2_7B_Mental_Health) | baichuan-inc/Baichuan2-7B-Base     |1024     |  
| [Tibetan_Llama2_13B_Mental_Health](https://huggingface.co/shajiu/Tibetan_Llama2_13B_Mental_Health)   | 在Llama2_13B上做了词汇表扩充/继续毓预训练/再指令微调      |1024     |


## 五、训练数据
🔔 目前本项目主要整理了如下指令数据集，并将其整理成统一的数据格式：

| 数据集                                                                                                           | 介绍                                                                                                      |
|---------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| [Tibetan_Mental_Health](https://huggingface.co/datasets/shajiu/Tibetan_Mental_Health)                             | 藏文心理健康支持对话数据集(Tibetan_Mental)，数据量为5万条多轮对话。 |

训练数据为jsonl格式，每一行的数据格式如下，其中conversation字段是必需的，可以根据实际需求添加或删除其他字段。
可以参考项目中的Tibetan_Mental_Health.json文件。
```json
{
	"conversation_id": 0,
	"category": "藏语健康多轮对话",
	"conversation": [{
		"human": "ང་རང་ཉེ་ལམ་མགོ་འཁོར་ཡོད།མཐོ་གསུམ་རྗེས་ཀྱི་ང་རང་མཐོ་རྒྱུགས་སྔོན་གྱི་འཇིགས་སྐྲག་ལས་གཞན་ད་དུང་རང་ཉིད་སྐྱེས་མ་ཐག་ནས་བཟུང་ལྷག་མ་ཞིག་ཡིན་པར་འདོད།དོན་སྙིང་ཅི་ཡང་མེད།ང་ལ་གྲོགས་པོ་མེད།ཁེར་རྐྱང་གིས་བཟོད་མི་ཐུབ།ཁྱེད་ཀྱིས་ང་ལ་བསམ་འཆར་འགའ་འདོན་ཐུབ་བམ།",
		"assistant": "ཁྱོད་ཀྱི་སྐད་ཆ་མཐོང་རྗེས།ངས་ཁྱོད་ལ་སེམས་ཁྲལ་ཆེན་པོ་སྐྱེས་བྱུང་།འདི་ནི་རྒྱུན་མཐོང་གི་མཐོ་འབྲིང་འཚོ་བའི་རྣམ་པ་རེད།ཐོག་མར་བཤད་འདོད་པ་ནི།ངས་ཁྱོད་ཀྱི་ཆོད་སེམས་མཐོང་བྱུང་།འདི་ནི་མི་མང་པོ་ཞིག་ལ་མེད་པ་རེད།མཐོ་རྒྱུགས་ཀྱི་ཆེས་གལ་ཆེ་བ་ནི་སེམས་ཁམས་རེད།ཁྱོད་ལ་རྒྱལ་ཁ་ཐོབ་པའི་སེམས་ཁམས་དགོས་པ་རེད།"
	}, {
		"human": "ཁྱོད་ཀྱིས་སྐུལ་མ་བྱས་པར་བཀའ་དྲིན་ཆེ།ངས་དངོས་གནས་ལྷོད་གཡེང་བྱེད་འདོད།འོན་ཀྱང་རང་ཉིད་ཀྱི་དམིགས་འབེན་དང་སྒུལ་ཤུགས་ཤོར་མི་རུང་བར་འདོད།",
		"assistant": "གལ་ཏེ་ཁྱོད་ཀྱིས་འཆར་གཞི་ཅི་ལྟར་བཟོ་དགོས་པ་མི་ཤེས་ན།ངས་བསམ་འཆར་འགའ་འདོན་ཐུབ།"
	}, {
		"human": "འོ་ན་ངས་འཆར་གཞི་ཅི་ལྟར་བཟོ་དགོས་སམ།",
		"assistant": "མཐོ་གཅིག་ནས་འགོ་བརྩམས་ཏེ་གོམ་པ་གང་རེ་གང་རེ་སྤོས་ན།འགོ་རྩོམ་ས་དམའ་ཞིང་།འདྲི་གཞིའི་འཐབ་རྩལ་ལ་ཚོད་ལྟ་བྱེད་དགོས།སྔོན་ལ་རྨང་གཞིའི་གཤར་སྦྱང་འགའ་བྱེད་པ་དང་།རིམ་བཞིན་དཀའ་ཚད་ཇེ་ཆེར་བཏང་ན།ཁྱོད་ལ་རིམ་བཞིན་གྲུབ་འབྲས་ཐོབ་པའི་ཚོར་བ་ཡོད་སྲིད།དེ་དང་མཉམ་དུ།རིགས་སོ་སོའི་ཁ་བྱང་ཚང་མ་བསྒྲུབ་དགོས།འདི་ལྟར་བྱས་ན་ཁྱོད་ཀྱིས་སྔར་ལས་མགྱོགས་པའི་སྒོ་ནས་འདྲི་གཞི་ལ་གོ་བ་ལེན་པ་དང་དྲིས་ལན་འདེབས་ཐུབ།"
	}, {
		"human": "གལ་ཏེ་ངས་ཁ་བྱང་ཁ་ཤས་ལ་གོ་བ་ལེན་མི་ཐུབ་ན་ཅི་ཞིག་བྱ།",
		"assistant": "ངོ་ཚ་མི་དགོས།དགེ་རྒན་ལ་འདྲི་རུ་སོང་།སྤོབས་པ་ཆེན་པོས་ཁོ་ཚོར་ཁྱོད་ཀྱིས་གནད་དོན་འདིར་གོ་བ་ལེན་མི་ཐུབ་ཅེས་ཤོད།དགེ་རྒན་གྱིས་ཁྱོད་ལ་རོགས་བྱས་ནས་རྒྱུས་ལོན་བྱེད་ངེས།དེ་མ་ཟད་ཁྱོད་ཀྱི་དྲི་བར་ལན་འདེབས་སྲིད།འདི་ནི་དགེ་རྒན་གྱིས་སྒྲུབ་འོས་པ་ཞིག་རེད།མཐར་ཕྱིན་རྗེས་བཤད་མི་དགོས།མྱུར་དུ་སོང་ནས་འདྲི།"
	}, {
		"human": "ངས་བལྟས་ན་རང་ཉིད་ཀྱི་དབྱིན་ཡིག་དང་དངོས་ལུགས་ཀྱི་རྒྱུགས་བླངས་པ་མི་བཟང་།གནས་ཚུལ་འདི་ཇི་ལྟར་ཐག་གཅོད་བྱེད་དགོས་སམ།",
		"assistant": "ཁྱོད་ཀྱིས་དབྱིན་ཡིག་མང་པོ་ཞིག་བློར་བཟུང་ན་ཆོག་པ་རེད།"
	}, {
		"human": "ཁྱོད་ཀྱིས་ང་ལ་རྒྱབ་སྐྱོར་དང་རྒྱབ་སྐྱོར་བྱས་པར་བཀའ་དྲིན་ཆེ།ཁྱོད་ཀྱི་བསམ་འཆར་དང་རྒྱབ་སྐྱོར་ཡོད་ན།ངས་སྔར་བས་སློབ་སྦྱོང་ལ་འབད་པ་དང་ཡིད་ཆེས་བརྟན་པོ་བྱས་ཏེ།མཐོ་རྒྱུགས་ལེགས་འགྲུབ་ཡོང་བར་བརྩོན་ལེན་བྱེད་སྲིད།",
		"assistant": "གནད་དོན་མེད།ངས་ནམ་ཡང་ཁྱོད་ལ་རྒྱབ་སྐྱོར་བྱེད།རང་ཉིད་ལ་ཡིད་ཆེས་བྱོས།རང་ཉིད་ཀྱི་མི་ཚེའི་དམིགས་འབེན་མངོན་འགྱུར་བྱེད་ཐུབ་ངེས་ཡིན།འབད་པ་བྱོས།"
	}]
}
```

## 六、模型训练
目前支持全量参数指令微调、QLoRA指令微调。我们将训练中使用的各种组件抽取出来，以便后续的扩展和优化，详见component目录下的实现。训练时的参数配置存储在train_args目录下，方便统一管理和更改。大家可以在train_args目录下查看不同模型的训练配置。

### 6.1 数据格式
训练时，我们将多轮对话拼接成如下格式，然后进行tokenize。其中<s\>表示bos_token，</s\> 表示eos_token。
```
<s>input1</s>target1</s>input2</s>target2</s>...
```
在计算loss时，我们通过mask的方式，input部分的loss不参与参数更新，只有“target</s>”部分的loss参与参数更新。
这种方式充分利用了模型并行计算的优势，训练更加高效，且多轮对话中的每个target部分都参与了训练，训练更充分。
否则，就需要把一个n轮对话，拆分成n条数据，且只计算最后一个target的loss，大大降低了训练效率。

### 6.2 全量参数微调
💻 执行如下命令即可进行全量参数微调：
```bash
deepspeed --num_gpus={num_gpus} train.py --train_args_file train_args/sft.json
```

📝 train_args/sft.json中的主要参数说明如下，以下参数可以根据需求进行修改，其他参数建议不做修改：
```sh
- output_dir：训练输出目录，存储checkpoint、tokenizer、tensorboard等
- model_name_or_path：预训练模型的本地目录，或者在huggingface上的模型名称。
- train_file：训练数据集路径。可以使用data/dummy_data.jsonl进行debug。
- num_train_epochs：训练的轮次。如果数据量足够大，一般建议只训一个epoch。
- per_device_train_batch_size：每张显卡的batch size。
- gradient_accumulation_steps：梯度累计步数。global batch=num_gpus * per_device_train_batch_size * gradient_accumulation_steps。
- gradient_checkpointing：如果显存捉襟见肘，可以开启。以时间换空间，模型不缓存激活状态，会进行两次forward计算，以节省显存。
- learning_rate：学习率。全量参数微调的时候，建议小一些，1e-5或5e-6。
- max_seq_length：训练时的最大长度。按照自己的设备进行设置，越长需要占用越多显存。
- logging_steps：每隔多少步统计一次train loss。
- save_steps：每隔多少步保存一个模型。
- save_total_limit：output_dir目录中最多保存多少个checkpoint，超出则会将最旧的删除。
- lr_scheduler_type：学习率变化策略。
- warmup_steps：warm up步数。学习率经过多少步，增长到指定的数值。
- optim：优化器。如果是全量参数微调，建议使用adamw_hf。
- seed：随机种子，用于复现实验结果。
- fp16：使用使用fp16混合精度。V100建议开启。
- bf16：使用使用bf16混合精度。A100建议开启。
```
### 6.3 QLoRA微调

QLoRA通过4-bit的nf4量化，且加入更多adapter，在大幅减少显存消耗的同时，尽可能逼近全量参数微调的效果。
QLoRA论文指出，该方法可以在一张V100上对33B的模型进行微调，并且性能逼近全量参数微调。

我们在bloom-7b1上使用qlora，adapter的参数量约1.2亿，超过bert-base模型参数量，可以在V100上使用1024的长度进行训练。

💻 执行如下命令即可进行QLoRA微调：
```bash
torchrun --nproc_per_node={num_gpus} train_qlora.py --train_args_file train_args/qlora/baichuan-7b-sft-qlora.json
```

📝 train_args/sft-qlora.json中的主要参数说明如下，基本与全量微调的参数一致，几个较为特殊：
- lora_rank：qlora矩阵的秩。一般设置为8、16、32、64等，在qlora论文中作者设为64。越大则参与训练的参数量越大，一般来说效果会更好，但需要更多显存，。
- lora_alpha: qlora中的缩放参数。一般设为16、32即可。
- lora_dropout: lora权重的dropout rate。
- learning_rate：qlora中的学习率设置更大一些，一般为1e-4、2e-4。


## 七、模型使用

### 权重合并
如果使用LoRA或者QLoRA进行训练，本项目仅保存adapter的权重和配置文件，需要将adapter权重与base model进行合并。脚本见script/merge_lora.py

### 7.1 模型推理
我们提供了单轮对话和多轮对话的脚本，详见script/chat目录，该脚本可同时兼容本项目训练的所有模型进行推理，不适用于非本项目训练的模型。
```bash
cd script/chat
python single_chat.py
```

生成脚本中的top_p、temperature、repetition_penalty、do_sample等参数对模型的生成效果影响较大，可按照自己的使用场景进行调试修改。

推理脚本中支持使用base model和adapter进行推理，缺点是每次启动脚本都需要合并一次权重，等待时间较久。

支持使用4bit进行推理，显存要求低，效果会略有下降。

### 7.2 服务部署
本项目支持将模型部署成HTTP服务，脚本在script/http下，使用flask进行开发。start_service.py为启动服务，post为发送请求，可按需进行修改。


基于以上模型的局限性，我们要求本项目的代码、数据、模型不得用于对社会造成危害的用途，且应当遵循基座模型的商业许可。


## 八、引用
若使用本项目的数据、代码或模型，请引用本项目。
```text
@misc{Tibetan_Mental_Health_Chat,
  author = {shajiu},
  title = {面向心理健康咨询的藏语数据集及大语言模型构建},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Shajiu/LLM/tree/main/Tibetan_Mental_Health_Chat}},
}
```
