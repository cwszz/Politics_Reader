# coding=gbk
# 中文模型
from transformers import BertTokenizer
from transformers import BertForQuestionAnswering
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('there are %d GPU(s) available.'% torch.cuda.device_count())
    print('we will use the GPU: ', torch.cuda.get_device_name(0))
else:
    print('No GPU availabel, using the CPU instead.')
    device = torch.device('cpu')

tokenizer = BertTokenizer.from_pretrained('../Model/chinese-roberta-wwm-ext-finetuned-cmrc2018')
model = BertForQuestionAnswering.from_pretrained('../Model/chinese-roberta-wwm-ext-finetuned-cmrc2018')
# model.load_state_dict(torch.load('../Model/pytorch_model.bin'))
model.to(device)

context = "株洲北站全称广州铁路（集团）公司株洲北火车站。除站场主体，另外管辖湘潭站、湘潭东站和三个卫星站，田心站、白马垅站、十里冲站，以及原株洲车站货房。车站办理编组、客运、货运业务。车站机关地址：湖南省株洲市石峰区北站路236号，邮编412001。株洲北站位于湖南省株洲市区东北部，地处中南路网，是京广铁路、沪昆铁路两大铁路干线的交汇处，属双向纵列式三级七场路网性编组站。车站等级为特等站，按技术作业性质为编组站，按业务性质为客货运站，是株洲铁路枢纽的主要组成部分，主要办理京广、沪昆两大干线四个方向货物列车的到发、解编作业以及各方向旅客列车的通过作业。每天办理大量的中转车流作业，并有大量的本地车流产生和集散，在路网车流的组织中占有十分重要的地位，是沟通华东、华南、西南和北方的交通要道，任务艰巨，作业繁忙。此外，株洲北站还有连接石峰区喻家坪工业站的专用线。株洲北站的前身是田心车站。"
question = "株洲北站的机关地址是什么"  # 株洲北站的前身是什么

inputs = tokenizer(question, context, return_tensors="pt", max_length=512).to(device)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
start_scores, end_scores = model(**inputs)
answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores) + 1
answer =tokens[answer_start:answer_end]
str=""
print(str.join(answer))
print("标准答案：湖南省株洲市石峰区北站路236号，邮编412001。" )

