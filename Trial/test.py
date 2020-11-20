# coding=gbk
# ����ģ��
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

context = "���ޱ�վȫ�ƹ�����·�����ţ���˾���ޱ���վ����վ�����壬�����Ͻ��̶վ����̶��վ����������վ������վ��������վ��ʮ���վ���Լ�ԭ���޳�վ��������վ������顢���ˡ�����ҵ�񡣳�վ���ص�ַ������ʡ������ʯ������վ·236�ţ��ʱ�412001�����ޱ�վλ�ں���ʡ�����������������ش�����·�����Ǿ�����·��������·������·���ߵĽ��㴦����˫������ʽ�����߳�·���Ա���վ����վ�ȼ�Ϊ�ص�վ����������ҵ����Ϊ����վ����ҵ������Ϊ�ͻ���վ����������·��Ŧ����Ҫ��ɲ��֣���Ҫ�����㡢������������ĸ���������г��ĵ����������ҵ�Լ��������ÿ��г���ͨ����ҵ��ÿ������������ת������ҵ�����д����ı��س��������ͼ�ɢ����·����������֯��ռ��ʮ����Ҫ�ĵ�λ���ǹ�ͨ���������ϡ����Ϻͱ����Ľ�ͨҪ���������ޣ���ҵ��æ�����⣬���ޱ�վ��������ʯ��������ƺ��ҵվ��ר���ߡ����ޱ�վ��ǰ�������ĳ�վ��"
question = "���ޱ�վ�Ļ��ص�ַ��ʲô"  # ���ޱ�վ��ǰ����ʲô

inputs = tokenizer(question, context, return_tensors="pt", max_length=512).to(device)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
start_scores, end_scores = model(**inputs)
answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores) + 1
answer =tokens[answer_start:answer_end]
str=""
print(str.join(answer))
print("��׼�𰸣�����ʡ������ʯ������վ·236�ţ��ʱ�412001��" )

