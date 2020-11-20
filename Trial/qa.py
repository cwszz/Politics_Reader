# coding=gbk
# ����ģ��
from transformers import AutoModelForQuestionAnswering, BertTokenizer, BertForQuestionAnswering
import torch
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print('there are %d GPU(s) available.'% torch.cuda.device_count())
#     print('we will use the GPU: ', torch.cuda.get_device_name(0))
# else:
#     print('No GPU availabel, using the CPU instead.')
#     device = torch.device('cpu')
device = torch.device('cpu')
# model = BertForQuestionAnswering.from_pretrained('E:\\Dataset\\pytorch-Bert\\bert-base-chinese')
# tokenizer = BertTokenizer.from_pretrained('E:\\Dataset\\pytorch-Bert\\bert-base-chinese')
model = BertForQuestionAnswering.from_pretrained('../Model/chinese-roberta-wwm-ext-finetuned-cmrc2018')
tokenizer = BertTokenizer.from_pretrained('../Model/chinese-roberta-wwm-ext-finetuned-cmrc2018')

# model = BertForQuestionAnswering.from_pretrained('E:\\Dataset\\pytorch-Bert\\albert-chinese-large-qa')
# tokenizer = BertTokenizer.from_pretrained('E:\\Dataset\\pytorch-Bert\\albert-chinese-large-qa')
# model = BertForQuestionAnswering.from_pretrained('E:\\Dataset\\pytorch-Bert\\chinese-roberta-wwm-ext\\finetune')
# tokenizer = BertTokenizer.from_pretrained('E:\\Dataset\\pytorch-Bert\\chinese-roberta-wwm-ext\\finetune')
model.to(device)


def read_text(text):
    text_name = "../Data/data_512.txt"
    file = open(text_name, mode='r', encoding="gbk")
    for line in file:
        text += line.strip('\n')
    file.close()
    print(text)
    return text


text = ""
text = read_text(text)
# text = "����֧�֡�˫��˫������ʮ�������Ϊ��ʵ��ί����������ͻ���������� ǿ����Ŀ�������Ĳ������ơ������ƻ����˲����ߣ��ٽ��������ʺ��в�����ʵЧ,Ϊ������ʵ�������ϳ��ṩ�˲ű��ϣ��������ʵ�ʣ���������������1.�������ﴴ�´�ҵ�ͳɹ�ת���Ĺ�����߲�δ��´�ҵ�Ŷӣ����϶����ڡ������ƻ��������Ļ����ϣ��ٸ��� 100 ��Ԫ���ڿ�ѧ�о�����ŵ����������ߡ�����Ժ��Ժʿ��������߿�ѧ����������ߵȶ����˲Ż�Я�����й����Ƚ�ˮƽ�����ɹ���ҵ����Ŀ�����仧�Ĵ�ҵ�Ŷӣ�ͨ����һ��һ�顱��ʽ������ί���������о�ͬ�⣬��߸��� 5000 ��Ԫ���������ε�λ����ί��֯�����пƼ��֡�2.��ʵ��Ͷ�� 2 ��Ԫ������˰�ܶ� 5000 ��Ԫ���ϵ�������Ŀ(����������ҵ����Ͷ�ʡ���ҵ��������Ͷ��)������ 500 ǿ��ҵ������ֱ��Ͷ����Ŀ(Ͷ�ʶ������)�������Ͷ�ʣ�ʹ�������ʽ���ͬ����λ 5000 ��Ԫ�����������϶������˲�ָ�� 1����ÿ����ʵ�ʵ�λͶ�ʶ� 5000 ��Ԫ���� 1 ��ָ�꣬ͬһ��ҵ�������಻���� 10 �������ε�λ���з���ί���в����֡��й��ž֡�������֡�"

questions = [
    "�����ƻ����ĸ��еģ�",
    "ŵ����������߿��Ի�ö��ٲ�����",
    "ͬһ��ҵ�������಻�����������˲�ָ�ꣿ",
    "��˫��˫����������Щ�����",
    "�ڡ������ƻ��������Ļ����ϣ��ٸ��������Ԫ���ڿ�ѧ�о���"
]

# text = "ө��湤������һ���ܲ�����Ӣ���׶غͿ����Ҹ��ݿ��٣������ո��������������������ŵĵ�����Ϸ�����̡�" \
#        "1999��8�£����ɡ����׵²������ˡ������غʹ�������˹�س���ө��湤���ң�һ�𿪷��˺ܶ���Ϸ��" \
#        "�����ǳ��ɹ��ġ������� �͡�����������ϵ�С���˾������ө��湤���ҷ�����һ��δ��ǰ���滮��" \
#        "��ө��湤����Ҫ����һ��������Ϸ���е�������Ŀ�������硣����Ҫ�ṩһ���ḻ��ʵ���Ϸ������" \
#        "����������ǵ�ͼ��ͱ��뼼��������������Ϸ�����ие���졣���ǵ�ר��������Ϸ�п���ս�ԣ�" \
#        "�����ǽ��Ҫ������չ�������Ǿ��ʵ��Ӿ�Ч����������Ŀ��������������ֵ��ص����ϡ�" \
#        "���������������ɹ�������ҽ��ᷢ��һ���Լ�����ģ��ӽ��Լ����Ե����硱��" \
#        "�ù�˾���г���λ��PC(Windows)��ƻ�������ϵļ�ʱս����Ϸ�����ر��ǹ�˾�ɹ��ġ�Ҫ����ϵ�С�" \
#        "Ŀǰ���������ڿ���PC��Xbox360�ϵĴ�ʱ����Ϸ��"
# questions = [
#     "ө��湤���ҵ��ܲ��������",
#     "�ù�˾���г���λ��ʲô��",
#     "˭������ө��湫˾��"
#     ]


# text = "����ܾ��ع�����Ӣ�Agents of S.H.I.E.L.D.����һ�������㲥��˾��2013�������������ĵ��Ӿ磬" \
#        "����˹���µǸ������������е�ͬ����֯��ܾ�Ϊ��������������������Ӱ�����ϵ����Ʒ֮һ��" \
#        "�ӵ�Ӱϵ�еĵڶ��׶ο�ʼ������ܾ֡���S.H.I.E.L.D.���ǡ�����ս�Է�����������ڱ��Ͼ֡�" \
#        "��Strategic Homeland Intervention��Enforcement and Logistics Division���ļ�ƣ�" \
#        "ǰ��Ϊ��ս�Կ�ѧ���š���Strategic Scientific Reserve, S.S.R.�����ڵڶ��������ս�ڼ䣬" \
#        "Ϊ�˶Կ�����ͷ�ߡ���HYDRA������ͬ�˹�������ɡ�"
#
# questions = [
#     "����ܾ��ع������ĸ���˾�����ģ�",
#     "����ܾ֡���ʲô�ļ�ƣ�",
#     "����ܾ֡���Ϊ�˶Կ��ĸ���֯����ͬ�˹�������ɵģ�",
#     "��ܾ���ʲôʱ������ģ�"
# ]

# text = "���ޱ�վȫ�ƹ�����·�����ţ���˾���ޱ���վ����վ�����壬�����Ͻ��̶վ����̶��վ����������վ������վ��������վ��ʮ���վ���Լ�ԭ���޳�վ��������վ������顢���ˡ�����ҵ�񡣳�վ���ص�ַ������ʡ������ʯ������վ·236�ţ��ʱ�412001�����ޱ�վλ�ں���ʡ�����������������ش�����·�����Ǿ�����·��������·������·���ߵĽ��㴦����˫������ʽ�����߳�·���Ա���վ����վ�ȼ�Ϊ�ص�վ����������ҵ����Ϊ����վ����ҵ������Ϊ�ͻ���վ����������·��Ŧ����Ҫ��ɲ��֣���Ҫ�����㡢������������ĸ���������г��ĵ����������ҵ�Լ��������ÿ��г���ͨ����ҵ��ÿ������������ת������ҵ�����д����ı��س��������ͼ�ɢ����·����������֯��ռ��ʮ����Ҫ�ĵ�λ���ǹ�ͨ���������ϡ����Ϻͱ����Ľ�ͨҪ���������ޣ���ҵ��æ�����⣬���ޱ�վ��������ʯ��������ƺ��ҵվ��ר���ߡ����ޱ�վ��ǰ�������ĳ�վ��"
# questions = [
#     "���ޱ�վ�Ļ��ص�ַ��ʲô",
#     "���ޱ�վ��Ͻ��Щվ��"
#     ]

for question in questions:
    # inputs = tokenizer(question, text, add_special_tokens=False, return_tensors="pt")
    inputs = tokenizer(question, text, add_special_tokens=False, return_tensors="pt")
    # input_ids = inputs["input_ids"].tolist()[0]
    input_ids = inputs["input_ids"][0]
    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs)
    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    # answer_end = torch.argmax(answer_end_scores)+1  # Get the most likely end of answer with the argmax of the score
    # answer = text_tokens[answer_start:answer_end]
    # answer = "".join(answer)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    # inputs = tokenizer(question, text, return_tensors="pt").to(device)
    # tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    # start_scores, end_scores = model(**inputs)
    # answer_start = torch.argmax(start_scores)
    # answer_end = torch.argmax(end_scores)+1
    # answer = tokens[answer_start:answer_end]
    # str = ""
    # print(str.join(answer))
    # print("��׼�𰸣�����ʡ������ʯ������վ·236�ţ��ʱ�412001��")
