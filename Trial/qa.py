# coding=gbk
# 中文模型
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
# text = "关于支持“双招双引”的十条意见。为落实市委、市政府“突出开放招商 强化项目带动”的部署，完善“壶兰计划”人才政策，促进招商引资和招才引智实效,为莆田落实高质量赶超提供人才保障，结合莆田实际，现提出以下意见。1.对来莆田创新创业和成果转化的国内外高层次创新创业团队，经认定，在“壶兰计划”补助的基础上，再给予 100 万元用于科学研究；对诺贝尔奖获得者、“两院”院士、国家最高科学技术奖获得者等顶尖人才或携带具有国际先进水平技术成果产业化项目来莆落户的创业团队，通过“一事一议”方式报请市委、市政府研究同意，最高给予 5000 万元资助。责任单位：市委组织部、市科技局。2.对实际投资 2 亿元或年纳税总额 5000 万元以上的招商项目(包括现有企业增加投资、企业并购增加投资)，世界 500 强企业的外商直接投资项目(投资额不设下限)，当年度投资（使用自有资金，下同）到位 5000 万元，给予自主认定四类人才指标 1名，每增加实际到位投资额 5000 万元增加 1 名指标，同一企业当年度最多不超过 10 名。责任单位：市发改委、市财政局、市工信局、市商务局。"

questions = [
    "壶兰计划是哪个市的？",
    "诺贝尔奖获得者可以获得多少补助？",
    "同一企业当年度最多不超过多少名人才指标？",
    "“双招双引”给出哪些意见？",
    "在“壶兰计划”补助的基础上，再给予多少万元用于科学研究？"
]

# text = "萤火虫工作室是一家总部设在英国伦敦和康涅狄格州坎顿，并在苏格兰阿伯丁设有质量部门的电子游戏开发商。" \
#        "1999年8月，西蒙·布雷德伯里，埃里克·乌列特和大卫·莱斯特成立萤火虫工作室，一起开发了很多游戏，" \
#        "包括非常成功的“凯撒” 和“王国霸主”系列。公司成立后，萤火虫工作室发布了一个未来前景规划：" \
#        "“萤火虫工作室要创造一个人们游戏其中的引人瞩目的新世界。我们要提供一个丰富多彩的游戏环境，" \
#        "令玩家在我们的图像和编码技术不断提升的游戏世界中感到愉快。我们的专长是在游戏中开发战略，" \
#        "而我们今后要继续发展，与我们精彩的视觉效果，引人瞩目的人物和易于上手的特点相结合。" \
#        "如果我们能这样完成工作，玩家将会发现一个自己创造的，加进自己个性的世界”。" \
#        "该公司将市场定位于PC(Windows)和苹果电脑上的即时战略游戏领域，特别是公司成功的“要塞”系列。" \
#        "目前，他们正在开发PC和Xbox360上的次时代游戏。"
# questions = [
#     "萤火虫工作室的总部设在哪里？",
#     "该公司将市场定位于什么？",
#     "谁成立了萤火虫公司？"
#     ]


# text = "《神盾局特工》（英语：Agents of S.H.I.E.L.D.）是一部美国广播公司于2013年制作并播出的电视剧，" \
#        "由乔斯·温登根据漫威漫画中的同名组织神盾局为蓝本创作，属于漫威电影宇宙的系列作品之一，" \
#        "从电影系列的第二阶段开始。“神盾局”（S.H.I.E.L.D.）是“国土战略防御攻击与后勤保障局”" \
#        "（Strategic Homeland Intervention，Enforcement and Logistics Division）的简称，" \
#        "前身为“战略科学军团”（Strategic Scientific Reserve, S.S.R.），在第二次世界大战期间，" \
#        "为了对抗“九头蛇”（HYDRA）而由同盟国联合组成。"
#
# questions = [
#     "《神盾局特工》是哪个公司制作的？",
#     "“神盾局”是什么的简称？",
#     "“神盾局”是为了对抗哪个组织而由同盟国联合组成的？",
#     "神盾局是什么时候成立的？"
# ]

# text = "株洲北站全称广州铁路（集团）公司株洲北火车站。除站场主体，另外管辖湘潭站、湘潭东站和三个卫星站，田心站、白马垅站、十里冲站，以及原株洲车站货房。车站办理编组、客运、货运业务。车站机关地址：湖南省株洲市石峰区北站路236号，邮编412001。株洲北站位于湖南省株洲市区东北部，地处中南路网，是京广铁路、沪昆铁路两大铁路干线的交汇处，属双向纵列式三级七场路网性编组站。车站等级为特等站，按技术作业性质为编组站，按业务性质为客货运站，是株洲铁路枢纽的主要组成部分，主要办理京广、沪昆两大干线四个方向货物列车的到发、解编作业以及各方向旅客列车的通过作业。每天办理大量的中转车流作业，并有大量的本地车流产生和集散，在路网车流的组织中占有十分重要的地位，是沟通华东、华南、西南和北方的交通要道，任务艰巨，作业繁忙。此外，株洲北站还有连接石峰区喻家坪工业站的专用线。株洲北站的前身是田心车站。"
# questions = [
#     "株洲北站的机关地址是什么",
#     "株洲北站管辖哪些站？"
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
    # print("标准答案：湖南省株洲市石峰区北站路236号，邮编412001。")
