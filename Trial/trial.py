# coding=utf-8
# 英翻中
# from transformers import pipeline
#
# if __name__ == '__main__':
#     question_answerer = pipeline('question-answering')
#     result = question_answerer({
#         'question': 'What is the name of the repository ?',
#         'context': 'Pipeline have been included in the huggingface/transformers repository'
#     })
#     print(result)

# from transformers import pipeline
#
# if __name__ == '__main__':
#     nlp = pipeline("question-answering")
#     context = r"""Extractive Question Answering is the task of extracting an answer from a text given a question.
#     An example of a question answering dataset is the SQuAD dataset, which is entirely based on that task.
#     If you would like to fine-tune a model on a SQuAD task,
#     you may leverage the examples/question-answering/run_squad.py script.
#     """
#     result = nlp(question="What is extractive question answering?", context=context)
#     print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)},
#     start: {result['start']}, end: {result['end']}")
#     result = nlp(question="What is a good example of a question answering dataset?", context=context)
#     print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)},
#     start: {result['start']}, end: {result['end']}")

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import RobertaTokenizer, RobertaForQuestionAnswering
import torch
# 效果最好
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
# 效果最差
# tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
# model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
# 效果次之
# tokenizer = AutoTokenizer.from_pretrained("E:\\Dataset\\pytorch-Bert\\bert-large-uncased-whole-word-masking-squad2")
# model = AutoModelForQuestionAnswering.from_pretrained("E:\\Dataset\\pytorch-Bert\\bert-large-uncased-whole-word-masking-squad2")
# text = r"""
# 🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
# architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
# Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
# TensorFlow 2.0 and PyTorch.
# """
#
# questions = [
#     "How many pretrained models are available in 🤗 Transformers?",
#     "What does 🤗 Transformers provide?",
#     " 🤗Transformers provides interoperability between which frameworks?",
# ]

# text = r"""
# Coulson is an agent of S.H.I.E.L.D., and later becomes the organization's director.[23]
# In April 2013, Gregg agreed to join the series after hearing creator Joss Whedon's explanation for
# Coulson's resurrection, following the character's death in The Avengers, which he called "fascinating"
# and "true to the world of the comics". After being possessed by the Spirit of Vengeance in the fourth season finale,
# the Kree blood that resurrected Coulson is burned away and he ultimately dies following the fifth-season
# finale.
# """
#
# questions = [
#     "Who is Coulson?",
#     "Who is the organization's director?",
#     "Which season Coulson died?",
# ]

# text = r"""
# You Hao comes from Fuyang, Anhui Province of China, who is pursuing his Doctor degree
# in University of Chinese Academy of Science in Beijing. Someday,
# he brought some milk in a near supermarket, after he got home, he put it on the desk of the living room,
# then he went out for dinner with his friends.
# """
#
# questions = [
#     "Is You Hao a student?",
#     "Where does You Hao come from?",
#     "Where is the milk?",
#     "Where does You Hao study?",
#     "Which city does You Hao study?",
#     "Who comes from Anhui?"
# ]

text = r"""
Ten Opinions in support of "two strokes and two quotations". In order to implement the deployment of the Municipal Party Committee
and the municipal government of "highlighting open investment promotion and strengthening project driving",
improve the talent policy of "Huolan Plan", promote investment attraction and talent attraction effectiveness,
and provide talent guarantee for the implementation of high-quality catch up in Putian, the following Suggestions
are put forward in light of the actual situation of Putian. 1. For high-level innovation and entrepreneurship teams
from home and abroad who come to Putian for innovation and entrepreneurship and transformation of achievements,
they will be granted an additional RMB 1 million for scientific research on the basis of the "Huelan Plan" subsidy;
For top talents such as Nobel Prize winners, academicians of the Chinese Academy of Sciences and the Chinese Academy of Sciences,
winners of the State's Highest Science and Technology Award, or entrepreneurial teams with internationally advanced technology
achievements industrialization projects in Putian, they will report to the Municipal party Committee and the municipal government
for study and approval through "one case, one discussion" with a maximum grant of 50 million yuan. Responsible units:
Municipal Organization Department, Municipal Science and Technology Bureau. 2. The year of the actual investment of
200 million yuan or 50 million yuan of the total amount of the tax organs (including the existing enterprises to
increase investment, mergers and acquisitions to increase investment), the world's top 500 enterprises of foreign
direct investment projects (investment do not set limit), total investment (to use its own funds, the same below)
in place of 50 million yuan, to give independent index 1 identified four types of talents,
increase the investment in 50 million yuan to increase an indicator, the same current is no more than 10 enterprises.
Responsible units: Municipal Development and Reform Commission, Municipal Bureau of Finance, Municipal Bureau of Industry
and Information Technology, municipal Bureau of Commerce.
"""

questions = [
    "In what city is the Huolan Project?",
    "How much do Nobel Prize winners get?",
    "What is the maximum number of talents for the same company in the current year?"
]


for question in questions:
    # inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs)
    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    print(f"Question: {question}")
    print(f"Answer: {answer}")
