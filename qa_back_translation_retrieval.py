from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import googletrans
from googletrans import Translator
from baidu_translator import baidu_translate
from question_context_match import dispose
from preliminary import read_json
from preliminary import get_word_embedding
from preliminary import sentences_partition
from preliminary import truncate_seq_pair


tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")


# text = r"""
# Ten Opinions in support of "two strokes and two quotations". In order to implement the deployment of the
# Municipal Party Committee and the municipal government of "highlighting open investment promotion and
# strengthening project driving", improve the talent policy of "Huolan Plan", promote investment attraction and
# talent attraction effectiveness, and provide talent guarantee for the implementation of high-quality
# catch up in Putian, the following Suggestions are put forward in light of the actual situation of Putian.
# 1. For high-level innovation and entrepreneurship teams from home and abroad who come to Putian for innovation
# and entrepreneurship and transformation of achievements, they will be granted an additional RMB 1 million for
# scientific research on the basis of the "Huelan Plan" subsidy; For top talents such as Nobel Prize winners,
# academicians of the Chinese Academy of Sciences and the Chinese Academy of Sciences, winners of the
# State's Highest Science and Technology Award, or entrepreneurial teams with internationally advanced technology
# achievements industrialization projects in Putian, they will report to the Municipal party Committee and
# the municipal government for study and approval through "one case, one discussion" with a maximum grant
# of 50 million yuan. Responsible units: Municipal Organization Department, Municipal Science and Technology Bureau.
# 2. The year of the actual investment of 200 million yuan or 50 million yuan of the total amount of the tax organs
# (including the existing enterprises to increase investment, mergers and acquisitions to increase investment),
# the world's top 500 enterprises of foreign direct investment projects (investment do not set limit), total investment
# (to use its own funds, the same below) in place of 50 million yuan, to give independent index 1 identified four
# types of talents, increase the investment in 50 million yuan to increase an indicator, the same current is
# no more than 10 enterprises. Responsible units: Municipal Development and Reform Commission,
# Municipal Bureau of Finance, Municipal Bureau of Industry and Information Technology, municipal Bureau of Commerce.
# """
#
# questions = [
#     "In what city is the Huolan Project?",
#     "How much do Nobel Prize winners get?",
#     "What is the maximum number of talents for the same company in the current year?"
# ]
#
# for question in questions:
#     inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
#     input_ids = inputs["input_ids"].tolist()[0]
#     text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
#     answer_start_scores, answer_end_scores = model(**inputs)
#     answer_start = torch.argmax(
#         answer_start_scores
#     )  # Get the most likely beginning of answer with the argmax of the score
#     answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
#     answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
#     print(f"Question: {question}")
#     print(f"Answer: {answer}")


# en zh-cn
print(googletrans.LANGUAGES)

text = read_json()


# translator = Translator(['translate.google.cn'])
# translated_text = translator.translate(text, src='zh-cn', dest='en')
# print(translated_text.text)


questions = [
    "壶兰计划是哪个市的？",
    "“壶兰计划”中诺贝尔奖获得者可以获得多少补助？",
    "“壶兰计划”中同一企业当年度最多不超过多少名人才指标？",
    "“壶兰计划”中的“双招双引”给出哪些意见？",
    "莆田市人才政策的全称是什么？",
    "特级人才需要符合哪些条件？",
    "莆田高层次人才服务窗口在哪？"
]

embeddings_index, emb_size = get_word_embedding()
all_sentences = sentences_partition(text)


def qa():
    for question in questions:
        # translated_question = translator.translate(question, src='zh-cn', dest='en')
        # print(translated_question.text)
        # inputs = tokenizer(translated_question.text, translated_text.text,
        #                    add_special_tokens=True, return_tensors="pt")
        context = dispose(text, question, all_sentences, embeddings_index, emb_size)
        context = truncate_seq_pair(context, question, max_length=509)
        print(f"上下文:{context}")
        translated_text = baidu_translate(context, fromLang='zh', toLang='en')
        # 防止翻译后的上下文+问题依然超出max_length
        # translated_text = truncate_seq_pair(translated_text, question, max_length=509)
        print(f"Context:{translated_text}")

        translated_question = baidu_translate(question, fromLang='zh', toLang='en')
        print(f"问题: {question}")
        print(f"Question:{translated_question}")
        inputs = tokenizer(translated_question, translated_text, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]
        # text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer_start_scores, answer_end_scores = model(**inputs)
        answer_start = torch.argmax(
            answer_start_scores
        )  # Get the most likely beginning of answer with the argmax of the score
        # Get the most likely end of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        print(f"Answer: {answer}")
        # translated_answer = translator.translate(answer, src='en', dest='zh-cn')
        # print(f"FinalAnswer: {translated_answer.text}")
        translated_answer = baidu_translate(answer, fromLang='en', toLang='zh')
        print(f"答案: {translated_answer}")


if __name__ == "__main__":
    qa()
