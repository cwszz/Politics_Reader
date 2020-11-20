from transformers import AutoModelForQuestionAnswering, BertTokenizer, BertForQuestionAnswering
import torch
from preliminary import read_json
from preliminary import get_word_embedding
from preliminary import sentences_partition
from preliminary import truncate_seq_pair
from question_context_match import dispose

# model = BertForQuestionAnswering.from_pretrained('E:\\Dataset\\pytorch-Bert\\bert-base-chinese')
# tokenizer = BertTokenizer.from_pretrained('E:\\Dataset\\pytorch-Bert\\bert-base-chinese')
# model = BertForQuestionAnswering.from_pretrained('E:\\Dataset\\pytorch-Bert\\albert-chinese-large-qa')
# tokenizer = BertTokenizer.from_pretrained('E:\\Dataset\\pytorch-Bert\\albert-chinese-large-qa')
# model = BertForQuestionAnswering.from_pretrained('E:\\Dataset\\pytorch-Bert\\chinese-roberta-wwm-ext\\cmrc2018')
# tokenizer = BertTokenizer.from_pretrained('E:\\Dataset\\pytorch-Bert\\chinese-roberta-wwm-ext\\cmrc2018')
model = BertForQuestionAnswering.from_pretrained('./Model/chinese-roberta-wwm-ext-finetuned-cmrc2018')
tokenizer = BertTokenizer.from_pretrained('./Model/chinese-roberta-wwm-ext-finetuned-cmrc2018')


text = read_json()
questions = [
    "壶兰计划是哪个市的？",
    "“壶兰计划”中诺贝尔奖获得者可以获得多少补助？",
    "“壶兰计划”中同一企业当年度最多不超过多少名人才指标？",
    "“壶兰计划”中的“双招双引”给出哪些意见？",
    "莆田市人才政策的全称是什么？"
]
embeddings_index, emb_size = get_word_embedding()
all_sentences = sentences_partition(text)

for question in questions:
    # inputs = tokenizer(question, text, add_special_tokens=False, return_tensors="pt")
    context = dispose(text, question, all_sentences, embeddings_index, emb_size)
    # context = truncate_seq_pair(context, question, max_length=509)
    print(f"上下文:{context}")
    inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt",
                       truncation=True, max_length=512)
    # input_ids = inputs["input_ids"].tolist()[0]
    input_ids = inputs["input_ids"][0]
    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs)
    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    # answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
    # answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    answer_end = torch.argmax(answer_end_scores)+1  # Get the most likely end of answer with the argmax of the score
    answer = text_tokens[answer_start:answer_end]
    answer = "".join(answer)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
