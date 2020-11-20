import json
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForQuestionAnswering
from Abandon.question_context_match_BM25_fusion_score import dispose
from preliminary import read_json
from preliminary import get_word_embedding
from preliminary import sentences_partition
from preliminary import get_embedding_matrix_in_memory
from construct_inverted_index import construct_corpus_sentence


class Example(object):
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text.replace(" ", "").replace("  ", "").replace(" ", "")
        # self.answer_text = ""
        self.answer_text = answer_text
        # for e in answer_text.replace(" ", "").replace("  ", "").replace(" ", ""):
        #     self.answer_text += e
        #     self.answer_text += " "
        # self.answer_text = self.answer_text[0:-1]
        self.is_impossible = is_impossible


def read_examples():
    data_file = "./Data/dataset_simple.json"
    with open(data_file, encoding='utf-8') as file:
        examples = []
        is_impossible = False
        dataset = json.load(file)["data"]
        # print(len(load_dict))
        for index, data in enumerate(dataset):
            context = data["context"]
            question = data["question"]
            answer = data["answer"]
            example = Example(
                qas_id=index, question_text=question,
                context_text=context, answer_text=answer, is_impossible=is_impossible
            )
            examples.append(example)
    return examples


def evaluate():
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print('there are %d GPU(s) available.' % torch.cuda.device_count())
    #     print('we will use the GPU: ', torch.cuda.get_device_name(0))
    # else:
    #     print('No GPU availabel, using the CPU instead.')
    #     device = torch.device('cpu')
    device = torch.device('cuda')
    examples = read_examples()
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "E:\\Dataset\\pytorch-Bert\\bert-large-uncased-whole-word-masking-finetuned-squad")
    # model = AutoModelForQuestionAnswering.from_pretrained(
    #     "E:\\Dataset\\pytorch-Bert\\bert-large-uncased-whole-word-masking-finetuned-squad")

    # tokenizer = AutoTokenizer.from_pretrained(
    #     "E:\\Dataset\\pytorch-Bert\\bert-large-uncased-whole-word-masking-squad2")
    # model = AutoModelForQuestionAnswering.from_pretrained(
    #     "E:\\Dataset\\pytorch-Bert\\bert-large-uncased-whole-word-masking-squad2")

    # tokenizer = AutoTokenizer.from_pretrained("E:\\Dataset\\pytorch-Bert\\electra_large_discriminator_squad2_512")
    # model = AutoModelForQuestionAnswering.from_pretrained(
    #     "E:\\Dataset\\pytorch-Bert\\electra_large_discriminator_squad2_512")

    # model = BertForQuestionAnswering.from_pretrained('E:\\Dataset\\pytorch-Bert\\chinese-roberta-wwm-ext\\cmrc2018')
    # tokenizer = BertTokenizer.from_pretrained('E:\\Dataset\\pytorch-Bert\\chinese-roberta-wwm-ext\\cmrc2018')

    model = BertForQuestionAnswering.from_pretrained('./Model/chinese-roberta-wwm-ext-finetuned-cmrc2018')
    tokenizer = BertTokenizer.from_pretrained('./Model/chinese-roberta-wwm-ext-finetuned-cmrc2018')
    model.to(device)

    text = read_json()
    embeddings_index, emb_size = get_word_embedding()
    with open("./Data/GovernmentQA_IDF_values.json") as json_file:
        idf_values = json.load(json_file)

    all_sentences = sentences_partition(text)
    # text, all_sentences = read_txt()
    predictions = []
    bm25model, _ = construct_corpus_sentence()

    all_sentences_matrix_word2vector, all_sentences_tokens_nf, all_sentences_matrix_bert, \
        model_bert, tokenizer_bert = get_embedding_matrix_in_memory(
            all_sentences, embeddings_index, emb_size, idf_values
        )

    for index, example in enumerate(tqdm(examples)):
        # context = example.context_text
        context = dispose(
            text, example.question_text, all_sentences, all_sentences_matrix_bert,
            model_bert, tokenizer_bert, embeddings_index, emb_size, idf_values, bm25model, 20
        )

        # context = truncate_seq_pair(context, example.question_text, max_length=512)
        # print(f"上下文:{context}")
        # translated_text = baidu_translate(context, fromLang='zh', toLang='en')
        # print(f"Context:{translated_text}")
        #
        # translated_question = baidu_translate(example.question_text, fromLang='zh', toLang='en')
        # print(f"问题: {example.question_text}")
        # print(f"Question:{translated_question}")
        #
        # inputs = tokenizer(
        #     translated_question, translated_text, add_special_tokens=True, return_tensors="pt",
        #     truncation='longest_first', max_length=512
        # ).to(device)
        # input_ids = inputs["input_ids"].tolist()[0]
        # # text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        # answer_start_scores, answer_end_scores = model(**inputs)
        # answer_start = torch.argmax(
        #     answer_start_scores
        # )  # Get the most likely beginning of answer with the argmax of the score
        # # Get the most likely end of answer with the argmax of the score
        # answer_end = torch.argmax(answer_end_scores) + 1
        # answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        # print(f"Answer: {answer}")
        # translated_answer = baidu_translate(answer, fromLang='en', toLang='zh')
        # print(f"答案: {translated_answer}")
        # prediction = Example(
        #     qas_id=index,
        #     question_text=example.question_text,
        #     context_text=example.context_text,
        #     answer_text=translated_answer
        # )
        # predictions.append(prediction)

        # context = truncate_seq_pair(context, example.question_text, max_length=509)
        print(f"上下文:{context}")
        inputs = tokenizer(example.question_text, context, add_special_tokens=True, return_tensors="pt",
                           max_length=512, truncation=True).to(device)
        # input_ids = inputs["input_ids"].tolist()[0]
        input_ids = inputs["input_ids"][0]
        text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer_start_scores, answer_end_scores = model(**inputs)
        answer_start = torch.argmax(
            answer_start_scores
        )  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(
            answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
        answer = text_tokens[answer_start:answer_end]
        answer = "".join(answer)
        print(f"Question: {example.question_text}")
        print(f"Answer: {answer}")
        prediction = Example(
            qas_id=index,
            question_text=example.question_text,
            context_text=example.context_text,
            answer_text=answer
        )
        predictions.append(prediction)

    # list_predictions = []
    dict_prediction = {}
    # dict_predictions = {}
    # with open("./Metric/electra_large_discriminator_squad2_512.json", "w") as f:
    with open("./Metric/chinese-roberta-wwm-ext-finetuned-cmrc2018+BM25(test).json", "w") as f:
        for prediction in predictions:
            dict_prediction["context"] = prediction.context_text
            dict_prediction["question"] = prediction.question_text
            dict_prediction["answer"] = prediction.answer_text
            # list_predictions.append(dict_prediction)
            # for each_dict in list_predictions:
            f.write(json.dumps(dict_prediction, ensure_ascii=False) + '\n')
        print("加载入文件完成...")
    # json.dump(dict_predictions, f, ensure_ascii=False)

    # exact_avg_score, f1_avg_score = get_raw_scores(examples, predictions)
    # print(f"平均EM分数: {exact_avg_score}")
    # print(f"平均F1分数: {f1_avg_score}")


if __name__ == "__main__":
    evaluate()
