import json
import math
import jieba
import numpy as np
import os
import torch
from transformers import BertTokenizer
from transformers import BertModel


def read_json():
    """
    读取json文件dataset.jsonl
    :return: 原始预处理里的政务数据
    """
    with open("./Data/dataset.jsonl", 'r', encoding='utf-8') as file:
        context = []
        for line in file.readlines():
            load_dict = json.loads(line)
            context.append(load_dict["title"] + "。" + load_dict["article"])
        # for con in context:
        #     print(len(con.split('。')))
        return context


def find_all_file(base, end=".pdf"):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith(end):
                fullname = os.path.join(root, f)
                yield fullname


def read_txt():
    base = 'E:\\Putian\\Dataset\\txt_documents\\'
    # 每个文档划分成句子（一行为一句）后组成的上下文
    all_sentences = []
    # 所有文档组成的上下文
    content = []
    for file_path in find_all_file(base, end='.txt'):
        with open(file_path, "r") as f:
            context = f.read()
            content.append(context)
            all_sentences.append(context.split('\n')[:-1])
    return content, all_sentences


def sentences_partition(context):
    """
    使用"。"将文章划分成句子
    :param context:由所有文章组成的列表，每个项都是一个字符串
    :return:所有篇章中的句子（嵌套列表的形式）
    """
    all_sentences = []
    for con in context:
        sentence_per_article = con.split('。')

        # print(len(sentence_per_article))
        all_sentences.append(sentence_per_article)
    return all_sentences


def sentence_segmentation(sentence, flag=0):
    """
    对句子进行分词，得到句子所有token
    :param sentence:待分词的原句子
    :param flag:是否排除停顿词，1排除，0不排除
    :return:停顿词过滤并且句子分词后的token列表
    """
    # 读取停用词
    stopwords_file = "./Data/hit_stopwords.txt"
    stop_f = open(stopwords_file, "r", encoding='utf-8')
    stop_words = list()
    for line in stop_f.readlines():
        line = line.strip()
        if not len(line):
            continue
        stop_words.append(line)
    stop_f.close()
    # print(len(stop_words))

    sentence = sentence.strip()
    if not len(sentence):
        return
    # out_str = ''
    result = []
    seg_list = jieba.cut(sentence, cut_all=False)
    for word in seg_list:
        if flag == 1:
            if word not in stop_words:
                if word != '\t':
                    # out_str += word
                    # out_str += " "
                    # seg_list = " ".join(seg_list)
                    result.append(word)
        elif flag == 0:
            if word != '\t':
                result.append(word)
    # result.append(out_str.strip())
    return result


def get_idf(question, context):
    """
    获取问题中每个token对应的idf值
    公式见论文"Alignment over Heterogeneous Embeddings for Question Answering": pp.2683, 式1
    :param question:当前问题
    :param context:原始上下文数据
    :return:问题中的token在所有篇章中的idf值
    """
    question_idf = {}
    article_nums = len(context)
    for question_token in question:
        # 包含每个问题中的token的文章总数
        doc_freq = 0
        for con in context:
            if con.find(question_token) > -1:
                doc_freq += 1
        # 每个问题token的idf值
        # idf = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5))
        # 由于这里的N比较小，为了防止idf是负值，因此只保留N，不减去doc_freq
        idf = math.log((article_nums + 0.5) / (doc_freq + 0.5))
        question_idf[question_token] = idf
    return question_idf


def get_word_embedding():
    """
    获得词嵌入
    :return:嵌入表示；
            嵌入表示维度大小，默认300维
    """
    model_path = "E:\\Dataset\\embedding-chinese\\sgns.baidubaike.bigram-char"
    f = open(model_path, 'r', encoding='utf-8')
    embeddings_index = {}
    index = 0
    emb_size = 0
    for line in f:
        if index == 0:
            index += 1
            continue
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            # 求二范数对向量归一化
            b = np.linalg.norm(coefs, ord=2)
            coefs = coefs / float(b)
            emb_size = coefs.shape[0]
        except ValueError:
            print(values[0])
            continue
        embeddings_index[word] = coefs
    print("Word2vc matrix len is : ", len(embeddings_index))
    print("Embedding size is: ", emb_size)
    # 归一化后的词嵌入
    return embeddings_index, emb_size


def get_sentence_embedding(sentence, embeddings_index, emb_size, idf_values, ques_text=0, object_subject_list=[]):
    """
    获得句子嵌入
    :param sentence:目标句子
    :param embeddings_index:嵌入表示
    :param emb_size:嵌入表示维度大小，默认300维
    :param idf_values:所有篇章中term的idf值
    :param ques_text:该句子是否是问题，是则是1，否则为0
    :param object_subject_list:特定token词列表
    :return:若是问题，返回句子的嵌入矩阵、问题中的token在所有篇章中的idf值变换后的idf矩阵、
            问题中未在embedding中找到的token、问题中在embedding找到的token；
            否则为证据句子的嵌入表示矩阵、证据句子未在embedding中找到的token、证据句子在embedding中找到的token
    """
    sentence_matrix = np.empty((0, emb_size), float)
    idf_mat = []
    tokens_not_found_embeddings = []
    tokens_embeddings_found = []
    for term in sentence:
        if term in embeddings_index:
            sentence_matrix = np.append(sentence_matrix, np.array([np.asarray(embeddings_index[term])]), axis=0)
            tokens_embeddings_found.append(term)
        else:
            tokens_not_found_embeddings.append(term)

        if ques_text == 1:
            if term in object_subject_list:
                important_term_coefficient = 1
            else:
                important_term_coefficient = 1
            # 这里是针对多项选择类型，即查询由问题和候选答案组成
            if term in idf_values:
                idf_mat.append(important_term_coefficient * idf_values[term])
            # 对于候选答案，将其term的idf值变为3
            else:
                idf_mat.append(important_term_coefficient * 3)
                # print ("the unknown IDF term is: ", q_term)
    # 如果该句子是问题，则多返回一个idf值矩阵
    if ques_text == 1:
        return sentence_matrix, idf_mat, tokens_not_found_embeddings, tokens_embeddings_found
    else:
        return sentence_matrix, tokens_not_found_embeddings, tokens_embeddings_found


def truncate_seq_pair(context, question, max_length=512):
    """
    Truncates a sequence pair in place to the maximum length.
    :param context:原始上下文数据
    :param question:当前问题
    :param max_length:模型最大输入序列长度
    :return:截断后的上下文（针对中文）
    """
    while True:
        total_length = len(context) + len(question)
        if total_length <= max_length:
            break
        if len(context) > len(question):
            context = context[:-1]
        else:
            question = question[:-1]
    return context
