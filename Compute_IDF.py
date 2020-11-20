import math
import json
from preliminary import sentences_partition
from preliminary import sentence_segmentation
from preliminary import read_json


def get_idf_weights(all_sentences, idf):
    doc_len = []  # 每个句子包含的term数
    corpus = []
    all_words = []
    for justifications_per_article in all_sentences:
        for line in justifications_per_article:  # each line is a doc
            # line = line.lower()
            # words = line.split()
            words = sentence_segmentation(line, 0)
            if words is not None:
                # words=tokenizer.tokenize(line)
                # words = [lmtzr.lemmatize(w1) for w1 in words]
                document = {}  # dictionary - having terms as key and TF as values of the key.
                doc_len.append(len(words))
                unique_words = list(set(words))
                # 统计每个句子中的term在所有文档（作者将每个句子作为文档）中出现的次数
                for w1 in unique_words:
                    if w1 in idf.keys():
                        idf[str(w1)] += 1
                        # print ("yes, we come here", w1)
                    # else:
                        # idf.update({str(w1):1})

                all_words += unique_words
                for term1 in unique_words:
                    # 统计每个句子中的term在该句子（作者将每个句子作为文档）中出现的次数
                    document[str(term1)] = words.count(term1)

                corpus.append(document)
    all_words = list(set(all_words))
    return doc_len, corpus, all_words, idf


def write_idf_values(all_words, all_sentences, file_name):
    idf = {}
    for each_word in all_words:
        idf[str(each_word)] = 0

    print("vocab len should be same", len(idf))
    doc_lengths, all_documents, aw1, idf2 = get_idf_weights(all_sentences, idf)
    # all_documents: 所有句子（每个句子是一个文档）组成的文档集
    print(len(idf2))
    for terms_TF in all_documents:
        for tf_key in terms_TF:
            # 对每个句子中term出现次数进行函数转变
            terms_TF[tf_key] = 1 + math.log(terms_TF[tf_key])

    total_doc = len(all_documents)
    # avg_doc_len = sum(doc_lengths) / float(len(doc_lengths))

    for each_word in all_words:
        doc_count = idf2[str(each_word)]
        # 公式见论文"Alignment over Heterogeneous Embeddings for Question Answering": pp.2683, 式1
        idf[str(each_word)] = math.log10((total_doc - doc_count + 0.5) / float(doc_count + 0.5))

    with open(file_name, 'w') as outfile:
        json.dump(idf, outfile, ensure_ascii=False)


def pre(all_sentences):
    # input_files = ["train_456-fixedIds.json", "dev_83-fixedIds.json"]
    vocab = []
    for justifications_per_article in all_sentences:
        for sentence in justifications_per_article:
            just_tokens = sentence_segmentation(sentence, 0)
            if just_tokens is not None:
                vocab += just_tokens

    vocab = list(set(vocab))

    # vocab：存放每个句子分词后token的列表（完全不同的token）
    # all_sentences：所有篇章句子
    write_idf_values(vocab, all_sentences, "./Data/GovernmentQA_IDF_values.json")


if __name__ == "__main__":
    pre(sentences_partition(read_json()))
