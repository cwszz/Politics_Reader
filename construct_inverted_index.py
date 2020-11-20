import jieba.posseg as pseg
import jieba
import codecs
import os
from gensim import corpora
from gensim.summarization import bm25
from preliminary import read_json
from preliminary import sentences_partition


def get_stop_words():
    file_path = './Data/hit_stopwords.txt'
    stop_words = codecs.open(file_path, 'r', encoding='utf8').readlines()
    stop_words = [w.strip() for w in stop_words]
    return stop_words

# stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']


def tokenization(filename):
    result = []
    with open(filename, 'r') as f:
        text = f.read()
        # words = list(jieba.cut_for_search(text))
        words = pseg.cut(text)
    for word, flag in words:
        # if flag not in stop_flag and word not in stopwords:
        stop_words = get_stop_words()
        if word not in stop_words:
            result.append(word)
    return result


def construct_corpus(dirname):
    corpus = []
    filenames = []
    for root, dirs, files in os.walk(dirname):
        for f in files:
            # if re.match(r'[\u4e00-\u9fa5]*.txt', f):
            corpus.append(tokenization(root + f))
            filenames.append(f)

    dictionary = corpora.Dictionary(corpus)
    print(len(dictionary))

    # doc_vectors = [dictionary.doc2bow(text) for text in corpus]
    # vec1 = doc_vectors[0]
    # vec1_sorted = sorted(vec1, key=lambda x: x[1], reverse=True)
    # print(len(vec1_sorted))
    # for term, freq in vec1_sorted[:5]:
    #     print(dictionary[term])

    bm25model = bm25.BM25(corpus)
    return bm25model, filenames


def tokenization_sentence(sentence):
    result = []
    # words = list(jieba.cut_for_search(text))
    words = pseg.cut(sentence)
    for word, flag in words:
        # if flag not in stop_flag and word not in stopwords:
        stop_words = get_stop_words()
        if word not in stop_words:
            result.append(word)
    return result


def construct_corpus_sentence():
    corpus = []
    sentence_dict = {}  # 存储所有句子所在文档以及句子索引（二级完全平铺）
    count = 0
    context = read_json()
    all_sentences = sentences_partition(context)
    for index_document, sentences_per_document in enumerate(all_sentences):
        for index_sentence, sentence in enumerate(sentences_per_document):
            corpus.append(tokenization_sentence(sentence))
            sentence_dict[count] = str(index_document) + "," + str(index_sentence)
            count += 1

    dictionary = corpora.Dictionary(corpus)
    print(len(dictionary))

    bm25model = bm25.BM25(corpus)
    return bm25model, all_sentences, sentence_dict


# average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())


def test(dirname='E:\\Putian\\Data\\txt_documents\\'):
    bm25model, filenames = construct_corpus(dirname)
    query_str = '海外人才引进安置补助标准是什么？'
    query = list(jieba.cut_for_search(query_str))
    scores = bm25model.get_scores(query)
    # scores.sort(reverse=True)
    print(scores)

    idx = scores.index(max(scores))
    print(idx)

    file_name = filenames[idx]
    print(file_name)

    with open(dirname + file_name, 'r') as f:
        print(f.read())


def get_score(query, dirname='E:\\Putian\\Dataset\\txt_documents\\'):
    bm25model, filenames = construct_corpus(dirname)
    # query_str = '海外人才引进安置补助标准是什么？'
    # query = list(jieba.cut_for_search(query_str))
    scores = bm25model.get_scores(query)
    # scores.sort(reverse=True)
    print(scores)

    idx = scores.index(max(scores))
    print(idx)

    file_name = filenames[idx]
    print(file_name)

    with open(dirname + file_name, 'r') as f:
        content = f.read()
        print(content)
    return content


if __name__ == "__main__":
    test()
