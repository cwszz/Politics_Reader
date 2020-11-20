import numpy as np
import json
from preliminary import read_json
from preliminary import sentence_segmentation
from preliminary import get_word_embedding
from preliminary import get_sentence_embedding
from preliminary import sentences_partition
from construct_inverted_index import construct_corpus_sentence


def compute_alignment_vector(ques_matrix, ques_tokens_nf, ques_tokens_found, just_sent_matrix,
                             just_tokens_nf, threshold=0.95):
    """
    计算对齐分数，获取剩余未对齐的tokens
    :param ques_matrix:问题的嵌入表示矩阵
    :param ques_tokens_nf:问题中未在embedding中找到的token
    :param ques_tokens_found:问题中在embedding找到的token
    :param just_sent_matrix:证据句子的嵌入表示矩阵
    :param just_tokens_nf:证据句子未在embedding中找到的token
    :param threshold:问题中的token与证据句子中的token类似的阈值
    :return:剩余未匹配的tokens
    """
    just_sent_matrix = just_sent_matrix.transpose()
    match_score = np.matmul(ques_matrix, just_sent_matrix)
    match_score = np.sort(match_score, axis=1)
    # taking the highest element column, namely max-pooling
    max_score_column = match_score[:, -1:]
    max_score_column = np.asarray(max_score_column).flatten()

    # 将大于阈值的设为1，即匹配，否则为0，即不匹配后剩余token
    max_score_column = [1 if score >= threshold else 0 for score in max_score_column]
    # max_score1 = [s1 for s1 in max_score1]

    remaining_terms = []

    # 将匹配分数为0对应的token添加到未匹配token中
    for index, score in enumerate(max_score_column):
        if score == 0:
            remaining_terms.append(ques_tokens_found[index])

    for token in ques_tokens_nf:
        if token in just_tokens_nf:
            max_score_column.append(1)
        else:
            max_score_column.append(0)
            remaining_terms.append(token)
    return remaining_terms


def compute_alignment_score(ques_matrix, ques_tokens_nf, ques_idf_mat, just_matrix, just_tokens_nf, idf_values):
    """
    获得该句子与问题的最终对齐分数
    :param ques_matrix:问题的嵌入表示矩阵
    :param ques_tokens_nf:问题中未在embedding中找到的token
    :param ques_idf_mat:问题中的token在所有篇章中的idf值变换后的idf矩阵
    :param just_matrix:证据句子的嵌入表示矩阵
    :param just_tokens_nf:证据句子未在embedding中找到的token
    :param idf_values:所有篇章中term的idf值
    :return:当前句子与问题的最终对齐分数
    """
    just_sent_matrix = just_matrix.transpose()
    match_score = np.matmul(ques_matrix, just_sent_matrix)
    match_score = np.sort(match_score, axis=1)
    # taking 3 highest element columns
    max_score_column = match_score[:, -1:]
    max_score_column = np.asarray(max_score_column).flatten()

    final_just_score = [a1 * b1 for a1, b1 in zip(max_score_column, ques_idf_mat)]

    # 针对问题中的不在词嵌入中的token
    for token in ques_tokens_nf:
        # 如果也在证据句子中的 不在词嵌入中的token中，如果在idf列表中，则将其idf值作为最终的分数；否则将其分数设为3
        if token in just_tokens_nf:
            if token in idf_values:
                final_just_score.append(idf_values[token])
            else:
                # ~3 was the average IDF score
                final_just_score.append(3)

    return sum(final_just_score)


# def get_index(index, all_sentences):
#     """
#     根据index中的整体索引来获得句子所在篇章索引以及在该篇章中的索引
#     :param index: 所有篇章的句子按顺序平铺放入列表后形成的索引
#     :param all_sentences: 所有篇章中的句子（嵌套列表的形式）
#     :return: 句子所在篇章索引以及在该篇章中的索引，即第几篇第几个
#     """
#     # 按照篇章顺序每判断一次该句子索引不在哪个篇章就要更新一次
#     current_len = index
#     for index_article, justifications_per_article in enumerate(all_sentences):
#         if current_len >= len(justifications_per_article):
#             current_len -= len(justifications_per_article)
#             continue
#         else:
#             return index_article, current_len.item()


def get_alignment_justification(
        question_tokens, all_sentences, embeddings_index, emb_size, idf_values,
):
    """
    获得对齐的证据句子
    :param question_tokens:分词后的问题
    :param all_sentences:所有篇章中的句子（顺序列表的形式）
    :param embeddings_index:嵌入表示
    :param emb_size:嵌入表示维度大小，默认300维
    :param idf_values:所有篇章中term的idf值
    :return:证据检索后按分数大小顺序排列的列表索引（对应句子在所有句子中的位置）；
            最大分数句子对应的剩余未匹配项；
            所有句子对应的剩余未匹配项
    """
    ques_matrix, ques_idf_mat, ques_tokens_nf, ques_tokens_found = get_sentence_embedding(
        question_tokens, embeddings_index, emb_size, idf_values, ques_text=1)
    justification_ques_remaining_terms = {}
    final_alignment_scores = []
    num_remaining_terms = []
    # 当前句子所在篇章内句子索引
    for index_sentence, justification in enumerate(all_sentences):
        just_tokens = sentence_segmentation(justification, flag=1)
        if just_tokens is None:
            try:
                all_sentences.remove(justification)
            except ValueError:
                pass
            continue

        if just_tokens is not None:
            # 获得证据句子的嵌入
            just_matrix, just_tokens_nf, just_tokens_found = get_sentence_embedding(
                just_tokens, embeddings_index, emb_size, idf_values)
            # 将证据句子与问题进行匹配，得到匹配token以及未匹配的剩余tokens
            index = index_sentence
            justification_ques_remaining_terms.update(
                {index: compute_alignment_vector(
                    ques_matrix, ques_tokens_nf, ques_tokens_found, just_matrix, just_tokens_nf)})
            # 当前索引句子与问题的对齐分数
            index_score = compute_alignment_score(
                ques_matrix, ques_tokens_nf, ques_idf_mat, just_matrix, just_tokens_nf, idf_values,
            )
            num_remaining_terms.append(len(justification_ques_remaining_terms[index]))
            final_alignment_scores.append(index_score)

    # 所有句子及其分数的索引
    all_indexes = list(np.argsort(final_alignment_scores)[::-1])  # [:subgraph_size]

    # # final_index[0]即最大分数对应的句子在所有句子中的索引，根据其寻找在context中的位置，即第几篇第几个
    # # 以获得该证据句子对应的justification_ques_remaining_terms[index]
    # index_article, index_sentence = get_index(all_indexes[0], all_sentences)
    index = all_indexes[0]
    return all_indexes, justification_ques_remaining_terms[index], justification_ques_remaining_terms


def one_iteration_block(final_indexes, first_iteration_index1, remaining_tokens1_3, ques_terms,
                        all_sentences, embedding_index, emb_size, idf_values):
    """
    一次迭代获取证据句子
    :param final_indexes:一个证据链最终产生的证据句子索引
    :param first_iteration_index1:前一次证据检索后按分数大小顺序排列的列表索引
    :param remaining_tokens1_3:前一次迭代最高匹配分数对应的句子的剩余未匹配项
    :param ques_terms:分词后的问题
    :param all_sentences:所有篇章中的句子（嵌套列表的形式）
    :param embedding_index:嵌入表示
    :param emb_size:嵌入表示维度大小，默认300维
    :param idf_values:所有篇章中term的idf值

    :return:当前迭代后一个证据链最终产生的证据句子索引；
            当前迭代后按分数大小顺序排列的列表索引；
            当前迭代后最高匹配分数对应的句子的剩余未匹配项
    """
    # 获取final_indexes中最后一个索引对应的句子在篇章中的索引(,)
    # index_article_inverse, index_sentence_inverse = get_index(final_indexes[-1], all_sentences)
    try:
        # selected_just_tokens = sentence_segmentation(
        #     all_sentences[index_article_inverse][index_sentence_inverse], flag=1)
        selected_just_tokens = sentence_segmentation(
            all_sentences[final_indexes[-1]], flag=1)

    except IndexError:
        # please pardon these error messages, they do not appear when running the main file
        # and were used only for debugging
        print("the error is coming because ", final_indexes, len(all_sentences))
        # index_article, index_sentence = get_index(final_indexes[0], all_sentences)
        # selected_just_tokens = sentence_segmentation(all_sentences[index_article][index_sentence], flag=1)
        selected_just_tokens = sentence_segmentation(all_sentences[final_indexes[0]], flag=1)
    if len(remaining_tokens1_3) <= 2:  # which can be considered as a very short query
        new_query_terms = remaining_tokens1_3 + list(set(selected_just_tokens) - set(ques_terms))

    else:
        new_query_terms = remaining_tokens1_3
        # new_query_terms = remaining_tokens1_3 + list(set(selected_just_tokens))
    # second_iteration_index1: 新一次迭代产生的证据句子索引
    # remaining_tokens1_4: 新一次迭代最高匹配分数对应的句子的剩余未匹配项
    # remaining_tokens2_all: 新一次迭代所有句子对应的剩余未匹配项
    second_iteration_index1, remaining_tokens1_4, remaining_tokens2_all = get_alignment_justification(
        new_query_terms, all_sentences, embedding_index, emb_size, idf_values)

    # 对于新一次迭代产生的证据句子索引
    for i1 in second_iteration_index1:
        # 如果该句子索引出现在之前的final_indexes中，则不做处理
        if i1 in final_indexes:
            pass
        else:
            # 前一次迭代的剩余项依然在当前迭代的所有剩余项中
            # i.e. none of the previously remaining ques terms were covered in this iteration
            # 则迭代结束（作者人为设置）
            # index_article, index_sentence = get_index(i1, all_sentences)
            # index = str(index_article) + ',' + str(index_sentence)
            if len(set(remaining_tokens1_3).intersection(set(
                    remaining_tokens2_all[i1]))) == len(set(remaining_tokens1_3)):
                continue
                # return final_indexes, [], []
            # 否则添加当前句子为满足条件的当前迭代的最高分数句子
            final_indexes.append(i1)
            # 更新满足条件的当前迭代的最高分数句子对应的剩余未匹配项
            remaining_tokens1_4 = remaining_tokens2_all[i1]
            break

    return final_indexes, second_iteration_index1, remaining_tokens1_4


def get_iterative_alignment_justifications(
        question_tokens, all_sentences, idf_values, embeddings_index,
        max_iteration=6, emb_size=300
):
    """
    Alignment over embeddings for sentence selection iteratively
    :param question_tokens:分词后的问题
    :param all_sentences:所有篇章中的句子（嵌套列表的形式）
    :param idf_values:所有篇章中term的idf值
    :param embeddings_index:嵌入表示
    :param max_iteration:一个证据链最大迭代次数
    :param emb_size:嵌入表示维度大小，默认300维
    :return:一个证据链最终检索的证据句子的索引
    """
    final_indexes = []
    # First iteration is here
    first_iteration_index1, remaining_tokens1, remaining_tokens2 = get_alignment_justification(
        question_tokens, all_sentences, embeddings_index, emb_size, idf_values)
    # print(first_iteration_index1[0])

    # 将前一次迭代产生的匹配分数最高的句子索引放入final_indexes
    # final_indexes += [first_iteration_index1[0]]  # , first_iteration_index1[1]]
    final_indexes.append(first_iteration_index1[0])  # , first_iteration_index1[1]]

    # i.e. we are making 6 iteration loop to keep the experiments fast but even if you make it 100
    # or the same as number of sentences in the paragraph, the F1 score remains the same as iteration
    # is completing within first 3-4 loop
    for i in range(max_iteration):
        # second and other iterations
        final_indexes, first_iteration_index1, remaining_tokens1 = one_iteration_block(
            final_indexes, first_iteration_index1, remaining_tokens1, question_tokens,
            all_sentences, embeddings_index, emb_size, idf_values)
        # 如果本次迭代满足条件的最高分数句子对应的剩余未匹配项数为0，即完全匹配，则结束迭代
        if len(remaining_tokens1) == 0:
            return final_indexes

    # print ("the final indices look like ", Final_indexes)
    # 或者一直迭代直到达到超参迭代上限
    return final_indexes


def get_iterative_alignment_justifications_non_parametric_parallel_evidence(
        ques_terms, all_sentences, idf_values, embedding_index,
        parallel_evidence_num=3, max_iteration=6, emb_size=300):
    """
    Alignment over embeddings for sentence selection, inducing several evidence chain
    :param ques_terms:分词后的问题
    :param all_sentences:所有篇章中的句子（嵌套列表的形式）
    :param idf_values:问题中的token在所有篇章中的idf值
    :param embedding_index:嵌入表示
    :param parallel_evidence_num:
    :param max_iteration:一个证据链最大迭代次数
    :param emb_size:嵌入表示维度大小，默认300维
    :return:多个证据链最终检索的证据句子的索引
    """
    all_final_indexes = []
    # creating a parallel evidence chain of size n, the size is 3 here
    for num_chain in range(parallel_evidence_num):
        final_indexes = []
        # First iteration is here
        first_iteration_index1, remaining_tokens1, remaining_tokens2 = get_alignment_justification(
            ques_terms, all_sentences, embedding_index, emb_size, idf_values)
        # 对于第一次迭代产生的证据句子索引中的每个句子
        for top_ind in first_iteration_index1:
            if top_ind in all_final_indexes:
                pass
            # 如果该句子不在最终产生的证据句子中（按照分数从大到小的顺序），则将其添加到final_indexes中
            # 该操作的目的是使得每个证据链的第一个证据都不相同
            else:
                final_indexes += [top_ind]  # , first_iteration_index1[1]]
                break
        if len(final_indexes) == 0:
            final_indexes += [first_iteration_index1[0]]
            print("so we did come in this case ")
        # i.e. we are making 6 iteration loop to keep the experiments fast but even if you make it 100 or
        # the same as number of sentences in the paragraph, the F1 score remains the same as
        # iteration is completing within first 3-4 loop
        for i in range(max_iteration):
            # second iteration
            final_indexes, first_iteration_index1, remaining_tokens1 = one_iteration_block(
                final_indexes, first_iteration_index1, remaining_tokens1, ques_terms, all_sentences,
                embedding_index, emb_size, idf_values)

            # 如果迭代几次后最高分数句子对应的剩余未匹配项为0，即全部匹配
            if len(remaining_tokens1) == 0:
                # 并且如果当前证据链是最后一组，则将final_indexes中筛选的句子送入all_final_indexes并且直接结束本函数
                if num_chain == parallel_evidence_num-1:
                    all_final_indexes += final_indexes
                    all_final_indexes = list(set(all_final_indexes))
                    return all_final_indexes
                # 否则结束迭代过程，进入下一个证据链
                else:
                    break

        all_final_indexes += final_indexes
    all_final_indexes = list(set(all_final_indexes))
    return all_final_indexes


def dispose(context, question, all_sentences, embeddings_index,
            emb_size, bm25model, sentence_dict, top_k=20):
    """
    按步实现无监督对齐迭代证据算法
    :param context:原始上下文数据
    :param question:当前问题
    :param all_sentences:所有篇章中的句子（嵌套列表的形式）
    :param embeddings_index:嵌入表示
    :param emb_size:嵌入表示维度大小，默认300维
    :param bm25model:构造的BM25模型
    :param sentence_dict:
    :param top_k:筛选前多少个最大BM25分数句子
    :return:一个证据链最终得到的证据句子或者多个证据链最终得到的证据句子
    """
    question_tokens = sentence_segmentation(question, flag=1)

    scores = bm25model.get_scores(question_tokens)
    idx_list = np.array(scores).argsort()[-top_k:][::-1].tolist()
    new_sentences = []
    for index in idx_list:
        index_list = sentence_dict[index].split(",")
        index_document = int(index_list[0])
        index_sentence = int(index_list[1])
        new_sentences.append(all_sentences[index_document][index_sentence])
        # current_len = index
        # for index_article, justifications_per_article in enumerate(all_sentences):
        #     if current_len >= len(justifications_per_article):
        #         current_len -= len(justifications_per_article)
        #         continue
        #     else:
        #         new_sentences.append(all_sentences[index_article][current_len])
        #         break

    # question_idf = get_idf(question_tokens, context)
    with open("./Data/GovernmentQA_IDF_values.json") as json_file:
        idf_values = json.load(json_file)
    final_indexes = get_iterative_alignment_justifications(
        question_tokens, new_sentences, idf_values, embeddings_index,
        max_iteration=6, emb_size=emb_size)

    # # 多条证据链
    # final_indexes = get_iterative_alignment_justifications_non_parametric_parallel_evidence(
    #     question_tokens, new_sentences, idf_values, embeddings_index,
    #     parallel_evidence_num=3, max_iteration=6, emb_size=emb_size)
    justifications = []
    for final_index in final_indexes:
        justifications.append(new_sentences[final_index.astype(np.int32)])
    selected = "。".join(justifications) + "。"

    return selected


if __name__ == '__main__':
    embeddings, size = get_word_embedding()
    bm25, _, sentence_dict = construct_corpus_sentence()
    # content, sentences = read_txt()
    context = dispose(
        read_json(), "壶兰计划是哪个市的？", sentences_partition(read_json()),
        embeddings, size, bm25, sentence_dict
    )
    # context = dispose(content, "壶兰计划是哪个市的？", sentences, embeddings, size, bm25)
    print(context)
