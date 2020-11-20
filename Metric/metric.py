import re
import string
import collections
from zhon.hanzi import punctuation


# def normalize_answer(s):
#     """Lower text and remove punctuation, articles and extra whitespace."""
#
#     def remove_articles(text):
#         regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
#         return re.sub(regex, " ", text)
#
#     def white_space_fix(text):
#         return " ".join(text.split())
#
#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return "".join(ch for ch in text if ch not in exclude)
#
#     def lower(text):
#         return text.lower()
#
#     return white_space_fix(remove_articles(remove_punc(lower(s))))


def remove_punc(text):
    # ÒÆ³ýÖÐÎÄ×Ö·û´®ÖÐµÄ·ûºÅ
    # :param text:ÐèÒªÒÆ³ýµÄ×Ö·û´®
    # :return:ÒÆ³ý·ûºÅºóµÄ×Ö·û´®
    if text:
        re.sub(r"[%s]+" % punctuation, "", text)
        return text
    else:
        return ""
    # exclude = set(string.punctuation)
    # return "".join(ch for ch in text if ch not in exclude)


def get_tokens(s):
    if not s:
        return []
    answer_text = ""
    for e in s.replace(" ", "").replace("??", "").replace("?", ""):
        answer_text += e
        answer_text += " "
    answer_text = answer_text[0:-1]
    return answer_text.split(" ")


def compute_exact(a_gold, a_pred):
    return int(remove_punc(a_gold) == remove_punc(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(remove_punc(a_gold))
    pred_toks = get_tokens(remove_punc(a_pred))
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    # exact_scores = {}
    # f1_scores = {}
    exact_scores = 0
    f1_scores = 0
    for example in examples:
        qas_id = example.qas_id
        # gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]
        gold_answers = example.answer_text

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]

        # if qas_id not in preds:
        #     print("Missing prediction for %s" % qas_id)
        #     continue

        prediction = preds[qas_id].answer_text
        exact_scores += compute_exact(gold_answers, prediction)
        f1_scores += compute_f1(gold_answers, prediction)
        # exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
        # f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)
    exact_avg_score = exact_scores / len(examples)
    f1_avg_score = f1_scores / len(examples)
    return exact_avg_score, f1_avg_score
