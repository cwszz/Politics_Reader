# -*- coding: utf-8 -*-
"""
Evaluation script for CMRC 2018
version: v5 - special
Note: 
v5 - special: Evaluate on SQuAD-style CMRC 2018 Datasets
v5: formatted output, add usage description
v4: fixed segmentation issues
"""
from __future__ import print_function
from collections import Counter, OrderedDict
import re
import json
import nltk
import sys
from predict import read_examples


# split Chinese with English
def mixed_segmentation(in_str, rm_punc=False):
	# in_str = str(in_str).decode('utf-8').lower().strip()
	in_str = str(in_str).lower().strip()
	segs_out = []
	temp_str = ""
	sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
			   '，','。','：','？','！','“','”','；','’','《','》','……','·',
			   '、', '「','」','（','）','－','～','『','』']
	for char in in_str:
		if rm_punc and char in sp_char:
			continue
		if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
			if temp_str != "":
				ss = nltk.word_tokenize(temp_str)
				segs_out.extend(ss)
				temp_str = ""
			segs_out.append(char)
		else:
			temp_str += char

	# handling last part
	if temp_str != "":
		ss = nltk.word_tokenize(temp_str)
		segs_out.extend(ss)

	return segs_out


# remove punctuation
def remove_punctuation(in_str):
	# in_str = str(in_str).decode('utf-8').lower().strip()
	in_str = str(in_str).lower().strip()
	sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
			   '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
			   '「','」','（','）','－','～','『','』']
	out_segs = []
	for char in in_str:
		if char in sp_char:
			continue
		else:
			out_segs.append(char)
	return ''.join(out_segs)


# find longest common string
def find_lcs(s1, s2):
	m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
	mmax = 0
	p = 0
	for i in range(len(s1)):
		for j in range(len(s2)):
			if s1[i] == s2[j]:
				m[i+1][j+1] = m[i][j]+1
				if m[i+1][j+1] > mmax:
					mmax = m[i+1][j+1]
					p = i+1
	return s1[p-mmax:p], mmax


def evaluate(ground_truth_file, prediction_file):
	f1 = 0
	em = 0
	total_count = 0
	skip_count = 0
	for instance in ground_truth_file["data"]:
		# context_id   = instance['context_id'].strip()
		# context_text = instance['context_text'].strip()
		for para in instance["paragraphs"]:
			for qas in para['qas']:
				total_count += 1
				query_id = qas['id'].strip()
				query_text = qas['question'].strip()
				answers = [x["text"] for x in qas['answers']]

				if query_id not in prediction_file:
					sys.stderr.write('Unanswered question: {}\n'.format(query_id))
					skip_count += 1
					continue

				prediction = str(prediction_file[query_id])
				f1 += calc_f1_score(answers, prediction)
				em += calc_em_score(answers, prediction)

	f1_score = 100.0 * f1 / total_count
	em_score = 100.0 * em / total_count
	return f1_score, em_score, total_count, skip_count


def get_raw_scores(examples, preds):
	"""
	Computes the exact and f1 scores from the examples and the model predictions
	"""
	exact_scores = 0
	f1_scores = 0
	for example in examples:
		qas_id = example.qas_id
		gold_answers = example.answer_text

		if not gold_answers:
			# For unanswerable questions, only correct answer is empty string
			gold_answers = [""]
		# if qas_id not in preds:
		#     print("Missing prediction for %s" % qas_id)
		#     continue

		prediction = preds[qas_id]
		exact_scores += calc_em_score(gold_answers, prediction)
		f1_scores += calc_f1_score(gold_answers, prediction)
		# exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
		# f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)
	exact_avg_score = exact_scores / len(examples)
	f1_avg_score = f1_scores / len(examples)
	return f1_avg_score, exact_avg_score


def calc_f1_score(ans, prediction):
	f1_scores = 0
	ans_segs = mixed_segmentation(ans, rm_punc=True)
	prediction_segs = mixed_segmentation(prediction, rm_punc=True)
	lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
	if lcs_len == 0:
		return f1_scores
	precision = 1.0*lcs_len/len(prediction_segs)
	recall = 1.0*lcs_len/len(ans_segs)
	f1 = (2*precision*recall)/(precision+recall)
	f1_scores = f1
	return f1_scores


def calc_em_score(ans, prediction):
	em = 0
	ans_ = remove_punctuation(ans)
	prediction_ = remove_punctuation(prediction)
	if ans_ == prediction_:
		em = 1
	return em


def read_json(prediction_file_path):
	"""
	读取json文件dataset.jsonl
	:return:
	"""
	with open(prediction_file_path, 'r', encoding='gbk') as file:
		prediction = {}
		index = -1
		for line in file.readlines():
			index += 1
			load_dict = json.loads(line)
			prediction[index] = load_dict["answer"]
		return prediction


def predict():
	examples = read_examples()
	prediction_file_path = "./Metric/chinese-roberta-wwm-ext-finetuned-cmrc2018+BM25(Top30, t=2).json"
	predictions = read_json(prediction_file_path)

	F1, EM = get_raw_scores(examples, predictions)
	AVG = (EM + F1) * 0.5
	output_result = OrderedDict()
	# output_result['AVERAGE'] = '%.3f' % AVG
	# output_result['F1'] = '%.3f' % F1
	# output_result['EM'] = '%.3f' % EM
	output_result['AVERAGE'] = AVG
	output_result['F1'] = F1
	output_result['EM'] = EM
	output_result['FILE'] = prediction_file_path
	print(json.dumps(output_result))


if __name__ == '__main__':
	predict()
