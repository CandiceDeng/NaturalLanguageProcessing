import __future__

import pickle
import numpy as np
from numpy import linalg as LA

import xml.etree.ElementTree as ET
import sys
import re

def col_softmax(matrix):
	exp_matrix = np.exp(matrix)
	exp_sum = np.sum(exp_matrix, axis = 0)
	rst = exp_matrix/exp_sum

	return rst

def col_l1_norm(matrix):
	sum_ = np.sum(matrix, axis = 0)
	rst = matrix/sum_

	return rst

def parse_data(filepath):
    tree = ET.ElementTree(file = filepath)
    root = tree.getroot()
    threads = []
    for thread in root.findall("Thread"):
        question, answer = dict(),dict()
        for child in thread.findall("RelQuestion"):
            question[thread.attrib['THREAD_SEQUENCE']] = question.get(thread.attrib['THREAD_SEQUENCE'], child.attrib)
            question[thread.attrib['THREAD_SEQUENCE']]["RelQSubject"]= question[thread.attrib['THREAD_SEQUENCE']].get("RelQSubject",child.find("RelQSubject").text)
            question[thread.attrib['THREAD_SEQUENCE']]["RelQBody"]= question[thread.attrib['THREAD_SEQUENCE']].get("RelQBody",child.find("RelQBody").text)
        for child in thread.findall("RelComment"):
            answer[child.attrib['RELC_ID']] = answer.get(child.attrib['RELC_ID'], child.attrib)
            answer[child.attrib['RELC_ID']]["RelCText"] = answer[child.attrib['RELC_ID']].get("RelCText", child.find("RelCText").text)
        threads.append((question,answer))
    return threads

def MAP_calulator(rst_labels, eliminate_no_good = True):
    MAP = 0
    num_rst = len(rst_labels) # total threads
    for labels in rst_labels:
        # print(labels == 'Good')
        ap = 0 # average precision for one thread
        num_good = 0

        for i, l in enumerate(labels):

            if l == 'Good' or l == 'PotentiallyUseful':
                num_good += 1;
                ap += num_good/(i+1)
        MAP += (ap / (i+1))
        if eliminate_no_good and num_good == 0: # no good
            num_rst -= 1

        # print(ap/(i+1))

    MAP /= num_rst
    return MAP, num_rst

# read data
in1 = open('../temp/features.pickle','rb')
features_dict = pickle.load(in1)
in1.close()
# print(features_dict['Q264_R36'])

in2 = open('../feats/treeKernel_pairwise.pickle','rb')
pairtree_dict = pickle.load(in2)
in2.close()

# transfer to probability
transit_dict = {}
for qid in features_dict:
	transit_dict[qid] = col_l1_norm(features_dict[qid]+pairtree_dict[qid])

# eigen-value
score_dict = {}
num_samples = len(transit_dict)
for qid in transit_dict:
	if any(np.isnan(row).any() for row in transit_dict[qid]):
		num_samples  -= 1
		continue
	temp_matrix = transit_dict[qid]
	w, v = LA.eig(temp_matrix)
	# print(round(w[0]) == 1)
	score_dict[qid] = v[:,0]/np.sum(v[:,0])

# parse xml
filepaths = ['../data/train_part1.xml', '../data/train_part2.xml']
rst_labels = []
for filepath in filepaths:
    threads = parse_data(filepath)

    # baseline
    # outfile = open('dev.txt','w+')
    for t in threads:
        question,answers = t

        text = []
        for qid in question:
            break
        if qid not in score_dict:
        	continue

            
        aids = []
        atext = []   
        labels = []
        for aid in answers:
            aids.append(aid)
            labels.append(answers[aid]['RELC_RELEVANCE2RELQ'])
            text.append(answers[aid]['RelCText'])
        # print(labels)
       
        # print(csr_matrix.transpose(qvec).shape, avecs.shape)
        
        indice = np.ndarray.tolist(np.array(score_dict[qid].argsort(axis = 0)[::-1]))
        
        # print(type(indice))
        ranked_labels = [labels[index] for index in indice]
        # print(ranked_labels)
        rst_labels.append(ranked_labels)
        ranked_aids = [aids[index] for index in indice]

MAP, num_rst = MAP_calulator(rst_labels)
print("The MAP of TextRank is {0:.2f}%".format(MAP*100))
num_goods = []
for labels in rst_labels:
	num = 0
	for l in labels:
		if l == 'Good' or l == 'PotentiallyUseful':
			num+=1
	num_goods.append(num)
# print(num_goods)

precision = []
recall = []
for i in range(10):
	num_pre = i + 1
	pre = 0
	re = 0
	for j, labels in enumerate(rst_labels):	
		if 'Good' not in labels and 'PotentiallyUseful' not in labels:
			continue
		# print(len(labels))	
		num_re = num_goods[j]
		co = 0
		k = 0
		while k < num_pre:
			# print(k)
			if labels[k] == 'Good' or labels[k] == 'PotentiallyUseful':
				co += 1
			k += 1
		pre += (co / num_pre)
		re += (co / num_re)
		# if i == 9:
		# 	print(co / num_re == 1)

	precision.append(pre/num_rst)
	recall.append(re/num_rst)
	

out_file = open('../temp/p_r_textrank.pickle','wb')
pickle.dump((precision, recall), out_file,protocol = 2)
out_file.close()
# print(precision, recall)












