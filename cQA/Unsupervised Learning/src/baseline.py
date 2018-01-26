import __future__

import xml.etree.ElementTree as ET
import sys
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from scipy.sparse import csr_matrix
import numpy as np

from random import shuffle, random

import pickle

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

# MAP: rst_labels: ranked label matrix, one row is one thread; 
# eliminate_no_good: whether to consider threads without good comments
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

rst_qids = []
rst_aids = []
rst_labels = []
random_labels = []

vectorizer = TfidfVectorizer()

# parse xml
filepaths = ['../data/train_part1.xml', '../data/train_part2.xml']

for filepath in filepaths:
    threads = parse_data(filepath)

    # baseline
    # outfile = open('dev.txt','w+')
    for t in threads:
        question,answers = t

        text = []
        for qid in question:
            if question[qid]['RelQBody'] != None:
                text.append(question[qid]['RelQSubject'] + " " + question[qid]['RelQBody'])
            else:
                text.append(question[qid]['RelQSubject'])

            
        aids = []
        atext = []   
        labels = []
        for aid in answers:
            aids.append(aid)
            labels.append(answers[aid]['RELC_RELEVANCE2RELQ'])
            text.append(answers[aid]['RelCText'])
        # print(labels)
        vects = vectorizer.fit_transform(text)
        # print(vects)
        qvec = vects[0]
        avecs = vects[1:]
        # print(csr_matrix.transpose(qvec).shape, avecs.shape)
        scores = avecs.dot(csr_matrix.transpose(qvec)).todense()
        indice = np.ndarray.tolist(np.array(scores.argsort(axis = 0)[::-1].reshape(10))[0])
        # print(type(indice))
        ranked_labels = [labels[index] for index in indice]
        ranked_aids = [aids[index] for index in indice]
        # print(scores)
        # print(ranked_labels, ranked_aids)

        rst_qids.append(qid)
        rst_aids.append(ranked_aids)
        rst_labels.append(ranked_labels)

        # random permutation of labels to calculate random baseline
        # print(shuffle(labels, random))
        shuffle(labels, random)
        random_labels.append(labels)

tfidf_baseline, num_rst_tfidf = MAP_calulator(rst_labels)
random_baseline, num_rst_random = MAP_calulator(random_labels)
print("The MAP of TF-IDF is {0:.2f}%".format(tfidf_baseline*100))
print("The MAP of Random is {0:.2f}%".format(random_baseline*100))


num_goods = []
for labels in rst_labels:
    num = 0
    for l in labels:
        if l == 'Good' or l == 'PotentiallyUseful':
            num+=1
    num_goods.append(num)


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
            # print(co / num_re == 1)

    precision.append(pre/num_rst_tfidf)
    recall.append(re/num_rst_tfidf)
    

out_file = open('../temp/p_r_tfidf.pickle','wb')
pickle.dump((precision, recall), out_file,protocol = 2)
out_file.close()
# print(precision, recall)

num_goods = []
for labels in random_labels:
    num = 0
    for l in labels:
        if l == 'Good' or l == 'PotentiallyUseful':
            num+=1
    num_goods.append(num)


precision = []
recall = []
for i in range(10):
    num_pre = i + 1
    pre = 0
    re = 0
    for j, labels in enumerate(random_labels): 
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
            # print(co / num_re == 1)

    precision.append(pre/num_rst_random)
    recall.append(re/num_rst_random)
    

out_file = open('../temp/p_r_random.pickle','wb')
pickle.dump((precision, recall), out_file,protocol= 2)
out_file.close()
# print(precision, recall)





