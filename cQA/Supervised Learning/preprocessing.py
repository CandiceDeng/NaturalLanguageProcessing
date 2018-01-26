import xml.etree.ElementTree as ET
import re
import numpy as np
from collections import Counter
from gensim.models import word2vec
from scipy.spatial.distance import cosine
import nltk
from nltk.tokenize import RegexpTokenizer
import sys
import pickle

def parse_data(filepath):
	tree = ET.ElementTree(file = filepath)
	root = tree.getroot()
	tokenizer = RegexpTokenizer(r'\w+')
	question, answer = dict(),dict()
	for thread in root.findall("Thread"):
		for child in thread.findall("RelQuestion"):
			question[thread.attrib['THREAD_SEQUENCE']] = question.get(thread.attrib['THREAD_SEQUENCE'], child.attrib)
			if child.find("RelQBody").text != None:
				question[thread.attrib['THREAD_SEQUENCE']]["RelQSubject"]= question[thread.attrib['THREAD_SEQUENCE']].get("RelQSubject",child.find("RelQSubject").text) 
				question[thread.attrib['THREAD_SEQUENCE']]["RelQBody"]= \
					question[thread.attrib['THREAD_SEQUENCE']].get("RelQBody"," ".join(tokenizer.tokenize(child.find("RelQBody").text.lower())))
			else:
				question[thread.attrib['THREAD_SEQUENCE']]["RelQSubject"]= question[thread.attrib['THREAD_SEQUENCE']].get("RelQSubject","Unknown") 
				question[thread.attrib['THREAD_SEQUENCE']]["RelQBody"]= \
							question[thread.attrib['THREAD_SEQUENCE']].get("RelQBody"," ".join(tokenizer.tokenize(child.find("RelQSubject").text.lower())))
		for child in thread.findall("RelComment"):
			answer[child.attrib['RELC_ID']] = answer.get(child.attrib['RELC_ID'], child.attrib)
			answer[child.attrib['RELC_ID']]["RelCText"] = \
					answer[child.attrib['RELC_ID']].get("RelCText", " ".join(tokenizer.tokenize(child.find("RelCText").text.lower())))
	return question, answer

def get_qa_pair(filepath):
    q,a = parse_data(filepath)
    return [(q[re.findall("(.*?)_C",k)[0]]['RelQBody'], a[k]['RelCText']) for k in a if re.findall("(.*?)_C",k)[0] in q]

def text_to_vector(text):
	WORD = re.compile(r'\w+')
	words = WORD.findall(text)
	return Counter(words)

def get_cosine(s1,s2):
	vec1, vec2 = text_to_vector(s1),text_to_vector(s2)
	intersection = set(vec1.keys()) & set(vec2.keys())
	numerator = sum([vec1[x] * vec2[x] for x in intersection])
	denominator = np.linalg.norm([v for v in vec1.values()]) * np.linalg.norm([v for v in vec2.values()])
	if not denominator:
		return 0.0
	else:
		return numerator / denominator

def get_jackard(s1,s2):
	vec1, vec2 = text_to_vector(s1),text_to_vector(s2)
	set_1 = set(vec1.keys())
	set_2 = set(vec2.keys())
	n = len(set_1.intersection(set_2))
	return n / (len(set_1) + len(set_2) - n)

#w2v model is pretrained in another script
def load_w2v_model():
	return word2vec.Word2Vec.load("semevalw2v.model")

def compute_cosine(filepath):
	#load w2v model
	w2v_model = load_w2v_model()
	#get the question, answer pair
	pair = get_qa_pair(filepath)
	postag_pair = [(" ".join([val[1] for val in nltk.pos_tag(nltk.word_tokenize(p[0]))])," ".join([val[1] for val in\
								nltk.pos_tag(nltk.word_tokenize(p[1]))])) for p in pair]
	#four similarity matrix
	cos_sim = np.zeros((int(len(pair)/10),10))
	jackard_sim = np.zeros((int(len(pair)/10),10))
	tag_sim = np.zeros((int(len(pair)/10),10))
	w2v_sim = np.zeros((int(len(pair)/10),10))
	ind = 10

	for i in range(int(len(pair)/10)):
		for j in range(10):
			cos_sim[i,j] = get_cosine(pair[i*ind+j][0],pair[i*ind+j][1])
			jackard_sim[i,j] = get_jackard(pair[i*ind+j][0], pair[i*ind+j][1])
			tag_sim[i,j] = get_cosine(postag_pair[i*ind+j][0], postag_pair[i*ind+j][1])
			#compute word2vec similarity next
			question = np.zeros(max(len(pair[i*ind+j][0]),len(pair[i*ind+j][1])))
			answer = np.zeros(max(len(pair[i*ind+j][0]),len(pair[i*ind+j][1])))
			for k in range(max(len(pair[i*ind+j][0]),len(pair[i*ind+j][1]))):
				question[k] = sum(w2v_model[pair[i*ind+j][0][k]]) if k < len(pair[i*ind+j][0]) \
								and pair[i*ind+j][0][k] in w2v_model else 0
				answer[k] = sum(w2v_model[pair[i*ind+j][1][k]]) if k < len(pair[i*ind+j][1]) \
										and pair[i*ind+j][1][k] in w2v_model else 0
			w2v_sim[i,j] = 1 - cosine(question, answer)

	return cos_sim, jackard_sim, tag_sim, w2v_sim

def concate(path1, path2):	
	cos_sim, jackard_sim, tag_sim, w2v_sim = compute_cosine(path1)
	cos_sim2, jackard_sim2, tag_sim2, w2v_sim2 = compute_cosine(path2)

	mat = np.vstack((cos_sim.flatten(),jackard_sim.flatten(), tag_sim.flatten(),w2v_sim.flatten())).T
	mat2 = np.vstack((cos_sim2.flatten(),jackard_sim2.flatten(), tag_sim2.flatten(),w2v_sim2.flatten())).T
	matrix = np.concatenate((mat,mat2))

	key_dict = dict()
	ind = 10
	array_id = 0
	paths = [path1, path2]

	for path in paths:
		q,a = parse_data(path)
		q_key, a_key = list(q.keys()), list(a.keys())
		for i in range(len(q_key)):
			key_dict[q_key[i]] = key_dict.get(q_key[i],{})
			for j in range(10):
				key_dict[q_key[i]][a_key[i*ind+j]] = key_dict[q_key[i]].get(a_key[i*ind+j],matrix[array_id].tolist())
				array_id += 1

	pickle.dump(key_dict, open("feature.pickle","wb"))

if __name__ == "__main__":
	concate(sys.argv[1],sys.argv[2])

	