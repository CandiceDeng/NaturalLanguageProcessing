import nltk
from nltk import word_tokenize
import re
import sys
import math

# Name: Nan Deng
# uniquename: dengnan

def find_unigram(fname):
	with open(fname,'rU') as file:
		unigram_dic={}
		content=file.read().strip().split()
		for word in content:
			if re.match('[A-Za-z0-9]+',word):
				if word not in unigram_dic:
					unigram_dic[word]=1
				else:
					unigram_dic[word]+=1
	return unigram_dic

def CHI_sq(bigram): 
	obs=bigram_dic[bigram]
	w1_count=0
	w2_count=0
	for w in bigram_dic.keys():
		if w[0]==bigram[0]:
			w1_count+=bigram_dic[w]
		if w[1]==bigram[1]:
			w2_count+=bigram_dic[w]
	exp=float(w1_count)/total_count*float(w2_count)
	return (obs-exp)*(obs-exp)/exp

def top_CHI():
	CHI_score=[]
	for bigram in bigram_dic.keys():
		CHI_score.append((bigram,CHI_sq(bigram)))
	sorted_CHI=sorted(CHI_score, key=lambda x:x[1], reverse=True)
	for idx in range(1,21):
		print "{} {}".format(sorted_CHI[idx-1][0],sorted_CHI[idx-1][1])

def PMI(bigram):
	p_w1w2=bigram_dic[bigram]/float(total_count)
	p_w1=unigram_dic[bigram[0]]/float(total_uni)
	p_w2=unigram_dic[bigram[1]]/float(total_uni)
	return math.log(p_w1w2/(p_w1*p_w2),2)

def top_PMI():
	PMI_score=[]
	for bigram in bigram_dic.keys():
		PMI_score.append((bigram,PMI(bigram)))
	sorted_PMI=sorted(PMI_score, key=lambda x:x[1], reverse=True)[:20]
	for idx in sorted_PMI:
		print "{} {}".format(idx[0],idx[1])

def main():
	fname=sys.argv[1]
	call=sys.argv[2]		

	global unigram_dic
	unigram_dic=find_unigram(fname)

	file=open(fname,'rU')
	content=file.read().split()
	copy=content[:]
	copy.append('.')
	copy.pop(0)
	bigram=zip(content,copy)
	all_bigram=[]
	global bigram_dic
	bigram_dic={}
	for gram in bigram:
		if re.match('[A-Za-z0-9]+',gram[0]) and re.match('[A-Za-z0-9]+',gram[1]):
			if gram not in bigram_dic:
				bigram_dic[gram]=1
			else:
				bigram_dic[gram]+=1	

	for key in bigram_dic.keys():
		if bigram_dic[key]<5:
			del bigram_dic[key]

	global total_count
	total_count=0
	for bigram in bigram_dic:
		total_count+=bigram_dic[bigram]

	global total_uni
	total_uni=0
	for gram in unigram_dic:
		total_uni+=unigram_dic[gram]

	if call=='chi-square':
		print "Top 20 CHI-square Score"
		top_CHI()
	if call=='PMI':
		print "Top 20 PMI Score"
		top_PMI()

	

if __name__=='__main__':
	main()
