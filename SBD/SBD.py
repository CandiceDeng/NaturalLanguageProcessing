import nltk
import numpy
import scipy
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from sklearn import preprocessing 
import sys

# Name: Nan Deng
# uniquename: dengnan

def data(fname):
	file=open(fname,'rU')
	content=file.readlines()
	datas=[]
	for idx in range(0,len(content)):
		if content[idx].strip().split(' ')[1][-1]=='.':
			token=content[idx].strip().split(' ')	
			L=token[1][:-1]
			try:
				R=content[idx+1].strip().split(' ')[1]
			except:
				R=' '
			#Features Vector
			isLVocab=L
			isRVocab=R
			isLless_than_3=int(len(L)>3)
			try:
				isRNoun=int(True if nltk.pos_tag(nltk.word_tokenize(R))[0][1]=='NN' or nltk.pos_tag(nltk.word_tokenize(R))[0][1]=='NNP' else False)
			except:
				isRNoun=int(False)
			isLmore_than_5=int(len(L)>5)
			isLStop=int(L.lower() in set(stopwords.words('english')))
			try:
				isLCap=int(L[0].isupper())
			except:
				isLCap=0
			try:
				isRCap=int(R[0].isupper())
			except:
				isRCap=0
			datas.append([isLVocab,isRVocab,isLless_than_3,isRNoun,isLmore_than_5,isLStop,isLCap,isRCap])
			# datas.append([isLVocab,isRVocab,isLless_than_3,isLCap,isRCap])
			# datas.append([isRNoun,isLmore_than_5,isLStop])
	return datas

def label(fname):
	file=open(fname,'rU')
	content=file.readlines()
	labels=[]
	for idx in range(0,len(content)):
		if content[idx].strip().split(' ')[1][-1]=='.':
			token=content[idx].strip().split(' ')
			labels.append(token[2])
	return labels

def main():
	train_name=sys.argv[1]
	test_name=sys.argv[2]

	train_data=data(train_name)
	train_label=label(train_name)
	test_data=data(test_name)
	test_label=label(test_name)

	X_train=numpy.array(train_data)
	Y_train=numpy.array(train_label)
	X_test=numpy.array(test_data)
	Y_test=numpy.array(test_label)

	le = preprocessing.LabelEncoder()
	le=le.fit(list(X_train[:,0])+list(X_train[:,1])+list(X_test[:,0])+list(X_test[:,1]))

	X_train[:,0]=le.transform(X_train[:,0])
	X_train[:,1]=le.transform(X_train[:,1])
	X_test[:,0]=le.transform(X_test[:,0])
	X_test[:,1]=le.transform(X_test[:,1])

	clf=DecisionTreeClassifier(criterion='entropy')
	clf.fit(X_train, Y_train)

	prediction=clf.predict(X_test)
	combination=zip(test_label,prediction)

	EOS_predict=0
	EOS_actrual=0
	for each in test_label:
		if each=='EOS':
			EOS_actrual+=1
	for each in combination:
		if each[1]=='EOS' and each[0]==each[1]:
			EOS_predict+=1
	accuracy_rate=float(EOS_predict)/float(EOS_actrual)
	print "{0:.2f}%".format(accuracy_rate * 100)

	with open(test_name,'rU') as test_file, open('SBD.test.out', 'w') as output:
		content=test_file.readlines()
		pre_idx=0
		for item in content:
			line=item.strip().split(' ')
			if line[1][-1]=='.':
				line[2]=prediction[pre_idx]
				pre_idx+=1
				output.write("{} {} {}\n".format(line[0],line[1],line[2]))
			else:
				output.write("{} {} {}\n".format(line[0],line[1],line[2]))

if __name__=='__main__':
	main()