from preprocessing import parse_data
import sys
from sklearn.svm import SVC
from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
import sys

def get_pickle(path):
	return pickle.load(open(path, "rb"))

def formatting(path):
	d = get_pickle(path)
	f = dict()
	for outer_k in d:
		f[outer_k] = f.get(outer_k, {})
		for inner_k, inner_v in d[outer_k].items():
			f[outer_k][inner_k] = f[outer_k].get(inner_k, [inner_v])
	return f

def get_all_feature(path1, path2, path3, path4):
	feature = {inner_k: inner_v for k in get_pickle(path1) for inner_k, inner_v in get_pickle(path1)[k].items()}
	for path in [path2, path3, path4]:
		pk_dict = formatting(path) if path == "treeKernel_for_vector.pickle" else get_pickle(path)
		feature = {inner_k: feature[inner_k] + list(pk_dict[outer_k][inner_k]) for outer_k in pk_dict for inner_k in pk_dict[outer_k] if inner_k in feature}
	return feature

def get_label(xml_path1, xml_path2):
	label = dict()
	for path in [xml_path1, xml_path2]:
		q,a = parse_data(path)
		for k in a:
			label[k] = label.get(k, a[k]["RELC_RELEVANCE2RELQ"])

	return {k:1 if label[k] == "Good" else -1 for k in label}

def training(xml_path1,xml_path2, pk1, pk2, pk3,pk4):
	k_label = get_label(xml_path1,xml_path2)
	k_feature = get_all_feature(pk1,pk2,pk3,pk4)
	feature = np.zeros((len(k_feature), len(k_feature['Q1_R1_C1'])))
	label = np.zeros(len(k_feature))
	ind = 0

	for k in k_feature:
		if k in k_label:
			feature[ind] = k_feature[k]
			label[ind] = k_label[k]
		ind += 1
	feature[np.isnan(feature)] = 0
	scaler = StandardScaler()
	feature= scaler.fit_transform(feature)

	clf = SVC(C = 5,gamma = 0.001, kernel = 'rbf', degree = 2)
	#scores = cross_val_score(clf, feature, label, cv= 5)

	return cross_val_score(clf, feature, label, cv= 5), cross_val_score(clf, feature, label, cv= 5, scoring = "precision"), \
			cross_val_score(clf, feature, label, cv= 5, scoring = "recall")

if __name__ == "__main__":
	#labels = get_label("data/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml","data/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml")
	#print(Counter(labels.values()))
	accu, prec, recall = training(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],sys.argv[5],sys.argv[6])
	#accu, prec, recall = training("data/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml", "data/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml", 
	#	"feature.pickle", "heuristics.pickle", "treeKernel_for_vector.pickle","LCS.pickle")
	print("accuracy: ", accu)
	print("precision: ",prec)
	print("recall: ", recall)
	



		
