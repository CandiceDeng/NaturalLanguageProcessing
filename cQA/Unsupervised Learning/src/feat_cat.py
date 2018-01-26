import __future__

import pickle
import numpy as np

# data preprocess
in1 = open('../feats/feature.pickle','rb')
feat_dict = pickle.load(in1)
in1.close()

in2 = open('../feats/LCS.pickle','rb')
LCS_dict = pickle.load(in2)
in2.close()
for qid in LCS_dict:
	for aid in LCS_dict[qid]:
		LCS_dict[qid][aid] = list(LCS_dict[qid][aid])


in3 = open('../feats/heuristics.pickle','rb')
heur_dict = pickle.load(in3)
in3.close()


in4 = open('../feats/treeKernel_for_vector.pickle','rb')
tree_dict = pickle.load(in4)
in4.close()
# print(feat_dict['Q264_R36'])

# print(tree_dict)
# print(LCS_dict.keys())
features_dict = {}
for qid in feat_dict:
	temp_dict = {}
	for aid in feat_dict[qid]:
		# vec = np.concatenate((feat_dict[qid][aid], LCS_dict[qid][aid]))
		feat_dict[qid][aid].append(tree_dict[qid][aid])
		vec = feat_dict[qid][aid] + LCS_dict[qid][aid] + heur_dict[qid][aid]
		temp_dict[aid] = vec

	matrix = []
	for i in range(1,11):
		matrix.append(temp_dict[qid+'_C'+str(i)])
	rst = np.array(matrix).dot(np.transpose(np.array(matrix)))
	np.fill_diagonal(rst, 0)


	features_dict[qid] = rst

out_file = open('../temp/features.pickle','wb')
pickle.dump(features_dict, out_file, protocol = 2)
out_file.close()

# print(features_dict)