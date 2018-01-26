import __future__

import matplotlib.pyplot as plt
import pickle
from matplotlib import rc

font = {
        'size'   : 15}
rc('font',**font)

in1 = open('../temp/p_r_textrank.pickle','rb')
tr_rst = pickle.load(in1)
in1.close()

in2 = open('../temp/p_r_tfidf.pickle','rb')
tfidf_rst = pickle.load(in2)
in2.close()

in3 = open('../temp/p_r_random.pickle','rb')
rd_rst = pickle.load(in3)
in3.close()

# print(tr_rst[1], tr_rst[0])
plt.plot(tr_rst[1], tr_rst[0],'-o',linewidth = 3)
plt.plot(tfidf_rst[1], tfidf_rst[0],'--x',linewidth = 1)
plt.plot(rd_rst[1], rd_rst[0],'-.^',linewidth = 1)

plt.xlabel('recall')
plt.ylabel('precision')
plt.legend(['TextRank','TF-IDF', 'Random'])


plt.show()