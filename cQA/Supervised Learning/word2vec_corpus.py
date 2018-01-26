from gensim.models import word2vec
import re
import sys
from project_595 import parse_data

def write_data(filepath):
    q, a = parse_data(filepath)
    q_body = [q[k]['RelQBody'] for k in q]
    a_body = [a[k]['RelCText'] for k in a]
    with open(filepath+".txt", "w") as w:
        for sent in q_body:
            w.write(sent + "\n")
        for s in a_body:
            w.write(s)
            w.write("\n")

def concate_docs(path1, path2,path3):
    doc_list = [path1, path2,path3]
    with open('semeval_corpus.txt', 'w') as outfile:
        for fname in doc_list:
            with open(fname) as infile:
                outfile.write(infile.read())

def word2vec_model(filepath):
    sentences = word2vec.Text8Corpus(filepath)
    model = word2vec.Word2Vec(sentences, size = 250, window=5, min_count=5, negative = 10, iter = 10)
    model.save("semevalw2v.model")
    return model

if __name__ == "__main__":
    #arguments are: training_part1, training_part2, dev
    write_data(sys.argv[1])
    write_data(sys.argv[2])
    write_data(sys.argv[3])
    concate_docs(sys.argv[1]+".txt", sys.argv[2]+".txt",sys.argv[3]+".txt")
    #model = word2vec_model("semeval_corpus.txt")
    #print(model.most_similar("good", topn= 20))
        

