#!/usr/bin/python
#-*- coding:utf-8 -*-
from gensim import corpora, models

def load_dictionary(input_path):
    print "load dictioanry"
    dictionary = corpora.Dictionary.load(input_path)
    return dictionary

def load_tfidf(input_path):
    print "load tfidf"
    tfidf_model = models.TfidfModel.load(input_path)
    return tfidf_model
def load_word_vector(vector_file, input_dim=200):
    file_dims = input_dim + 1
    word2vec_dict = {}
    with open(vector_file) as f:
        for i in f.readlines():
            i = i.strip()
            items = i.split(" ")
            if len(items) != file_dims: continue
            word2vec_dict[items[0]] = tuple(map(lambda x: float(x), items[1:]))
    return word2vec_dict


def load_vector_file(vector_file):
    dims = -1
    word2vec_dict = {}
    with open(vector_file) as f:
        first_line = f.readline()
        first_line = first_line.strip()
        line_num, dims = first_line.split()
    dims = int(dims)
    if dims == -1: return word2vec_dict
    return load_word_vector(vector_file, input_dim=dims)


def load_word2vec(input_path):
    print "load word2vec"
    return load_vector_file(input_path)

def load_stopword(input_path):
    print "load stoplist"
    stoplist = []
    fw = open(input_path)
    while True:
        line = fw.readline()
        if not line: break
        line = line.strip()
        stoplist.append(line)
    return stoplist



