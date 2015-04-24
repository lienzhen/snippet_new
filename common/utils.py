#/usr/bin/python
#-*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import math
import numpy as np
import jieba

stopwords = set(["、","兼","以及","且","及","和","与","(",")","+","？","—","！","】","【","·"," ",",","（","）","~","!","．","<",":",".","\"","•",">","《","》","/","//","，", "、", "。"])




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


#编辑距离
def ld_distance(str_1, str_2):
    list_1 = list(str_1.decode("utf-8"))
    list_2 = list(str_2.decode("utf-8"))
    distanceArray = [[-1 for col in range(len(list_2)+1)] for row in range(len(list_1)+1)] #初始化距离矩阵
    for col in range(len(list_2)+1):
        distanceArray[0][col] = col
    for row in range(len(list_1)+1):
        distanceArray[row][0] = row
    for i in range(1, len(list_1)+1):
        for j in range(1, len(list_2)+1):
            cost = 1
            if list_1[i-1] == list_2[j-1]:
                cost = 0
            distanceArray[i][j] = min(distanceArray[i-1][j-1] + cost, distanceArray[i-1][j] + 1, distanceArray[i][j-1] + 1)
    return distanceArray[len(list_1)-1][len(list_2)-1]

def vec_add(v1, v2):
    return [(v1[i]+v2[i]) for i in range(len(v1))]

def vec_sub(v1, v2):
    return [(v1[i]-v2[i]) for i in range(len(v1))]

#euclidean metric
def euclidean_distance(v1, v2):
    sum = 0.0
    for i in range(len(v1)):
        sum += (v1[i]-v2[i])**2
    return math.sqrt(sum)

def euclidean_dist(x, y):
    """ This is a hot function, hence some optimizations are made. """
    diff = np.array(x) - y
    return np.sqrt(np.dot(diff, diff))


#cosine similarity
def similarity(d1, d2):
    hit = sum([(d1[i]*d2[i]) for i in range(len(d1))]) + 0.0
    sum1 = math.sqrt(sum([d1[k]*d1[k] for k in range(len(d1))]))
    sum2 = math.sqrt(sum([d2[k]*d2[k] for k in range(len(d2))]))
    if sum1 == 0 or sum2 == 0:
        return -1
    return hit / (sum1 * sum2)


def cosine_similar(x, y):
    if len(x) != len(y): return 0
    part1 = np.dot(x, y)
    part2 = ((np.dot(x, x) * np.dot(y, y)) ** 0.5)
    if part2 == 0: return 0
    return part1/part2

def gaussian(dist, sigma=10.0):
    return math.e ** (-dist**2/(2*sigma**2))


def build_vector_from_file(data_file, word2vector_dict, sep="\x01"):
    detail_vector_dict = {}
    with open(data_file) as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.strip()
            line = line.upper()
            if line.find(sep) != -1:
                sentence, cv_ids = line.split(sep, 1)
            else:
                sentence = line
            vector = build_sentence_vector(sentence, word2vector_dict)
            if vector:
                detail_vector_dict[line] = tuple(vector)
            else:
                detail_vector_dict[line] = ()
    return detail_vector_dict


def build_sentence_vector(sentence, word_vec):
    sentence = sentence.upper()
    words = list(jieba.cut(sentence))
    vector = []
    for i in words:
        i = i.encode("utf-8")
        if i in stopwords: continue
        if word_vec.has_key(i):
            if vector:
                vector = vec_add(vector, word_vec[i])
            else:
                vector = word_vec[i]
    return vector


def build_sentence_vector_2(sentence, word_vec,stopword_list):
    sentence = sentence.upper()
    words = list(jieba.cut(sentence))
    vector = []
    for i in words:
        i = i.encode("utf-8")
        if i in stopwords: continue
        if i in stopword_list: continue
        if word_vec.has_key(i):
            if vector:
                vector = vec_add(vector, word_vec[i])
            else:
                vector = word_vec[i]
    return vector


