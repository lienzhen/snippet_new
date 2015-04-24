#! -*- coding:utf-8 -*-
import jieba
import gensim
import sys
import time
import re
import numpy as np
from gensim import corpora, models
from common import utils
from operator import itemgetter
word2vec_load_file = '/home/lizz/word2vec_trained/vectors_50.bin'
word2vec = utils.load_vector_file(word2vec_load_file)

stop_words = ["(",")","+","？","—","-","！","】","【","·"," ",",","（","）","~","!","．","<",":",".","\"","•",">","《","》","/","//","，", "、", "。", "；", "、", "：", ";", "\"","“", "”","－"," "]
file_stopwords = '/home/lizz/full_stopwords.txt'

stop = [line.strip() for line in open(file_stopwords).readlines()]

dictionary = corpora.Dictionary.load('/home/lizz/similar_job_describe/data/meta_2014-08-30/dictionary.dat')
tfidf_model = models.TfidfModel.load('/home/lizz/similar_job_describe/data/meta_2014-08-30/tfidf.dat')

def split_words_cv(input_file):
    with open(input_file) as f:
        lines = f.readlines()
        for line in lines:
            if not line:continue
            line = line.strip()
            line = re.sub(r"\s+","",line)
            line = jieba.cut(line)

            output_list = []
            for word in line:
                if word.encode('utf-8') not in stop_words and word.encode('utf-8') not in stop:
                    output_list.append(word.encode('utf-8'))

    return output_list

def get_words_jd(input_file):
    max_length_words = 100
    length = 0
    jd_str =""
    jd_position = {}
    jd_words =[]
    with open(input_file) as f:
        lines = f.readlines()
        for line in lines:
            if not line:continue
            line = line.strip()
            line = re.sub(r"\s+","",line)
            line = line.decode('utf-8')
            for word in line:
                jd_str += word.encode('utf-8')
                length += 1
                if length >= max_length_words:break

    result = jieba.tokenize(jd_str.decode('utf-8'))
    for tk in result:
        if tk[0].encode('utf-8') not in stop_words and tk[0].encode('utf-8') not in stop:
            jd_words.append(tk[0].encode('utf-8'))
            if not jd_position.has_key(tk[0].encode('utf-8')):
               jd_position[tk[0].encode('utf_8')] = {"start_pos" : tk[1], "end_pos" : tk[2]}
    return jd_str, jd_words, jd_position


def tfidf_sort(cv_list):
    doc_bow = dictionary.doc2bow(cv_list)
    cv_tfidf = tfidf_model[doc_bow]

    cv_list_new = []
    cv_tfidf_sort = sorted(cv_tfidf, key = lambda x: (x[0], [1]))
    length = int(len(cv_tfidf_sort))
    for i in range(length):
        if i >= 20:break
        cv_list_new.append(dictionary[cv_tfidf_sort[i][0]].encode('utf-8'))

    return cv_list_new
#cv中tfidf值最高的前20个词与jd前50个词,逐一计算word2vec余弦相似度
def gen_result_1(words_cv, words_jd):
    words_similar_result = []

    for i in range(len(words_jd)):
        words_jd[i] = words_jd[i].lower()
        if(not word2vec.has_key(words_jd[i])):continue
        for j in range(len(words_cv)):
            words_cv[j] = words_cv[j].lower()
            if(not word2vec.has_key(words_cv[j])):continue
            similar = cosine_similar(word2vec[words_cv[j]], word2vec[words_jd[i]])
            if(similar >= 0.5):
                words_similar_result.append([words_jd[i], similar])

    words_similar = sorted(words_similar_result, key = itemgetter(1), reverse = True)
    words_result = []
    for i in range(len(words_similar)):
        words_result.append(words_similar[i][0])
    words_result = list(set(words_result))

    if len(words_result) <= 2 :return words_result
    words = []
    for i in range(3):
        words.append(words_result[i])
    return words

#cv中tfidf值最高的前20个词对应的word2vec向量和与jd前50个词逐一计算余弦相似度
def gen_result_2(words_cv, words_jd):
    cv_word2vec_sum = tuple([0] * 50)
    for i in range(len(words_cv)):
        words_cv[i] = words_cv[i].lower()
        if(not word2vec.has_key(words_cv[i])):continue
        cv_word2vec_sum = vec_add(cv_word2vec_sum, word2vec[words_cv[i]])

    words_similar_result = []
    for i in range(len(words_jd)):
        words_jd[i] = words_jd[i].lower()
        if(not word2vec.has_key(words_jd[i])):continue
        similar = cosine_similar(cv_word2vec_sum, word2vec[words_jd[i]])
        words_similar_result.append([words_jd[i], similar])

    words_similar = sorted(words_similar_result, key = itemgetter(1), reverse = True)
    words_result = []
    for i in range(len(words_similar)):
        words_result.append(words_similar[i][0])
    words_result = list(set(words_result))

    if len(words_result) <= 2 :return words_result
    words = []
    for i in range(3):
        words.append(words_result[i])
    return words

#直接找cv中tfidf最大的前三个词
def gen_result_3(words_jd):
    doc_bow = dictionary.doc2bow(words_jd)
    jd_tfidf = tfidf_model[doc_bow]

    jd_words_new = []
    jd_tfidf_new = sorted(jd_tfidf, key = itemgetter(1), reverse = True)

    for i in range(len(jd_tfidf_new)):
        if i >= 3:break
        jd_words_new.append(dictionary[jd_tfidf_new[i][0]].encode('utf-8'))
    return jd_words_new


if __name__ == '__main__':
    input_cv_file, input_jd_file = sys.argv[1:3]
    cv_list = split_words_cv(input_cv_file)
    cv_words =tfidf_sort(cv_list)

    print "cv中tfidf值最高的前20个词:"
    for i in range(len(cv_words)):print cv_words[i]
    print '\n'

    jd_str, jd_words, jd_position = get_words_jd(input_jd_file)

    print "cv的snippet:"
    print jd_str
    print '\n'

    words_result_1 = gen_result_1(cv_words, jd_words)
    words_result_2 = gen_result_2(cv_words, jd_words)
    words_result_3 = gen_result_3(jd_words)
    positions_result_1 = []
    positions_result_2 = []
    positions_result_3 = []
    print "cv中tfidf值最高的前20个词与jd中前50个词,逐对计算余弦相似度的,结果:"
    for i in range(len(words_result_1)):
        print str(words_result_1[i])
        positions_result_1.append(jd_position[words_result_1[i]])
        print jd_position[words_result_1[i]]["start_pos"]
    print '\n'

    print "cv中tfidf值最高的前20个词的word2vec向量相加与jd中前50个词计算余弦相似度的,结果:"
    for i in range(len(words_result_2)):
        print words_result_2[i]
        positions_result_2.append(jd_position[words_result_2[i]])
        print jd_position[words_result_2[i]]["start_pos"]
    print '\n'

    print "直接找jd中tfidf最大的前三个词,结果:"
    for i in range(len(words_result_3)):
        print words_result_3[i]
        positions_result_3.append(jd_position[words_result_3[i]])
        print jd_position[words_result_3[i]]["start_pos"]
    print '\n'
    #print cv_top20_tfidf, type(cv_list[0])






