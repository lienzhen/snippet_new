#!/usr/bin/python
#-*- coding:utf-8 -*-
from common import sentence_sperate,utils
import time
import jieba
import re
from operator import itemgetter
## method for sentence
def get_recommend_sentence(cv_desc,jd_desc,wordvec_model,tfidf_model,dictionary,stopword_list):
    max_word = 100
    ## get the vector from cv_desc
    start = time.time()
    # cv vector
    cv_desc = cv_desc.decode("utf-8")
    jd_desc = jd_desc.decode("utf-8")
    cv_desc = re.sub(r"\s+"," ",cv_desc)
    jd_desc = re.sub(r"\s+"," ",jd_desc)
    cv_vector = utils.build_sentence_vector_2(cv_desc,wordvec_model,stopword_list)
    jd_desc = sentence_sperate.clean(jd_desc)
    if not cv_vector: cv_vector = [0] * 50
    jd_sentences = sentence_sperate.cut_detail(jd_desc)
    jd_sim = []
    index = 0
    for sen in jd_sentences:
        sen = sen.encode("utf-8")
        vec = utils.build_sentence_vector_2(sen,wordvec_model,stopword_list)
        sim = utils.similarity(vec,cv_vector)
        jd_sim.append((sen.decode("utf-8"),sim,index))
        index += 1
    topk = sorted(jd_sim,cmp=lambda x,y:cmp(x[1],y[1]),reverse=True)
    # calc the num of words
    tot_count = 0
    want_disp = []
    for sen,sim,ids in topk:
        if tot_count + len(sen) <= max_word:
            if len(sen) <= 5:
                sen = ""
            tot_count += len(sen)
            want_disp.append((sen,ids,False))
        else:
            cuts = max_word- tot_count
            ol = len(sen)
            sen = sen[0:cuts]
            ll = len(sen)
            if 2 * ll < ol:
                sen = ""
            elif ll <= 5:
                sen = ""
            want_disp.append((sen,ids,True))
            tot_count += len(sen)
    dips = sorted(want_disp,cmp=lambda x,y:cmp(x[1],y[1]))
    dp = []
    for sen,_,flag in dips:
        if len(sen) == 0:
            if len(dp) == 0:
                dp.append(u"...")
            elif dp[-1] != u"..." :
                dp.append(u"...")
        else :
            dp.append( sen)
            if flag:
                dp.append(u"...")
    Dic = {}
    Dic["sen"] ="".join(dp)
    end = time.time()
    print str(end - start) + "sec"
    Dic["time"] = str(end-start) + " sec"
    return Dic

## give snippet by segment
def get_recommend_segment(cv_desc,jd_desc,wordvec_model,tfidf_model,dictionary,stopword_list):
    max_word = 100
    start = time.time()
    # cv_desc = re.sub(r"\s+","",cv_desc)
    # jd_desc = re.sub(r"\s+","",jd_desc)
    cv_vec = utils.build_sentence_vector(cv_desc,wordvec_model)
    if len(cv_vec) == 0:
        cv_vec = tuple([0] * 50)
    jd_word_list = list(jieba.cut(jd_desc))
    word_list = []
    word_vec = tuple([0] * 50)
    tot_len = 0

    max_sim = 0
    max_str = ""
    for wd in jd_word_list:
        if tot_len + len(wd) > max_word:
            sim = utils.similarity(word_vec,cv_vec)
            if sim > max_sim:
                max_sim = sim
                max_str = "".join(word_list)
            while tot_len + len(wd) > max_word:
                wd2 = word_list.pop(0)
                if wordvec_model.has_key(wd2.lower().encode("utf-8")) and wd2.lower().encode("utf-8") not in stopword_list:
                    vec = wordvec_model[wd2.lower().encode("utf-8")]
                    word_vec = utils.vec_sub(word_vec,vec)
                tot_len -= len(wd2)
        word_list.append(wd)
        if wordvec_model.has_key(wd.lower().encode("utf-8")) and wd.lower().encode("utf-8") not in stopword_list:
            vec = wordvec_model[wd.lower().encode("utf-8")]
            word_vec = utils.vec_add(word_vec,vec)
        tot_len += len(wd)
    sim = utils.similarity(word_vec,cv_vec)
    if sim > max_sim:
        max_sim = sim
        max_str = "".join(word_list)
    Res = {}
    Res["sen"] = "..." + max_str +"..."
    end = time.time()
    print str(end - start) + " sec"
    Res["time"] = str(end-start) + " sec"
    return Res













def tfidf_sort(cv_list,dictionary,tfidf_model):
    doc_bow = dictionary.doc2bow(cv_list)
    cv_tfidf = tfidf_model[doc_bow]
    cv_list_new = []
    cv_tfidf_sort = sorted(cv_tfidf, key = lambda x: (x[0], [1]))
    length = int(len(cv_tfidf_sort))
    for i in range(length):
        if i >= 20:break
        cv_list_new.append(dictionary[cv_tfidf_sort[i][0]].encode('utf-8'))
    return cv_list_new

## lizhzh method 1
def get_recommend_word_1(cv_desc,jd_desc,word2vec_model,tfidf_model,dictionary,stopword_list):
    start = time.time()
    # settings
    max_length_words = 100
    # get cv tfidf top 20
    # cv_desc = re.sub(r"\s+","",cv_desc)
    # jd_desc = re.sub(r"\s+","",jd_desc)
    cv_wordlist = list(jieba.cut(cv_desc))
    stop_words = ["(",")","+","？","—","-","！","】","【","·"," ",",","（","）","~","!","．","<",":",".","\"","•",">","《","》","/","//","，", "、", "。", "；", "、", "：", ";", "\"","“", "”","－"," "]
    cv_wordlist = [wd for wd in cv_wordlist if wd not in stopword_list and wd not in stop_words]
    cv_sorted_list = tfidf_sort(cv_wordlist,dictionary,tfidf_model)
    #find first 50 words in jd
    jd_desc = jd_desc.decode("utf-8")
    jd_desc = jd_desc[0:max_length_words] ## need return
    result = jieba.tokenize(jd_desc)
    jd_words = []
    jd_position = {}
    for tk in result:
        wd = tk[0].lower()
        if wd.encode('utf-8') not in stop_words and wd.encode('utf-8') not in stopword_list:
            jd_words.append(wd.encode('utf-8'))
            if not jd_position.has_key(wd.encode('utf-8')):
               jd_position[wd.encode('utf_8')] = {"start_pos" : tk[1], "end_pos" : tk[2]}
    #cv中tfidf值最高的前20个词与jd前50个词,逐一计算word2vec余弦相似度
    #                                                     -------李贞贞
    words_cv = cv_sorted_list
    words_jd = jd_words
    words_similar_result = []
    for i in range(len(words_jd)):
        words_jd[i] = words_jd[i].lower()
        if(not word2vec_model.has_key(words_jd[i])):continue
        for j in range(len(words_cv)):
            words_cv[j] = words_cv[j].lower()
            if(not word2vec_model.has_key(words_cv[j])):continue
            similar = utils.cosine_similar(word2vec_model[words_cv[j]], word2vec_model[words_jd[i]])
            if(similar >= 0.5):
                words_similar_result.append([words_jd[i], similar])
    words_similar = sorted(words_similar_result, key = itemgetter(1), reverse = True)
    words_result = []
    words = []
    for i in range(len(words_similar)):
        words_result.append(words_similar[i][0])
    words_result = list(set(words_result))
    if len(words_result) <= 3 :
        words = words_result
    else :
        for i in range(3):
            words.append(words_result[i])





##########################################################


    positions_result_1 = []
    for i in range(len(words)):
        positions_result_1.append(jd_position[words[i]])

    Res = {}
    Res["sen"] = jd_desc + "..."
    Res["hl"] = positions_result_1
    end = time.time()
    print str(end - start)  + " sec"
    Res["time"] = str(end-start) + " sec"
    return Res












## lizhzh method 2
def get_recommend_word_2(cv_desc,jd_desc,word2vec_model,tfidf_model,dictionary,stopword_list):
    start = time.time()
    # settings
    max_length_words = 100
    # get cv tfidf top 20
    # cv_desc = re.sub(r"\s+","",cv_desc)
    # jd_desc = re.sub(r"\s+","",jd_desc)
    cv_wordlist = list(jieba.cut(cv_desc))
    stop_words = ["(",")","+","？","—","-","！","】","【","·"," ",",","（","）","~","!","．","<",":",".","\"","•",">","《","》","/","//","，", "、", "。", "；", "、", "：", ";", "\"","“", "”","－"," "]
    cv_wordlist = [wd for wd in cv_wordlist if wd not in stopword_list and wd not in stop_words]
    cv_sorted_list = tfidf_sort(cv_wordlist,dictionary,tfidf_model)
    #find first 50 words in jd
    jd_desc = jd_desc.decode("utf-8")
    jd_desc = jd_desc[0:max_length_words] ## need return
    result = jieba.tokenize(jd_desc)
    jd_words = []
    jd_position = {}
    for tk in result:
        wd = tk[0].lower()
        if wd.encode('utf-8') not in stop_words and wd.encode('utf-8') not in stopword_list:
            jd_words.append(wd.encode('utf-8'))
            if not jd_position.has_key(wd.encode('utf-8')):
               jd_position[wd.encode('utf_8')] = {"start_pos" : tk[1], "end_pos" : tk[2]}
    #cv中tfidf值最高的前20个词对应的word2vec向量和与jd前50个词逐一计算余弦相似度
    #                                                     -------李贞贞
    words_cv = cv_sorted_list
    words_jd = jd_words
    cv_word2vec_sum = tuple([0] * 50)
    for i in range(len(words_cv)):
        words_cv[i] = words_cv[i].lower()
        if(not word2vec_model.has_key(words_cv[i])):continue
        cv_word2vec_sum = utils.vec_add(cv_word2vec_sum, word2vec_model[words_cv[i]])

    words_similar_result = []
    for i in range(len(words_jd)):
        words_jd[i] = words_jd[i].lower()
        if(not word2vec_model.has_key(words_jd[i])):continue

        similar = utils.cosine_similar(cv_word2vec_sum, word2vec_model[words_jd[i]])
        words_similar_result.append([words_jd[i], similar])

    words_similar = sorted(words_similar_result, key = itemgetter(1), reverse = True)
    words_result = []
    for i in range(len(words_similar)):
        words_result.append(words_similar[i][0])
    words_result = list(set(words_result))
    words = []
    if len(words_result) <= 3 :words = words_result
    else :
        for i in range(3):
            words.append(words_result[i])


    ##################################################################

    positions_result_1 = []
    for i in range(len(words)):
        positions_result_1.append(jd_position[words[i]])

    Res = {}
    Res["sen"] = jd_desc + "..."
    Res["hl"] = positions_result_1
    end = time.time()
    print str(end - start) + " sec"
    Res["time"] = str(end-start) + " sec"
    return Res










## lizhzh method 3
def get_recommend_word_3(cv_desc,jd_desc,word2vec_model,tfidf_model,dictionary,stopword_list):
    start = time.time()
    # settings
    max_length_words = 100
    # get cv tfidf top 20
    # jd_desc = re.sub(r"\s+","",jd_desc)
    stop_words = ["(",")","+","？","—","-","！","】","【","·"," ",",","（","）","~","!","．","<",":",".","\"","•",">","《","》","/","//","，", "、", "。", "；", "、", "：", ";", "\"","“", "”","－"," "]
    #find first 50 words in jd
    jd_desc = jd_desc.decode("utf-8")
    jd_desc = jd_desc[0:max_length_words] ## need return
    result = jieba.tokenize(jd_desc)
    jd_words = []
    jd_position = {}
    for tk in result:
        wd = tk[0].lower()
        if wd.encode('utf-8') not in stop_words and wd.encode('utf-8') not in stopword_list:
            jd_words.append(wd.encode('utf-8'))
            if not jd_position.has_key(wd.encode('utf-8')):
               jd_position[wd.encode('utf_8')] = {"start_pos" : tk[1], "end_pos" : tk[2]}
    #直接找cv中tfidf最大的前三个词
    #                                                     -------李贞贞
    words_jd = jd_words
    doc_bow = dictionary.doc2bow(words_jd)
    jd_tfidf = tfidf_model[doc_bow]

    jd_words_new = []
    jd_tfidf_new = sorted(jd_tfidf, key = itemgetter(1), reverse = True)

    for i in range(len(jd_tfidf_new)):
        if i >= 3:break
        jd_words_new.append(dictionary[jd_tfidf_new[i][0]].encode('utf-8'))
    words = jd_words_new
    ########################################################################

    positions_result_1 = []
    for i in range(len(words)):
        positions_result_1.append(jd_position[words[i]])
    Res = {}
    Res["sen"] = jd_desc + "..."
    Res["hl"] = positions_result_1
    end = time.time()
    print str(end - start) + " sec"
    Res["time"] = str(end-start) + " sec"
    return Res
