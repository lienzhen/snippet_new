#!/usr/bin/python
#-*- coding:utf-8 -*-
import re

def cut_detail(paragraph):
    def _add(match):
        if match.group(0) == "\n":
            return "\x01"
        if match.group(0) == "\r":
            return "\x01"
        if match.group(0) == "\t":
            return "\x01"
        return match.group(0) +"\x01"
    split_pattern= u"。|；|！|？|\n|\r|\t|!|;|\?|,|，"
    sentence = re.sub(split_pattern,_add,paragraph)
    sentence = sentence.split("\x01")
    return sentence

def clean(paragraph):
    #pattern = u'(职位描述|工作职责|岗位职责|工作内容|您主要职责|主要职责|职位概要|工作描述|相关要求|工作要求|任职资格|任职要求|岗位要求|能力要求|职位要求|要求|Requirement\(要求\)|岗位素质要求|薪酬福利|待遇|待遇福利|薪资待遇|公司文化|工作地点|招聘条件)\s*(:|：)*'
    #clean_word = re.sub(pattern,"",paragraph)
    pattern2 = u'\d+(\.|,|、|\s)+'
    clean_word = re.sub(pattern2,"",paragraph)
    return clean_word
