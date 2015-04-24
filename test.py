#!/usr/bin/env python
#-*- coding:utf-8 -*-
from flask import Flask,request,Response, render_template
import recommend
from common import models
app = Flask(__name__)
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import job_classify

def gen_str(dic):
    html = ""
    html += "<div><br/> Response time: {}   <br/></div>".format(dic["time"])
    if dic.has_key("hl"):
        str_list =dic["sen"]
        array = dic["hl"]
        array = [(dt["end_pos"],dt["start_pos"]) for dt in array]
        array = sorted(array,cmp=lambda x,y:cmp(x[0],y[0]),reverse=True)
        for ed,st in array:
            str_list = str_list[:ed] + "</font>" + str_list[ed:]
            str_list = str_list[:st] + "<font color=red>" + str_list[st:]
        html += "<div><br/> sen:<br/> {} <br/> </div>".format(str_list)
    else :
        html += "<div><br/> sen:<br/> {} <br/> </div>".format(dic["sen"])
    return html

@app.route('/')
def test():
    cv = request.args.get("cv")
    if not cv and not jd:
        cv = ""
    cv = cv.encode("utf-8")
    dic =
    print "6"
    html =  "<html>"
    html += "<body>"
    count = 0
    html += "<div><br/> cv: size {}<br/> {} <br/> </div>".format(len(cv.decode("utf-8")),cv)
    for info in dic:
        count += 1
        html += "<br/><br/><div><h1>Method "+str(count)+"</h1></div>"
        html += gen_str(info)
    html += "<br/><br/><form name='input' action='' method= 'get'> <input type='submit' />"
    html += "<input type='text' name='cv'/> "
    html += "</form>"

    html += "</body></html>"
    return html




if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)




