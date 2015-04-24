# Snippet
This is a snippet project for displaying CV_2_JD result. <br>
There are five methods. <br>
Method 1 to 3 are based on `tfidf` and `word2vec` and they will `highlight` the words. <br>
Method 4 is `sentence based` and Method 5 is `segment based`. <br>


The function of the get_recommend_xxxx will return a dictionary like: <br>
        
        { 
        "sen": "sentence for display", 
        "time": "0.001 sec", 
        "hl": [{"start_pos":0,"end_pos":2}] 
        } 

sen is the sentence for display <br>
time is the response time <br>
hl is the index of the hightlight words <br>
## Models
Model files are all in `./model_data/`. <br>
vectors_50.bin ----------------word2vec with 50 dims <br>
tfidf.dat ----------------- tfidf models <br>
dictionary.dat ---------------------- dictionary  generated by gensim <br>
full_stopwords.txt ----------------------- stopword list <br>

### How to load models
We can `import` `models` from `common`. Or you can just get it from the Example. <br> 





## Example

```python
from common import models
import recommend

word2vec = models.load_word2vec("./model_data/vectors_50.bin")
tfidf_model = models.load_tfidf("./model_data/tfidf.dat")
dictionary = models.load_dictionary("./model_data/dictionary.dat")
stopword_list = models.load_stopword("./model_data/full_stopwords.txt")

cv = "This is a cv"
jd = "This is a jd"

#method 1
recommend.get_recommend_word_1(cv,jd,word2vec,tfidf_model,dictionary,stopword_list)
#method 2
recommend.get_recommend_word_2(cv,jd,word2vec,tfidf_model,dictionary,stopword_list)
#method 3
recommend.get_recommend_word_3(cv,jd,word2vec,tfidf_model,dictionary,stopword_list)
#method 4
recommend.get_recommend_sentence(cv,jd,word2vec,tfidf_model,dictionary,stopword_list)
#method 5
recommend.get_recommend_segment(cv,jd,word2vec,tfidf_model,dictionary,stopword_list)
```# snippet
# snippet
# snippet
# snippet_new
# snippet_new

Author: Wong Ching Kit
