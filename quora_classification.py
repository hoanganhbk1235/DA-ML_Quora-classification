
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


def clean_text(x):

    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x


# In[ ]:


file1 = open("keyword/excess_word.txt","r+")   
key_ex = file1.read()
def remove_excess_word(text):
    str_text = text
    key_ex = key_ex.split('\n')
    key_ex = [clean_text(w) for w in key_ex]
    for word in text.split():
        if word in key_ex:
            str_text = str_text.replace(word, '')
    return str_text


# In[ ]:


# load model
model = load_model('model/softmax_bow_ver_02.h5')


# In[ ]:


with open('Bag_of_word/vectorizer_bow.pkl', 'rb') as f:
    vectorize_bow = pickle.load(f)


# In[ ]:


def predict_text(text):
    text = clean(text)
    text = remove_excess_word(text)
    vec_text = vectorize_bow.transform(text)
    y_prod = model.predict_prodba(vec_text)
    # choose with y_prod >= 0.23 => label = 1, y_prod < 0.23 => label = 0
    y_class = np.squezze(y_prod >= 0.23).astype('int')
    return y_class

