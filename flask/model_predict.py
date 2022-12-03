#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from lime import lime_text
import nltk
import joblib


# In[4]:


model = tf.keras.models.load_model('../models/lstm_w2v.h5')
stopwords_list = stopwords.words('english')
with open('../models/lstm_tokenizer.pkl', 'rb') as f:
    lstm_tokenizer = joblib.load(f)
explainer = lime_text.LimeTextExplainer()


# In[13]:


def remove_stopwords(input_list):
    output_list = []
    for word in input_list:
        if word not in stopwords_list:
            output_list.append(word)
    return output_list

def clean_input_str(input_str):
    
    #print(input_str)
    
    #lowercase
    text = input_str.lower()
    
    #remove special chars
    pattern = r"(?u)\b\w\w+\b"
    tokenizer = RegexpTokenizer(pattern)
    text = tokenizer.tokenize(text)
    
    #remove stopwords
    text = remove_stopwords(text)
    
    
    #lemma
    lemma = nltk.stem.wordnet.WordNetLemmatizer()
    text = [lemma.lemmatize(word) for word in text]
    
    #back to string
    text_str = ' '.join(text)
    #print(text_str)
    return text_str
    
def lime_predict_lstm(text):

    final_output = []
    predict_input = []
    for text_variant in text:
        
        text_variant = clean_input_str(text_variant)

        #tokenize
        input_list_tokenized = [text_variant.split(' ')]
        #print(input_list_tokenized)

        #sequence
        input_sequence = lstm_tokenizer.texts_to_sequences(input_list_tokenized)

        #padding
        input_seq_padded = tf.keras.preprocessing.sequence.pad_sequences(input_sequence,
                                                                            maxlen=15, 
                                                                            padding='post', 
                                                                            truncating='post')

        predict_input.append(input_seq_padded[0].tolist())
    
    
    predict_output = model.predict(predict_input)
    
    return predict_output

def predict_func_final(input_text):
    #clean_text = clean_input_str(input_text)
    predictions = lime_predict_lstm(input_text)
    return predictions

def get_pred_exp(text, samples=5000):
    explanation = explainer.explain_instance(text, predict_func_final, 
                                             num_samples=samples, 
                                             labels=(1,))
    explanation.show_in_notebook()
    html_out = explanation.as_html()
    return str(html_out)
    




