# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import pandas as pd
import numpy as np
 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import BaggingClassifier
import string
import unicodedata
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from .constants import *
from .meta_vectorizer import *

import inflect
p = inflect.engine()

#from nltk.sentiment.vader import SentimentIntensityAnalyzer
 
def bin_(x):
    """ Binarize the target to predict """
    return int(x>=3)


def strip_accents_unicode(s):
    """ Deals with string encoding issues """
    try:
        s = unicode(s, 'utf-8')
    except:  # unicode is a default on python 3
        pass
    s = unicodedata.normalize('NFD', s)
    s = s.encode('ascii', 'ignore')
    s = s.decode("utf-8")
    return str(s)

def to_singular(word, p):
    if (nltk.pos_tag([word])[0][1] == 'PRP') or (word == 'his'): return word
    x = p.singular_noun(word)
    if x==False: return word
    return x
   
def process(s):
    """ Simple text processing """
    punctuation = set(string.punctuation)
    punctuation.update(["``", "`", "..."])
    def clean_str(sentence):
        verbs_stemmer = WordNetLemmatizer()
        return list((filter(lambda x: x.lower() not in punctuation and x.lower() not in english_stopwords,
                    [to_singular(verbs_stemmer.lemmatize(t.lower(), 'v'), p) for t in word_tokenize(sentence)
                     if t.isalpha()]))) 
    s_ = strip_accents_unicode(s)
    s_ = s_.replace("I'm ", "I am ")
    s_ = s_.replace("It's ", "It is ")
    s_ = s_.replace("it's ", "it is ")
    s_ = s_.replace("he's ", "he is ")
    s_ = s_.replace("she's ", "she is ")
    s_ = s_.replace("He's ", "He is ")
    s_ = s_.replace("She's ", "She is ")
    s_ = s_.replace("'ll ", " will ")
    s_ = s_.replace("can't", "can not")
    s_ = s_.replace("won't", "will not")
    s_ = s_.replace("n't ", " not ")
    s_ = s_.replace("'re ", " are ")
    s_ = s_.replace("-", ' ')
 
    return " ".join(clean_str(s_))


def count_numbers(s):
    """ Counts all numbers that occur in the statement, excepting dates """
    try:
        return len([e for e in re.findall(r'[0-9]*[.,]*[0-9]+', s) if ((len(e)!=4) or (len(e)==4 and int(e[0]) not in [1,2]))])
    except:
        return 0

def count_the(s):
    """ Counts undedfined articles occurences in the statement """
    return len([e for e in word_tokenize(s) if e.lower() in ['the', 'a', 'an']])
 
class FeatureExtractor():
    """ Extract features : TFIDF on words and TFIDF on POS-taggs are embedded with a Bagged Logistic Regression
        and mixed with meta features and a few other variables made from the statements.
    """
 
    def __init__(self):
        self.clf_bagged = BaggingClassifier(LogisticRegression(C= 5.), n_estimators = 300, max_features=0.8, bootstrap_features =True) ## 600
        self.vectorizer_text = TfidfVectorizer(ngram_range= (1,2), min_df=40) #50
        self.vectorizer_pos = TfidfVectorizer(ngram_range=(1,2), min_df=15) #20
        self.vectorizer_meta = MetaVectorizer(dict_parameters = BAYESIAN_PARAMETERS)
        pass
    
    def fit(self, X_df, y):
        self.vectorizer_meta.fit(X_df, y)
        
        tfidf_text = self.vectorizer_text.fit_transform(X_df.statement.apply(lambda s: process(s)))
        tfidf_pos = self.vectorizer_pos.fit_transform(X_df.statement.apply(lambda s: " ".join(list(zip(*nltk.pos_tag(word_tokenize(strip_accents_unicode(s)))))[1])))
        self.clf_bagged.fit(np.concatenate([tfidf_text.toarray(), tfidf_pos.toarray()], axis = 1), (y>2).astype(int))
        
        return self
 
    def fit_transform(self, X_df, y):
        self.fit(X_df, y)
        return self.transform(X_df)
 
    def transform(self, X_df):
        tfidf_text = self.vectorizer_text.transform(X_df.statement.apply(lambda s: process(s)))
        tfidf_pos = self.vectorizer_pos.transform(X_df.statement.apply(lambda s: " ".join(list(zip(*nltk.pos_tag(word_tokenize(strip_accents_unicode(s)))))[1])))
        dense_tfidf = self.clf_bagged.predict_proba(np.concatenate([tfidf_text.toarray(), tfidf_pos.toarray()], axis=1)) #[:,1].reshape(-1,1)

        df_train_meta = self.vectorizer_meta.transform(X_df)
        df_train_meta['n_fig']  = X_df.statement.apply(lambda s : count_numbers(strip_accents_unicode(s)))
        df_train_meta['n_fig'] = (df_train_meta['n_fig'] - df_train_meta['n_fig'].min()) / (df_train_meta['n_fig'].max() - df_train_meta['n_fig'].min())
        
        df_train_meta['n_the'] = X_df.statement.apply(lambda s : count_the(strip_accents_unicode(s)))
        df_train_meta['n_the'] = (df_train_meta['n_the'] - df_train_meta['n_the'].min()) / (df_train_meta['n_the'].max() - df_train_meta['n_the'].min())
 
        for f in usefull_sources:
            df_train_meta['is_'+'_'.join(f.lower().split())] = X_df.source.apply(lambda s : int(strip_accents_unicode(s) == f))
        
        #clf = SentimentIntensityAnalyzer()
        #df_train_meta['polarity'] = X_df.statement.apply(lambda s : np.floor(np.abs(clf.polarity_scores(s)['compound'])/0.2)*0.2)
        
        return np.concatenate([dense_tfidf, df_train_meta], axis=1)
