from nltk.tokenize import WordPunctTokenizer
from gensim.models import Word2Vec
import pandas as pd
import os
import sys
import re

#
path = '/Users/franckthang/Work/PersonalWork/sentiment-analysis'
#yelp_path = 'yelp.csv'
#texts = pd.read_csv(os.path.join(path, yelp_path))['text'].values
#
#texts1 = pd.read_csv(os.path.join(path, 'labeledTrainData.tsv'), sep='\t')['review'].values
#texts2 = pd.read_csv(os.path.join(path, 'testData.tsv'), sep='\t')['review'].values
#

import numpy as np
from bs4 import BeautifulSoup

tok = WordPunctTokenizer()


def preprocessing(text):
  text = BeautifulSoup(x, 'lxml').get_text()
  text = re.sub(r'@[A-Za-z0-9]+','',text)
  text = text.decode("utf-8-sig")
  text = re.sub("[^a-zA-Z]", " ", text)
  text = re.sub('https?://[A-Za-z0-9./]+','',text)

  try:
    text = text.decode("utf-8-sig").replace(u"\ufffd", "?")
  except:
    pass

  words = tok.tokenize(lower_case)
  return (" ".join(words)).strip()


cols = ['sentiment','id','date','query_string','user','text']
df = pd.read_csv(os.path.join(path,"training.1600000.processed.noemoticon.csv"),header=None, names=cols, encoding='latin-1')
df.drop(['id','date','query_string','user'],axis=1,inplace=True)
# df['pre_clean_len'] = [len(t) for t in df.text]

df['text'] = [preprocessing(x) for x in df.text]
print(df['text'].head())
