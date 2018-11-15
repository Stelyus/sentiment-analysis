import numpy as np
import pickle
import keras
import os
import sys
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from preprocessing_git import data_preprocessing
from sentiment_vectors import  wv_afin, wv_emoji_valence, wv_depech_mood, \
      wv_emoji_sentiment_lexicon, wv_opinion_lexicon_english, wv_emolex


EMBEDDING_DIM = 348

def get_max_len(tweets, tokenizer):
  return len(max([max(tokenizer.texts_to_sequences(tweet), key=len) for tweet in tweets], key=len))

# Specific for the csv
def prepare_data(corpora3, corpora7):
    tweet3, sentiment3 = data_preprocessing(corpora3, 'train')
    tweet7, sentiment7 = data_preprocessing(corpora7, 'test')

    all_tweet = tweet3.append(tweet7)

    tokenizer = keras.preprocessing.text.Tokenizer(filters=' ')
    tokenizer.fit_on_texts(all_tweet)
    word_index = tokenizer.word_index

    return word_index, tokenizer, tweet3, tweet7, sentiment3, sentiment7

def get_train_test(tweet, sentiment, max_len, tokenizer):
    sequences_train = tokenizer.texts_to_sequences(tweet)

    data_train = keras.preprocessing.sequence.pad_sequences(sequences_train, maxlen=max_len)
    indices_train = np.arange(data_train.shape[0])
    data_train = data_train[indices_train]

    labels_train = to_categorical(np.asarray(sentiment), 3)
    labels_train = labels_train[indices_train]

    split_idx = int(len(data_train) * 0.70)
    x_train, x_val = data_train[:split_idx], data_train[split_idx:]
    y_train, y_val = labels_train[:split_idx], labels_train[split_idx:]

    return x_train, x_val, y_train, y_val



def embedding_matrix_sentiment(word_index, w2vpath, sentiment):
  word2vec = pickle.load(open(w2vpath, 'rb'))
  embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
  oov = []
  oov.append((np.random.rand(EMBEDDING_DIM) * 2.0) - 1.0)
  oov = oov / np.linalg.norm(oov)

  path = "../resources/embeddings" 
  afin_path = os.path.join(path, 'afin')
  ev_path = os.path.join(path, 'ev')
  depech_path = os.path.join(path, 'depech')
  emolex_path= os.path.join(path, 'emolex')
  emoji_path = os.path.join(path, 'emoji_sentiment_lexicon')
  opi_path = os.path.join(path, 'opinion_lexicon')

  # Load sentiment vectors
  sentiment_wv_dict = {
    'afin': [pickle.load(open(afin_path, 'rb')), wv_afin],
    'ev': [pickle.load(open(ev_path, 'rb')), wv_emoji_valence],
    'depech': [pickle.load(open(depech_path,'rb')), wv_depech_mood],
    'emolex': [pickle.load(open(emolex_path, 'rb')), wv_emolex],
    'emoji': [pickle.load(open(emoji_path, 'rb')), wv_emoji_sentiment_lexicon],
    'opinion': [pickle.load(open(opi_path, 'rb')), wv_opinion_lexicon_english]
  }


  for word, i in word_index.items():
      if word in word2vec:
          embedding_matrix[i] = concatenate_word_vectors(word, word2vec, sentiment_wv_dict)
      else:
          embedding_matrix[i] = oov

  return embedding_matrix



def concatenate_word_vectors(word, word2vec, wv_sentiment_dict):
  concat = [word2vec[word]]
  for keys in wv_sentiment_dict:
    dictionary, fct  = wv_sentiment_dict[keys]
    concat.append(fct(dictionary, word))

  return np.concatenate(concat)

if __name__ == '__main__':
  corpora_train_3 = '../resources/data_train_3.csv'
  corpora_train_7 = '../resources/data_train_7.csv'
  corpora_test_7 = "'../resources/data_test_7.csv'"
  word2vec_path = '../resources/datastories.twitter.300d.pickle'

  word_index, tokenizer, tweet3, tweet7, sentiment3, sentiment7 = prepare_data(corpora_train_3, corpora_train_7)
  #model = Word2Vec.load('../resources/model_5M.bin')
  #saveKeyedVectors('../resources/model2.kv', model)
  
  MAX_SEQUENCE_LENGTH = get_max_len([tweet3, tweet7], tokenizer)

  embedding_matrix = embedding_matrix_sentiment(word_index, word2vec_path, EMBEDDING_DIM)

  x_train_3, x_val_3, y_train_3, y_val_3 = get_train_test(tweet3, sentiment3, MAX_SEQUENCE_LENGTH, tokenizer)
  #embedding_layer = Embedding(len(word_index) + 1,
                          #EMBEDDING_DIM,
                          #weights=[embedding_matrix],
                          #input_length=MAX_SEQUENCE_LENGTH,
                          #trainable=False, name='embedding_layer')
#
  #model1(x_train_3, y_train_3,x_val_3, y_val_3, embedding_layer)
#
