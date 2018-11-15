import os
import sys
import re
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split


path = '/Users/franckthang/Work/PersonalWork/sentiment-analysis'
yelp_path = 'yelp.csv'

df = pd.read_csv(os.path.join(path, yelp_path))

df = df.assign(text=lambda x: [re.sub('[^a-zA-z0-9\s]','',x) for x in df.text])
df = df.assign(sentiment=lambda x: ["neg" if x < 3 else "pos" for x in x.stars])
for idx, row in df.iterrows():
  row[0] = row[0].replace('rt', ' ') 

texts = df['text'].values


tokenizer = Tokenizer(num_words=2500, lower=True, split=" ")
tokenizer.fit_on_texts(texts)

X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X)

embed_dim = 128
lstm_out = 200
batch_size = 32

model = Sequential()
model.add(Embedding(2500, embed_dim, input_length = X.shape[1], dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

Y = pd.get_dummies(df['sentiment']).values
X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size = 0.20, random_state = 36)

model.fit(X_train, Y_train, batch_size =batch_size, nb_epoch = 1,  verbose = 5)

