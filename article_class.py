# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 12:15:31 2022

This script is written to categorize articles according to their category.

@author: ANAS
"""

#%% Imports

import pandas as pd
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import os
from datetime import datetime
from module_for_article_class import ModelCreation
import pickle
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Input
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Bidirectional,Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#%% Statics
URL_PATH = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'

LOG_PATH = os.path.join(os.getcwd(), 'logs')

ENCODER_PATH = os.path.join(os.getcwd(), 'saved_models', 'oh_encoder.pkl')
TOKENIZER_JSON_PATH = os.path.join(os.getcwd(), 'saved_models',
                                   'tokenizer.json')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_models', 'model.h5')
PLOT_MODEL_PATH = os.path.join(os.getcwd(), 'static', 'model-architecture.png')

#%%EDA

#%% STEP 1) DATA LOADING

df=pd.read_csv(URL_PATH)
df_copy = df.copy()

#%% STEP 2) DATA INSPECTION

df.head(10)
df.tail(10)
df.info()
stats = df.describe().T

df['category'].unique() # to get the uniques targets # 5

df['text'][0]
df['category'][0]

df.duplicated().sum()

category_data = df['category']
text_data = df['text']

# Split the data into different variables

cat_names = list(df['category'].unique())
n_categories = len(cat_names)


#%% STEP 3) DATA CLEANING

df = df.drop_duplicates() # To remove duplicates

cleaned_text = text_data.replace(to_replace='[^a-zA-Z]', value=' ',
                                 regex=True).str.lower().str.split()

#Remove numbers, lowercase all and split the text

#%% STEP 4) FEATURES SELECTION

# NOTHING TO SELECT

#%% STEP 5) PREPROCESSING

vocab_size = 30000

# We choose the vocab size to be 30000


oov_token = 'OOV'



tokenizer = Tokenizer(num_words = vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(text_data)
word_index = tokenizer.word_index


token_json = tokenizer.to_json()

with open(TOKENIZER_JSON_PATH,'w') as file:
    json.dump(token_json,file)

print(word_index)

train_sequences = tokenizer.texts_to_sequences(text_data)

length_of_review = [len(i) for i in train_sequences]


print(np.mean(length_of_review))

max_len = 394

# Based on the max length of text



padded_review = pad_sequences(train_sequences,maxlen=max_len,
                              padding='post',
                              truncating='post')


ohe = OneHotEncoder(sparse = False)
category_data = ohe.fit_transform(np.expand_dims(category_data, axis=-1))

# We encode the targeted text file

with open(ENCODER_PATH,'wb') as file:
    pickle.dump(ohe,file)


X_train,X_test,y_train,y_test = train_test_split(padded_review,
                                                 category_data,
                                                 test_size=0.3,
                                                 random_state=123)

#%% Model development

X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)



model = Sequential()
model.add(Input(shape=(394)))
model.add(Embedding(vocab_size, 128))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.2))
model.add(Dense(n_categories, activation='softmax'))
model.summary()

# %%% Callbacks
log_dir = os.path.join(LOG_PATH, datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_cb = EarlyStopping(monitor='val_loss', patience=2)

model.compile(optimizer='adam',
              loss = 'categorical_crossentropy',
              metrics=['acc'])

hist = model.fit(X_train,y_train,
                 epochs=5,
                 validation_data=(X_test,y_test))

hist.history.keys()

plt.figure()
plt.plot(hist.history['loss'],label='Training Loss')
plt.plot(hist.history['val_loss'],label='Validation Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history['acc'],label='Training acc')
plt.plot(hist.history['val_acc'],label='Validation acc')
plt.legend()
plt.show()

#%% Model evaluation

mc = ModelCreation()
mc.model_evaluation(model,y_test, X_test)


plot_model(model)
model.save(MODEL_PATH)
