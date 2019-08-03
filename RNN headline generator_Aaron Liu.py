# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 15:23:40 2019

@author: aaronliu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, load_model
from keras.utils import to_categorical
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import LSTM
from pickle import dump,load
import random
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
import datetime
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load and Explore Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
data = pd.read_csv('C:\\Users\\user\\Desktop\\W&M BA Fall\\Course\\Spring Semester\\AI\\assignment\\RNN headline generator\\abcnews-date-text.csv')
data.shape
data.head()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def clean_doc(doc):
	# replace '--' with a space ' '
	doc = doc.replace('--', ' ')
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# make lower case
	tokens = [word.lower() for word in tokens]
	return tokens 

def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
    
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

### using 0.5% of all the dataset because of memory error
random.seed(11)
idx = [n for n in range(data.shape[0])]
np.random.shuffle(idx) 
data = data.iloc[idx[0:int(len(idx)*0.005)],1]

data_token=data.apply(clean_doc)
del data
word_counts=data_token.apply(len)
data_used=pd.DataFrame({'headlines':data_token,'counts':word_counts})
del data_token
# drop lines with 1 word
data_used.groupby('counts').count()
data_X = data_used[(data_used.counts==8)]# | (data_used.counts==8)]
data_X.groupby('counts').count()

# encoding and padding
t = Tokenizer()
t.fit_on_texts(data_X.headlines)
# vocabulary size
vocab_size = len(t.word_index) + 1

# integer encode the documents
encoded_docs = t.texts_to_sequences(data_X.headlines)

max_length = 9
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='pre')

# get X and y
n = padded_docs.shape[0]
X =  padded_docs[0:n,:-1]
y = padded_docs[0:n,-1]
y = to_categorical(y, num_classes= vocab_size)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define and Train Model Section (for all experiments)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# count the time of execution
start_time = datetime.datetime.now()
# define model
def AaronModel_original():
    seq_length = 8#X.shape[1]
    num_input_words = 100
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(num_input_words, return_sequences=True))
    model.add(LSTM(num_input_words))
    model.add(Dense(num_input_words, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy'
                  , optimizer='adam'
                  , metrics=['accuracy'])
    return model 

model = AaronModel_original()
model.summary()
###########################################################################
# define model
def AaronModel_HunitDouble():
    seq_length = X.shape[1]
    num_input_words = 200
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(num_input_words, return_sequences=True))
    model.add(LSTM(num_input_words))
    model.add(Dense(num_input_words, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy'
                  , optimizer='adam'
                  , metrics=['accuracy'])
    return model 

model = AaronModel_HunitDouble()
model.summary()
###########################################################################
# define model
def AaronModel_HunitHalve():
    seq_length = X.shape[1]
    num_input_words = 50
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(num_input_words, return_sequences=True))
    model.add(LSTM(num_input_words))
    model.add(Dense(num_input_words, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy'
                  , optimizer='adam'
                  , metrics=['accuracy'])
    return model 

model = AaronModel_HunitHalve()
model.summary()
###########################################################################
# define model
def AaronModel_HiddenLayer():
    seq_length = X.shape[1]
    num_input_words = 100
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(num_input_words, return_sequences=True))
    model.add(LSTM(num_input_words))
    model.add(Dense(num_input_words, activation='relu'))
    model.add(Dense(num_input_words, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy'
                  , optimizer='adam'
                  , metrics=['accuracy'])
    return model 

model = AaronModel_HiddenLayer()
model.summary()
###########################################################################
# define model
def AaronModel_LengthDouble():
    seq_length = 16#X.shape[1]
    num_input_words = 100
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(num_input_words, return_sequences=True))
    model.add(LSTM(num_input_words))
    model.add(Dense(num_input_words, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy'
                  , optimizer='adam'
                  , metrics=['accuracy'])
    return model 

model = AaronModel_LengthDouble()
model.summary()
###########################################################################
# define model
def AaronModel_LengthHalve():
    seq_length = 4#X.shape[1]
    num_input_words = 100
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(num_input_words, return_sequences=True))
    model.add(LSTM(num_input_words))
    model.add(Dense(num_input_words, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy'
                  , optimizer='adam'
                  , metrics=['accuracy'])
    return model 

model = AaronModel_LengthHalve()
model.summary()
###########################################################################

# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
history = model.fit(X, y, batch_size=128, epochs=300 , callbacks=callbacks_list)
# count the time required for training model
stop_time = datetime.datetime.now()
print ("Time required for training model:",stop_time - start_time)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# save the model to file
model.save('model_headline_original')
# save the tokenizer
dump(t, open('tokenizer_headline_original.pkl', 'wb'))

# load the model
model = load_model('weights-improvement-100-0.5137.hdf5')
# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))

# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = t.texts_to_sequences([in_text])[0]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text.append(out_word)
		result.append(out_word)
	return ' '.join(result)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Generate Headlines Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# select a seed text
data_seed = data_X.headlines.reset_index()
seed_text = data_seed.headlines[np.random.randint(0,len(data_seed.headlines))]
# generate new text
num_words_to_generate = 7
generated = generate_seq(model, t, max_length-1, seed_text, num_words_to_generate)
print(generated)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plot Training Loss Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#plot train loss in related to train iterations(epochs) 
acc = history.history['acc']
loss = history.history['loss']

epochs = range(1,len(acc)+1)

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training loss')
plt.legend()
plt.show()



loss = history.history['loss'][19]

