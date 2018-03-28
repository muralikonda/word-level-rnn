import numpy as np
import re
from keras.models import Sequential
from keras.layers.core import Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.optimizers import RMSprop
import random
import sys
import io
import urllib
import os
print os.getcwd()
def readInput(file,wordFlag = False):
    text = open(file, 'r').read()
    print "Number of Charecters:", len(text)
    if wordFlag:
        words = re.split('(\W)',text)
        words = filter(None, words)
    else:
        words = text
    print "Number of Words:", len(words)
    print "Number of Unique Words:", len(set(words))
    wd_ind = {w: i for i, w in enumerate(sorted(words))}
    ind_wd = {i: w for i, w in enumerate(sorted(words))}
    return words,wd_ind,ind_wd

def get_model(words):
    maxLength = 40
    model = Sequential()
    model.add(LSTM(256, input_shape = (maxLength,len(words)),return_sequences=True ))
    model.add(LSTM(256))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) #/ temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = words[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(20):
            x_pred = np.zeros((1, maxlen, len(words)))
            for t, word in enumerate(sentence):
                x_pred[0, t, ind_wd[word]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = ind_wd[next_index]

            generated += next_word
            sentence = sentence[1:] + next_word

            sys.stdout.write(next_word)
            sys.stdout.flush()
        print()
file = "./Data/hp.txt"
words,wd_ind,ind_wd = readInput(file,True)
maxlen = 40
step = 3
sentences = []
next_word = []
sentiment = []
for i in range(0, len(words) - maxlen, step):
    sentences.append(words[i: i + maxlen])
    next_word.append(words[i + maxlen])
print('Number of Inputs:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)
y = np.zeros((len(sentences), len(words)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        x[i, t, wd_ind[word]] = 1
    y[i, wd_ind[next_word[i]]] = 1


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
model = get_model(words)
model.fit(x, y, batch_size=128, epochs=60, callbacks=[print_callback])