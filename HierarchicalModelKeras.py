from __future__ import print_function, division
from builtins import range

import os
import sys
import re
import codecs
import random
import json
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from keras_wc_embd import get_word_list_eng, get_dicts_generator, get_batch_input, get_embedding_layer


#----------------------#
#---  Global Value  ---#
#----------------------#

#--- hyperparameters
MAX_SENTENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 5


#--- load pretrained word2vec model
PATH = "~/BillZhan"
MODEL_FILE = PATH+"PretrainedWordVec/PubMed-shuffle-win-30.txt"
TRAIN_FILE = PATH+"/PubMed_200k_RCT/train.csv"
DEV_FILE = PATH+"/PubMed_200k_RCT/dev.csv"
TEST_FILE = PATH+"/PubMed_200k_RCT/test.csv"
SAVE_MODELNAME = 'BiLSTM_Model.h5'
WORD_DICT = 'word_dict.json'
CHAR_DICT = 'char_dict.json'
WORD_LEN = 'max_wlen.txt'

# prepare text samples and their labels
print('Loading in Abstracts...')

#--- load data
train = pd.read_csv(TRAIN_FILE)
dev = pd.read_csv(DEV_FILE)
test = pd.read_csv(TEST_FILE)

train_num = len(train)
val_num = len(dev)
train_steps = train_num // BATCH_SIZE
val_steps = val_num // BATCH_SIZE

print('Train: %d  Validate: %d' % (train_num, val_num))

# X
sentences_train = train['SENTENCE'].fillna("DUMMY_VALUE").values  #numpy array
sentences_dev = dev['SENTENCE'].fillna("DUMMY_VALUE").values  #numpy array
sentences_test = test['SENTENCE'].fillna("DUMMY_VALUE").values  #numpy array
# Y
le = LabelEncoder()
targets_train = train['LABEL'].values
targets_dev = dev['LABEL'].values
targets_test = test['LABEL'].values
encoded_labels_train, encoded_labels_dev, encoded_labels_test = le.fit_transform(targets_train), le.fit_transform(targets_dev), le.fit_transform(targets_test)
# one-hot encoding for class labels
onehot_train, onehot_dev, onehot_test = to_categorical(encoded_labels_train),to_categorical(encoded_labels_dev),to_categorical(encoded_labels_test)

print("max sequence length:", max(len(s) for s in sentences_train))
print("min sequence length:", min(len(s) for s in sentences_train))
s = sorted(len(s) for s in sentences_train)
print("median sequence length:", s[len(s) // 2])
del train

#--- Generate dictionaries for words and characters
dicts_generator = get_dicts_generator(
    word_min_freq=5,
    char_min_freq=2,
    word_ignore_case=True,
    char_ignore_case=False,
)
for sentence in sentences_train:
    dicts_generator(get_word_list_eng(sentence))
word_dict, char_dict, max_word_len = dicts_generator(return_dict=True)  #dict object here are word2index dict(or char2index), gives index of word(char) in the vocabulary 
print('Word dict size: %d  Char dict size: %d  Max word len: %d' % (len(word_dict), len(char_dict), max_word_len))

#--- Write word and char dict to json files
with open(WORD_DICT,'a') as output_wd:
    json.dump(word_dict,output_wd,ensure_ascii=False)
    output_wd.write('\n')
with open(CHAR_DICT,'a') as output_cd:
    json.dump(char_dict,output_cd,ensure_ascii=False)
    output_cd.write('\n')
with open(WORD_LEN,'w') as f:
    f.write(str(max_word_len))

#--- Embedding Layer
def get_embedding_weights_from_file(word_dict, file_path, ignore_case=True):
    """Load pre-trained embeddings from a text file.
    Each line in the file should look like this:
        word feature_dim_1 feature_dim_2 ... feature_dim_n
    The `feature_dim_i` should be a floating point number.
    :param word_dict: A dict that maps words to indice.
    :param file_path: The location of the text file containing the pre-trained embeddings.
    :param ignore_case: Whether ignoring the case of the words.
    :return weights: A numpy array.
    """
    pre_trained = {}
    with codecs.open(file_path, 'r', 'utf8') as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts)<200:  #skip first line
                continue
            if True:
                parts[0] = parts[0].lower()
            for i in range(1,len(parts)):
                if re.search('\.\d{7,}',parts[i]):  #for some lines, the formats are messy, the first several elements can be tokens, the remaining is vector
                                                    #to deal with this, find the first elements contains a '.' and 7 digits of numbers, should be the start of vector       
                    pre_trained[''.join(parts[:i])] = list(map(float, parts[i:]))
                    break
    assert len(pre_trained) > 0
    embd_dim = len(next(iter(pre_trained.values())))
    weights = [[0.0] * embd_dim for _ in range(max(word_dict.values()) + 1)]
    for word, index in word_dict.items():
        if not word:
            continue
        if ignore_case:
            word = word.lower()
        if word in pre_trained:
            weights[index] = pre_trained[word]
        else:
            weights[index] = np.random.random((embd_dim,)).tolist()
    return np.asarray(weights)


word_embd_weights = get_embedding_weights_from_file(word_dict, MODEL_FILE, ignore_case=True)  #return the word embedding matrix
inputs, embd_layer = get_embedding_layer(  #inputs: [word_input_layer, char_input_layer]
                                           #embd_layer: concatenate
    word_dict_len=len(word_dict),
    char_dict_len=len(char_dict),
    max_word_len=max_word_len,
    word_embd_dim=EMBEDDING_DIM,
    char_embd_dim=EMBEDDING_DIM,
    char_hidden_dim=25,
    word_embd_weights=word_embd_weights,
    rnn='lstm',
)

#--- Build Model
print('Building model...')
lstm_layer = Bidirectional(
    LSTM(units=100),
    name='Bi-LSTM',
)(embd_layer)
dense_layer = Dense(
    units=5,
    activation='softmax',
    name='Dense',
)(lstm_layer)
model = Model(inputs=inputs, outputs=dense_layer)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy'],
)
model.summary()
#--- Train model
def train_batch_generator(batch_size=32, training=True):
    while True:
        sentences = []
        if training:  #use training set
            data = sentences_train
            batch_ix = random.sample(range(len(data)), batch_size)
            for ix in batch_ix:
                text = data[ix]
                sentences.append(get_word_list_eng(text))
            onehot_labels = onehot_train[batch_ix,:]
        else:  #use validation set
            data = sentences_dev
            batch_ix = random.sample(range(len(data)), batch_size)
            for ix in batch_ix:
                text = data[ix]
                sentences.append(get_word_list_eng(text))
            onehot_labels = onehot_dev[batch_ix,:]

        word_input, char_input = get_batch_input(
            sentences=sentences,
            max_word_len=max_word_len,
            word_dict=word_dict,
            char_dict=char_dict,
            word_ignore_case=True,
            char_ignore_case=False,
        )
        yield [word_input, char_input], onehot_labels

#--- Fit Model
model.fit_generator(  #fit model batch-by-batch
    generator=train_batch_generator(batch_size=BATCH_SIZE, training=True),
    steps_per_epoch=train_steps,
    epochs=EPOCHS,
    validation_data=train_batch_generator(batch_size=BATCH_SIZE, training=False),
    validation_steps=val_steps,
    verbose=True,
)
model.save(SAVE_MODELNAME)

#--- Check on Test Set
test_num = len(test)

test_steps = test_num // BATCH_SIZE

def test_batch_generator(batch_size=32):
    index = 0
    while index < test_num:
        sentences = []
        batch_ix = range(index,min(index + batch_size, test_num))
        index += batch_size
        for ix in batch_ix:
            text = sentences_test[ix]
            sentences.append(get_word_list_eng(text))
        word_input, char_input = get_batch_input(
            sentences=sentences,
            max_word_len=max_word_len,
            word_dict=word_dict,
            char_dict=char_dict,
            word_ignore_case=True,
            char_ignore_case=False,
        )
        yield [word_input, char_input]

predicts = model.predict_generator(
    generator=test_batch_generator(BATCH_SIZE),
    steps=test_steps,
    verbose=True,
)
predicts = np.argmax(predicts, axis=-1).tolist()
actual_testsize = len(predicts)

test_acc = np.sum(np.equal(predicts,encoded_labels_test.tolist()[:actual_testsize]))/actual_testsize

print('Accuracy on test set is {acc}'.format(acc=test_acc))