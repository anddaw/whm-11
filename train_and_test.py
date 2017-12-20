#!/usr/bin/env python
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas
import argparse
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn import preprocessing

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('trainfile')
    parser.add_argument('testfile')
    return parser.parse_args()

def get_data(file):

    raw_data = pandas.read_csv(file, header=None, sep='\t').values
    x_columns = len(raw_data[0]) - 1
    x = preprocessing.StandardScaler().fit_transform(raw_data[:, 0:x_columns])
    y = raw_data[:, x_columns]

    return x, y

def generate_labels(data):
    values = set(data)
    return dict(zip(values, range(len(values))))

def encode_to_one_hot(data, labels):
    return keras.utils.to_categorical(list(map(lambda k: labels[k], data)), len(labels))
  
args = parse_args()

train_x, train_y = get_data(args.trainfile)
labels = generate_labels(train_y)
train_y = encode_to_one_hot(train_y, labels)

test_x, test_y = get_data(args.testfile)
test_y = encode_to_one_hot(test_y, labels)

model = Sequential()

model.add(Dense(36, activation='relu', input_dim=36))
model.add(Dropout(0.2))
model.add(Dense(36, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(25, activation='softmax'))

sgd = SGD(lr=0.02, decay=1e-6, momentum=0.0, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

early_stopping = EarlyStopping(verbose=2, patience=2)

model.fit(train_x, train_y,
          epochs=150,
          batch_size=3,
          callbacks=[early_stopping])
score = model.evaluate(test_x, test_y, batch_size=3)
print(score)
