#!/usr/bin/env python
import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('trainfile')
    parser.add_argument('testfile')
    parser.add_argument('layers', type=int)
    parser.add_argument('width', type=int)
    parser.add_argument('epochs', type=int)
    return parser.parse_args()

def get_data(file):

    raw_data = pandas.read_csv(file, header=None, sep='\t').values
    x_columns = len(raw_data[0]) - 1
    x = raw_data[:, 0:x_columns]
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
model.add(Dense(args.width, activation='relu', input_dim=36))
for i in range(args.layers):
    model.add(Dense(args.width, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=3, epochs=args.epochs)

score = model.evaluate(test_x, test_y, batch_size=3)

print("PARAMS: layers: " + str(args.layers) + ", width: " +
      str(args.width) + ", epochs: " + str(args.epochs))
print("SCORE: " + str(score))