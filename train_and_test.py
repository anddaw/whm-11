#!/usr/bin/env python
import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas
import argparse
import math
import numpy
from sklearn.preprocessing import StandardScaler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('trainfile')
    parser.add_argument('testfile')
    parser.add_argument('layers', type=int)
    parser.add_argument('width', type=int)
    parser.add_argument('epochs', type=int)
    parser.add_argument("--encoding", type=str, default="one_hot")
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


def int_to_vector_of_bits(integer, vector_length):

    return [int(i) for i in format(integer, "0" + str(vector_length) + "b")]


def encode_to_moc(data, labels):
    vector_length = math.ceil(math.log(len(labels), 2.0))

    return numpy.array([int_to_vector_of_bits(labels[x], vector_length) for x in data])


def normalize_moc(moc_vector):
    threshold = 0.5
    return list(map(lambda x: 1 if x >= threshold else 0, moc_vector))

def test_model(model, test_x_batch, test_y_batch, encoding):

    success = 0

    predictions = model.predict(test_x_batch)

    if encoding == "one_hot":
        for i in range(len(predictions)):
            if predictions[i].argmax() == test_y_batch[i].argmax():
                success += 1
    elif encoding == "moc":
        for i in range(len(predictions)):
            if normalize_moc(predictions[i]) == normalize_moc(test_y_batch[i]):
                success += 1

    accuracy = float(success)/len(predictions)

    return accuracy


args = parse_args()

encoding = args.encoding

scaler = StandardScaler()

train_x, train_y = get_data(args.trainfile)
labels = generate_labels(train_y)

train_x = scaler.fit_transform(train_x)

test_x, test_y = get_data(args.testfile)

test_x = scaler.transform(test_x)

output_layer_activation = ""
loss_function = ""

if encoding == "one_hot":
    train_y = encode_to_one_hot(train_y, labels)
    test_y = encode_to_one_hot(test_y, labels)
    output_layer_activation = "softmax"
    loss_function = "categorical_crossentropy"
elif encoding == "moc":
    test_y = encode_to_moc(test_y, labels)
    train_y = encode_to_moc(train_y, labels)
    output_layer_activation = "sigmoid"
    loss_function = "binary_crossentropy"

model = Sequential()

model.add(Dense(36, activation='linear', input_dim=36))
for i in range(args.layers):
    model.add(Dense(args.width, activation='tanh'))
model.add(Dense(len(train_y[0]), activation=output_layer_activation))

model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=10, epochs=args.epochs)

# score = model.evaluate(test_x, test_y, batch_size=3)

print("PARAMS: layers: " + str(args.layers) + ", width: " +
      str(args.width) + ", epochs: " + str(args.epochs))
# print("SCORE: " + str(score))

print("Training:" + str(test_model(model, train_x, train_y, encoding)))
print("Testing:" + str(test_model(model, test_x, test_y, encoding)))
