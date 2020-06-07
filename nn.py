import tensorflow as tf
import numpy as np

from data import create_features_and_labels, create_lexicon
from input import process_input

DATA_FILE = 'data/dataset.txt'

lexicon = create_lexicon(DATA_FILE)


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, input_shape=(len(lexicon),), activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    return model


def train_new(model_path):
    x_train, y_train, x_test, y_test = create_features_and_labels(DATA_FILE, test_factor=0.1, lexicon=lexicon)

    model = create_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=100, epochs=5)
    tf.keras.models.save_model(model, model_path, overwrite=False)

    return model


def train(model, model_path):
    x_train, y_train, x_test, y_test = create_features_and_labels(DATA_FILE, test_factor=0.1, lexicon=lexicon)
    model.fit(x_train, y_train, batch_size=100, epochs=5)
    tf.keras.models.save_model(model, model_path, overwrite=False)
    return model


def load_model(model_path):
    model = tf.keras.models.load_model(model_path, compile=True)
    return model


def determine_sentiment(model, lexicon):
    sentence = input('Input a statement: ')
    predictions = model.predict(process_input(sentence, lexicon))
    if np.argmax(predictions[0]) == 0:
        print('Your statement is positive! :)\n')
    else:
        print('Your statement is negative :(\n')


model = load_model('tmp/model.hdf5')

while 1:
    determine_sentiment(model, lexicon)


