from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf
import numpy as np
import keras
from keras.utils import pad_sequences


# Creating the model, then training and saving it. Will be used only once.
def create_model():
    # Setting sizes
    VOCAB_SIZE = 88584
    MAXLEN = 250
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

    # Padding with zeros the text so it will have the same input size to the RNN.
    train_data = pad_sequences(train_data, MAXLEN)
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 32),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # Training the model
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])
    model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

    # Saving the model
    model.save('sense_model')


def use_model(text):
    # Encoding input text for our own use with the dataset word index
    word_index = imdb.get_word_index()

    # Loading model
    model = keras.models.load_model("sense_model")
    encoded = encode_text(text, word_index)
    pred = np.zeros((1,250))
    pred[0] = encoded
    result = model.predict(pred)*100
    value = result[0][0]
    if value > 50:
        print("The text you put is {} positive.".format(value))
    else:
        print("The text you put is {} negative.".format(100 - value))


# Helper function for using the trained the model
def encode_text(text, word_index, MAXLEN=250):
    tokens = keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return pad_sequences([tokens], MAXLEN)[0]
