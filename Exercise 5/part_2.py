from tensorflow import keras
import pickle
import numpy as np

# Load data
print("Loading data")
with (open('Data/keras-data.pickle', 'rb')) as pickled_data:
    data = pickle.load(pickled_data)

x_train = np.array(data['x_train'])
y_train = np.array(data['y_train'])
x_test = np.array(data['x_test'])
y_test = np.array(data['y_test'])
vocab_size = data['vocab_size']
max_length = data['max_length']


# use list slice for testing purposes
# Preprocessing
print("Preprocessing data")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_length)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_length)

# Set up model

print("setting up model")
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 10, input_length=max_length))
model.add(keras.layers.LSTM(units=3))
model.add(keras.layers.Dense(units=1, activation='tanh'))

print("Compiling model")
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
print("Fitting model")
model.fit(x_train, y_train, epochs=1, verbose=0)

print("Evaluating model")
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy}")