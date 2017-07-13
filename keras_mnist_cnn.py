import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

def main():
  seed = 7
  numpy.random.seed(seed)

  # load data
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
  # reshape to be [samples][pixels][width][height]
  X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
  X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

  # normalize inputs from 0-255 to 0-1
  X_train = X_train / 255
  X_test = X_test / 255
  # one hot encode outputs
  y_train = np_utils.to_categorical(y_train)
  y_test = np_utils.to_categorical(y_test)
  num_classes = y_test.shape[1]

  # build the model
  model = larger_model(num_classes)
  # Fit the model
  model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=4, batch_size=200, verbose=2)
  # Final evaluation of the model
  scores = model.evaluate(X_test, y_test, verbose=0)
  print("Baseline Error: %.2f%%" % (100-scores[1]*100))

  # serialize model to JSON
  model_json = model.to_json()
  with open("keras_mnist_cnn.json", "w") as json_file:
      json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("keras_mnist_cnn.h5")
  print("Saved model to disk")

  return 0

def baseline_model(num_classes):
  # create model
  model = Sequential()
  model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dense(num_classes, activation='softmax'))
  # Compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

def larger_model(num_classes):
  # create model
  model = Sequential()
  model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Convolution2D(15, 3, 3, activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(num_classes, activation='softmax'))
  # Compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

main()
