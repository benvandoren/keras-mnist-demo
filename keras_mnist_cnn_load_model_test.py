import os
import numpy
import sys
from keras.models import model_from_json
import scipy.ndimage

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(img_fname):
  seed = 7
  numpy.random.seed(seed)

  # load json and create model
  json_file = open('keras_mnist_cnn.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights("keras_mnist_cnn.h5")
  print("Loaded model from disk")

  im = scipy.ndimage.imread(img_fname, flatten=True)
  data = im
  data = data.reshape(1, 1, 28, 28).astype('float32')
  # print(data[0][0])
  for i in range(28):
    for x in range(28):
      data[0][0][i][x] = 255 - data[0][0][i][x]

  data = data/255
  np_data = numpy.array(data)
  predict = loaded_model.predict(np_data)
  i = 0
  for x in predict[0]:
    if (round(x) > 0):
      print("")
      print("num is: %d" % (i))
      print("")
    i=i+1

fname = sys.argv[1]
main(fname)
