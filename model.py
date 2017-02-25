#!/usr/bin/env python

import cv2
import numpy as np
import json
from random import randint
import traceback

from keras.models import Sequential
from keras.layers import Dense, Lambda
from keras.layers import Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import ELU, LeakyReLU
from keras.regularizers import l2

from data_parser import DataParser



class BehaviorCloner:
  """Create Keras model to train learn from driving images in simulator and 
     learn to control the car on it's own"""

  def __init__(self):
    self._data_parser = DataParser()

  def _flip_image(self, img_):
    return cv2.flip(img_, 1)

  def _combine_LCR(self, labels_, epoch_):
    left_imgs = self._data_parser.left_imgs
    center_imgs = self._data_parser.center_imgs
    right_imgs = self._data_parser.right_imgs

    angle_adjust = 0.1 
    left_labels = np.copy(labels_) + angle_adjust
    center_labels = np.copy(labels_)
    right_labels = np.copy(labels_) - angle_adjust

    batch_size = left_imgs.shape[0]
    row_size = left_imgs.shape[1]
    col_size = left_imgs.shape[2]
    total_imgs = np.zeros((batch_size, row_size, col_size, 3))
    total_labels = np.zeros(batch_size)
    for pic_num in range(total_imgs.shape[0]):
      while 1:
        # get index
        index = randint(0,total_imgs.shape[0]-1)
        # pick different images
        lrc_rand = randint(0,100)
        if lrc_rand > 66:
          img   = right_imgs[index]
          label = right_labels[index]
        elif lrc_rand > 33:
          img   = center_imgs[index]
          label = center_labels[index]
        else:
          img   = left_imgs[index]
          label = left_labels[index]

        # probability says we shouldn't keep it
        if (abs(label)*100 + epoch_*10) < randint(0,100):
          continue

        # flip images
        flip_rand = randint(0,100)
        if flip_rand > 50:
          img = self._flip_image(img)
          label *= -1

        # add and go to next in for loop
        total_imgs[pic_num] = img
        total_labels[pic_num] = label
        break

    return total_imgs, total_labels


  def _generator_training(self, labels_, batch_size_, xDiv_, yDiv_):
      def _f():
          epoch = 0
          max_epoch = 5
          start = 0
          end = start + batch_size_
          num_imgs = labels_.shape[0]
  
          while True:
              self._data_parser.combine_batch(start, end, xDiv_, yDiv_) #setup data
              X_batch, y_batch = self._combine_LCR(labels_[start:end], epoch)  #get data
              start += batch_size_
              end += batch_size_
              if start >= num_imgs:
                start = 0
                end = batch_size_
                epoch += 1
                if epoch >= max_epoch:
                  epoch = 0
              if end >= num_imgs:
                end = num_imgs
  
              yield (X_batch, y_batch)
  
      return _f

  def _generator_validation(self, labels_, batch_size_, xDiv_, yDiv_):
      def _f():
          start = 0
          end = start + batch_size_
          num_imgs = labels_.shape[0]
  
          while True:
              self._data_parser.combine_batch(start, end, xDiv_, yDiv_) #setup data
              X_batch = self._data_parser.center_imgs
              y_batch = labels_[start:end]
              start += batch_size_
              end += batch_size_
              if start >= num_imgs:
                start = 0
                end = batch_size_
              if end >= num_imgs:
                end = num_imgs
  
              yield (X_batch, y_batch)
  
      return _f

  '''
  External API
  '''
  def setup_data(self):
    self._data_parser.parse_data()

  # Build model based on
  # Nvidia "End to End Learning for Self-Driving Cars"
  def build_model(self, xDiv_, yDiv_):

    input_height = int(self._data_parser.img_height/yDiv_)
    input_width = int(self._data_parser.img_width/xDiv_)
    input_channels = self._data_parser.img_channels

    self._model = Sequential()

    # normalize -1<>+1
    self._model.add(Lambda(lambda x: x/127.5 - 1.,
              input_shape=(input_height, input_width, input_channels),
              output_shape=(input_height, input_width, input_channels)))


    # Conv Layer #0 (depth=3, kernel=1x1) - change color space
    self._model.add(Convolution2D(3, 1, 1, border_mode='same'))

    # Conv Layer #1 (depth=24, kernel=5x5)
    self._model.add(Convolution2D(24, 5, 5, border_mode='valid'))
    self._model.add(ELU())
    self._model.add(MaxPooling2D(pool_size=(2,2)))
    self._model.add(Dropout(0.5))

    # Conv Layer #2 (depth=36, kernel=5x5)
    self._model.add(Convolution2D(36, 5, 5, border_mode='valid'))
    self._model.add(ELU())
    self._model.add(MaxPooling2D(pool_size=(2,2)))
    self._model.add(Dropout(0.5))

    # Conv Layer #3 (depth=48, kernel=3x3)
    self._model.add(Convolution2D(48, 3, 3, border_mode='valid'))
    self._model.add(ELU())
    self._model.add(MaxPooling2D(pool_size=(2,2)))
    self._model.add(Dropout(0.5))

    # Conv Layer #4 (depth=64, kernel=3x3)
    self._model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    self._model.add(ELU())
    self._model.add(MaxPooling2D(pool_size=(2,2)))
    self._model.add(Dropout(0.5))

    self._model.add(Flatten())

    # Hidden Layer #1
    self._model.add(Dense(100))
    self._model.add(ELU())

    # Hidden Layer #2
    self._model.add(Dense(50))
    self._model.add(ELU())

    # Hidden Layer #3
    self._model.add(Dense(10))
    self._model.add(ELU())

    # Answer
    self._model.add(Dense(1))
    self._model.summary()



  def train_model(self, num_epochs_, batch_size_, xDiv_, yDiv_):
    print('BehaviorCloner: train_model()...')

    # setup for training
    self._model.compile(optimizer="adam", loss="mse")

    # train the model
    train_gen = self._generator_training(self._data_parser.steering_angles,
                                         batch_size_, xDiv_, yDiv_)
    num_imgs_train = self._data_parser.steering_angles.shape[0]*3   #3x for left, center, right
    history = self._model.fit_generator(train_gen(), num_imgs_train, num_epochs_)

    # validation
    validation_gen = self._generator_validation(self._data_parser.steering_angles,
                                                batch_size_, xDiv_, yDiv_)
    num_imgs_validate = self._data_parser.steering_angles.shape[0]   #1x center
    accuracy = self._model.evaluate_generator(validation_gen(), num_imgs_validate)
    print("Accuracy = ", accuracy)

    print('... train_model() done')


  def save_model(self):
    model_json = self._model.to_json()
    with open('model.json', 'w') as outfile:
      json.dump(model_json, outfile)

    self._model.save_weights('model.h5')


if __name__ == '__main__':
  print('Running main in model.py')

  try:
    behavior_cloner = BehaviorCloner()
    behavior_cloner.setup_data()

    x_down_sample = 5
    y_down_sample = 2.5
    behavior_cloner.build_model(x_down_sample, y_down_sample)

    test_num_epochs = 5
    test_batch_size = 16
    behavior_cloner.train_model(test_num_epochs, test_batch_size, 
                                x_down_sample, y_down_sample)

    behavior_cloner.save_model()

    print('... main done')
  except:
    print(traceback.format_exc())

