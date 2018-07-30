# Generic classification functionality.
# Uses keras resnet archirecture.
#
# Copyright (C) 2017-2018  Author: Misha Orel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.


import os
import sys
import csv
import numpy as np
import dill
import random

from shared.pyutils.imageutils import *
from shared.pyutils.utilMultiProc import UtilMultithreadQueue

np.set_printoptions(threshold=np.nan)

import mlearn.tfHelpers.kerasResnet as resnet
import keras
from keras import optimizers
from keras import backend as K

ImageChannels = 3 # RGB images

TrainingBatchCount = 10


class ImageClassifKerasResnet:

  # Dictionaries map file name to a class id
  def __init__(self, trainFilesDict, validFilesDict, classCount, imgHeight, imgWidth, clrMeans, clrStds, augmenter,
               batchSize, bestModelName=None):
    # Calculates weights for each class in teh raining set
    counts = [0] * classCount
    for cl in trainFilesDict.values():
      counts[cl] += 1
    classWeights = []
    for count in counts:
      classWeights.append(1. / (count * classCount))

    self.classWeights_ = classWeights
    self.trainFilesDict_ = trainFilesDict
    self.trainFilesList_ = list(self.trainFilesDict_.keys())
    self.trainFilesWeights_ = [classWeights[trainFilesDict[fn]] \
      for fn in self.trainFilesList_] 
    self.validFilesDict_ = validFilesDict
    self.validFilesList_ = list(self.validFilesDict_.keys())
    self.classCount_ = classCount
    self.height_ = imgHeight
    self.width_ = imgWidth
    self.clrMeans_ = clrMeans
    self.clrStds_ = clrStds
    self.augmenter_ = augmenter
    self.batchSize_ = batchSize
    self.bestModelName_ = bestModelName
    self.bestAccuracy_ = 1.0

  def modelInit(self, modelName=None, learningRate=None, architectureName='build_resnet_18'):
    if modelName is None:
      # Creating model from scratch
      assert learningRate is not None
      architectureFunc = getattr(resnet.ResnetBuilder, architectureName)
      self.model_ = architectureFunc((ImageChannels, self.height_, self.width_), self.classCount_)
      sgd = optimizers.SGD(lr=learningRate, decay=0., momentum=0.9, nesterov=True)
      self.model_.compile(loss='categorical_crossentropy',
                    # optimizer='adam',
                    optimizer=sgd,
                    metrics=['accuracy'])
    else:
      # Loading model from file
      self.model_ = keras.models.load_model(modelName)
      if learningRate is None:
        learningRate = K.get_value(self.model_.optimizer.lr)

    self.learningRate_ = learningRate
    self.learningRateScale_ = None


  def modelSave(self, modelName):
    keras.models.save_model(self.model_, modelName)

  def summary(self):
    return self.model_.summary()

  def train(self, rounds):
    def _trainCback(state, lock):
      fileName = np.random.choice(self.trainFilesList_,
        p=self.trainFilesWeights_)
      img = UtilImageFileToArray(fileName)
      assert img.shape[:2] == (self.height_, self.width_)
      img = self.augmenter_.augment(img)
      img = (img - self.clrMeans_) / self.clrStds_
      return (img, self.trainFilesDict_[fileName])

    # Overwrite learning rate
    print('Setting learning rate %f' % self.learningRate_)
    K.set_value(self.model_.optimizer.lr, self.learningRate_)

    mthreadTraining = UtilMultithreadQueue(None, _trainCback,
      maxQueueSize=self.batchSize_ * TrainingBatchCount)
    for round in range(rounds):
      print('TRAINING ROUND %d' % round)
      images = []
      labels = []
      for _ in range(self.batchSize_ * TrainingBatchCount):
        img, label = mthreadTraining.getData()
        images.append(img)
        labels.append(label)

      images = np.stack(images, axis=0)

      # Make target 1-hot encoding
      target = np.zeros((self.batchSize_ * TrainingBatchCount, self.classCount_),
                        dtype=np.int)
      target[range(target.shape[0]), labels] = 1

      self.model_.fit(images, target,
                batch_size=self.batchSize_,
                epochs=1,
                initial_epoch=0,
                shuffle=False,
                verbose=2)

    mthreadTraining.terminate()

  def validate(self):
    def _valid_cback(state, lock):
      lock.acquire()
      ind = state[0]
      state[0] += 1
      lock.release()
      if ind >= len(self.validFilesList_):
        return None
      fileName = self.validFilesList_[ind]
      img = UtilImageFileToArray(fileName)
      img = (img - self.clrMeans_) / self.clrStds_
      return (img, fileName, self.validFilesDict_[fileName])

    validationTuples = []
    mthreadValidation = UtilMultithreadQueue([0], _valid_cback)
    for _ in range(len(self.validFilesList_)):
      t = mthreadValidation.getData()
      validationTuples.append(t)

    imageArray = np.stack([t[0] for t in validationTuples], axis=0)
    predictions = self.model_.predict(imageArray, batch_size=self.batchSize_, verbose=1)
    errorFileList = []
    for ind, pr in enumerate(predictions):
      a = np.argmax(pr)
      if a != validationTuples[ind][2]:
        errorFileList.append(validationTuples[ind][1])

    mthreadValidation.terminate()

    accuracy = len(errorFileList) / len(self.validFilesDict_)
    # Scal the learning rate
    if self.learningRateScale_ is not None:
      self.learningRate_ = self.learningRateScale_ * accuracy
    else:
      self.learningRateScale_ = self.learningRate_ / accuracy

    if accuracy < self.bestAccuracy_:
      self.bestAccuracy_ = accuracy
      print('Best accuracy %f' % self.bestAccuracy_)
      if self.bestModelName_ is not None:
        self.modelSave(self.bestModelName_)
        print('Model saved in %s' % self.bestModelName_)
    return errorFileList

