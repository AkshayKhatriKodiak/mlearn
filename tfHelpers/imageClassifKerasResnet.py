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

  allowedArchitects_ = [18, 34, 50, 101, 152]

  def __init__(self, classCount=None, imgHeight=None, imgWidth=None,
    clrMeans=None, clrStds=None, modelName=None, learningRate=0.,
    architectureNumber=18, batchSize=1):

    assert architectureNumber in ImageClassifKerasResnet.allowedArchitects_

    self.classCount_ = classCount
    self.height_ = imgHeight
    self.width_ = imgWidth
    self.clrMeans_ = clrMeans
    self.clrStds_ = clrStds
    self.batchSize_ = batchSize

    if modelName is None:
      # Creating model from scratch
      assert learningRate is not None
      architectureName = ('build_resnet_%d' % architectureNumber)
      architectureFunc = getattr(resnet.ResnetBuilder, architectureName)
      self.model_ = architectureFunc((ImageChannels, imgHeight, imgWidth),
        classCount)
      sgd = optimizers.SGD(lr=learningRate, decay=0., momentum=0.9, nesterov=True)
      self.model_.compile(loss='categorical_crossentropy',
                    # optimizer='adam',
                    optimizer=sgd,
                    metrics=['accuracy'])
    else:
      # Loading model from file
      self.model_ = keras.models.load_model(modelName)
      if learningRate is None:
        try:
          learningRate = K.get_value(self.model_.optimizer.lr)
        except:
          # If learning rate has not been saved as part of a model
          # (eg inference model only), then leave it as None
          pass

    self.learningRate_ = learningRate
    self.learningRateScale_ = None

    self.classCount_ = self.getClassCount()
    self.height_, self.width_ = self.getInputShape()


  # Dictionaries map file name to a class id
  def trainInit(self, trainFilesDict, validFilesDict, augmenter,
               bestModelName=None):
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
    self.augmenter_ = augmenter
    self.bestModelName_ = bestModelName
    self.bestAccuracy_ = 1.0


  def modelSave(self, modelName):
    keras.models.save_model(self.model_, modelName)

  def summary(self):
    return self.model_.summary()

  def getInputShape(self):
    shape = self.model_.layers[0].input.get_shape()[1:3]
    return((shape[0], shape[1]))

  def getClassCount(self):
    return self.model_.layers[-1].output.get_shape()[1]

  def imgNormalize(self, img):
    return (img - self.clrMeans_) / self.clrStds_

  def fileToInput(self, imageFileName):
    img = UtilImageFileToArray(imageFileName)
    img = UtilImageResize(img, self.height_, self.width_)
    img = self.imgNormalize(img)
    return img

  def train(self, rounds):
    def _trainCback(state, lock):
      fileName = np.random.choice(self.trainFilesList_,
        p=self.trainFilesWeights_)
      img = UtilImageFileToArray(fileName)
      assert img.shape[:2] == (self.height_, self.width_)
      img = self.augmenter_.augment(img)
      img = self.imgNormalize(img)
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
      img = self.imgNormalize(img)
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

  def predict(self, imageFileName):
    img = self.fileToInput(imageFileName)
    imgArray = np.stack([img], axis=0)
    predictions = self.model_.predict(imgArray,
      batch_size=self.batchSize_, verbose=1)
    return np.argmax(predictions[0])

  def classActivationMap(self, imageFileName, outCamFile):
    """
    outCamFile is a name of output image file containing Class
    Activation Map superimposed on the input image
    """

    # First do normal prediction
    classPred = self.predict(imageFileName)

    # Last layer is linear regression -> soft max input values
    classWeights = self.model_.layers[-1].get_weights()[0]
    # Final convolution layer in resnet is -4
    finalConvOutput = self.model_.layers[-4].output

    # Let's get function mapping input to outputs of last conv layer
    # and softmax
    func = K.function([self.model_.layers[0].input], [finalConvOutput])
    img = self.fileToInput(imageFileName)
    imgArray = np.stack([img], axis=0)
    lastConvOutput = func([imgArray])[0][0]
    assert len(lastConvOutput.shape) == 3 # height x width x conv channels

    # Class activation map.
    clActMap = np.zeros(dtype = np.float32, shape = lastConvOutput.shape[:2])
    for ind, weight in enumerate(classWeights[:, classPred]):
      clActMap += weight * lastConvOutput[:, :, ind]

    # TODO - combine with input image, save
    print('CAM:\n%s' % str(clActMap))
    

# If used as utility

def _dumpSummary():
  classCount = int(sys.argv[2])
  height = int(sys.argv[3])
  width = int(sys.argv[4])
  architecture = int(sys.argv[5])
 
  c = ImageClassifKerasResnet(classCount, height, width,
    architectureNumber=architecture)
  print(c.summary())


def _predict():
  modelName = sys.argv[2]
  imageName = sys.argv[3]
  metaDataFile = sys.argv[4]

  with open(metaDataFile, 'rb') as fin:
    clrMeans, clrStds = dill.load(fin)

  c = ImageClassifKerasResnet(clrMeans=clrMeans, clrStds=clrStds,
    modelName=modelName)
  print('PREDICTION: %d' % c.predict(imageName))
  

if __name__ == '__main__':
  if sys.argv[1] == 'summary':
    _dumpSummary()
  elif sys.argv[1] == 'predict':
    _predict()
