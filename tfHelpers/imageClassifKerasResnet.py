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
from multiprocessing import cpu_count
import multiprocessing
import traceback

from shared.pyutils.imageutils import *
from shared.pyutils.utilMultiProc import UtilMultithreadQueue, \
  UtilMultiprocQueue

np.set_printoptions(threshold=np.nan)

import mlearn.tfHelpers.kerasResnet as resnet
import keras
from keras import optimizers
from keras import backend as K
from keras.losses import categorical_crossentropy, mean_absolute_error
from keras.utils import Sequence

ImageChannels = 3 # RGB images

# For debugging only, to speed up and of epoch, should be None otherwise
DebugLimitValidationCount = None
# DebugLimitValidationCount = 3

# How often to log current model
LogIntervalInSeconds = 7200.

def _less(x, y):
  return x < y

def _greater(x, y):
  return x > y

class ImageClassifKerasResnetTrainGenerator(Sequence):
  def __init__(self, classif):
    self.classif_ = classif

  def __len__(self):
    return self.classif_.trainBatchesPerEpoch()

  def __getitem__(self, index):
    print("Train index %d" % index)
    ret = self.classif_.getBatch(training=True)
    return ret

  def on_epoch_end(self):
    print("Trining epoch done")


class ImageClassifKerasResnetValidGenerator(Sequence):
  def __init__(self, classif):
    self.classif_ = classif

  def __len__(self):
    ret = self.classif_.validBatchesPerEpoch()
    if DebugLimitValidationCount is not None:
      ret = DebugLimitValidationCount
    print("Validation len %d" % ret)
    return ret

  def __getitem__(self, index):
    print("Valid index %d" % index)
    ret = self.classif_.getBatch(training=False)
    return ret

  def on_epoch_end(self):
    #self.classif_.resetValidState()
    print("Validation epoch done")


class CustomLogger(keras.callbacks.Callback):
  def __init__(self, classif, logMetrics):
    self.classif_ = classif
    self.logMetrics_ = logMetrics

  def on_epoch_begin(self, epoch, logs=None):
    pass

  def on_batch_end(self, batch, logs=None):
    pass

  def on_epoch_end(self, epoch, logs=None):
    print("Epoch %d done, logs %s" % (epoch, str(logs)))
    self.classif_.logCallback(epoch, logs[self.logMetrics_])



class ImageClassifKerasResnet:

  allowedArchitects_ = [18, 34, 50, 101, 152]

  def __init__(self, classCount=None, imgHeight=None, imgWidth=None,
    modelName=None, architectureNumber=18, lossFunc=None, buildRegressor=False,
    dropout=None):

    assert architectureNumber in ImageClassifKerasResnet.allowedArchitects_

    self.buildRegressor_ = buildRegressor

    self.trainGenerator_ = ImageClassifKerasResnetTrainGenerator(self)
    self.validGenerator_ = ImageClassifKerasResnetValidGenerator(self)

    self.classCount_ = classCount
    self.height_ = imgHeight
    self.width_ = imgWidth
    
    # if clrMeans and clrStds are None then we normalize image by
    # dividing it by 255
    self.clrMeans_ = None
    self.clrStds_ = None
    if lossFunc is None:
      if self.buildRegressor_:
        lossFunc = mean_absolute_error
        self.metrics_ = "mae"
        logMetrics = "val_mean_absolute_error"
        self.comparison_ = _less
        self.bestValidLoss_ = float("inf")
      else:
        lossFunc = categorical_crossentropy
        self.metrics_ = "acc"
        logMetrics = "val_acc"
        self.comparison_ = _greater
        self.bestValidLoss_ = float("-inf")
    self.lossFunc_ = lossFunc
    self.customLogger_ = CustomLogger(self, logMetrics)

    self.prevBestModel_ = None
    self.lastLog_ = time.time()

    # Creating model
    resnet.ResnetBuilder.set_dropout(dropout)
    if self.buildRegressor_:
      self.model_ = resnet.ResnetBuilder.build_resnet_regressor(
        architectureNumber, (ImageChannels, imgHeight, imgWidth), classCount)
    else:
      architectureName = ('build_resnet_%d' % architectureNumber)
      architectureFunc = getattr(resnet.ResnetBuilder, architectureName)
      self.model_ = architectureFunc((ImageChannels, imgHeight, imgWidth),
        classCount)
    if modelName is not None:
      # Load weights that were saved earlier
      self.model_.load_weights(modelName)
    sgd = optimizers.SGD(decay=0., momentum=0.9, nesterov=True)
    self.model_.compile(loss=self.lossFunc_,
                  # optimizer='adam',
                  optimizer=sgd,
                  metrics=[self.metrics_])

  # Dictionaries map file name to a class id
  def trainInit(self, trainFilesDict, validFilesDict,
      trainBatchSize=None, validBatchSize=None, batchesPerEpoch=None,
      pointsFilesDict=None,
      augmenter=None, bestModelPrefix=None, logDir=None):

    # Calculates weights for each class in teh raining set
    counts = [0] * self.classCount_
    for cl in trainFilesDict.values():
      counts[cl] += 1
    classWeights = []
    for count in counts:
      classWeights.append(1. / (count * self.classCount_))
    print("CLASS WEIGHTS: %s" % str(classWeights))

    self.classWeights_ = classWeights
    self.trainFilesDict_ = trainFilesDict
    self.pointsFilesDict_ = pointsFilesDict
    self.trainFilesList_ = list(self.trainFilesDict_.keys())
    self.trainFilesWeights_ = [classWeights[trainFilesDict[fn]] \
      for fn in self.trainFilesList_] 
    assert np.abs(sum(self.trainFilesWeights_) - 1.) < 0.00001
    self.validFilesDict_ = validFilesDict
    self.validFilesList_ = list(self.validFilesDict_.keys())
    self.logDir_ = logDir
    self.augmenter_ = augmenter
    self.bestModelPrefix_ = bestModelPrefix
    self.trainBatchSize_ = trainBatchSize
    if validBatchSize is None:
      validBatchSize = trainBatchSize
    self.validBatchSize_ = validBatchSize
    self.batchesPerEpoch_ = batchesPerEpoch

    self.mprocTraining_ = UtilMultiprocQueue(self.trainCback,
      logFileDir=self.logDir_, procCount = 2*cpu_count(),
      maxQueueSize=300, name="train")

    self.mprocValidation_ = UtilMultiprocQueue(
      self.validCback,
      logFileDir=self.logDir_, procCount = 2*cpu_count(),
      maxQueueSize=300, name="valid")
    
    self.validState_ = self.mprocValidation_.getManager().list()
    self.validState_.append(0)
    self.mprocValidation_.setState(self.validState_)

    self.mprocTraining_.start()
    self.mprocValidation_.start()


  def trainFinish(self):
    self.mprocValidation_.terminate()
    self.mprocValidation_ = None
    self.mprocTraining_.terminate()
    self.mprocTraining_ = None


  def resetValidState(self):
    self.mprocValidation_.lock()
    self.validState_[0] = 0
    self.mprocValidation_.unlock()

  def saveModel(self, modelNamePrefix, epoch, errorDict={}):
    s = "_%d" % epoch
    for k, v in errorDict.items():
      s += ("_%s_%.3f" % (k, v)) 
    modelName = modelNamePrefix + s + '.h5'
    self.model_.save_weights(modelName)
    return modelName

  def logCallback(self, epoch, valLoss):
    saveDict = { "metrics" : valLoss }

    if time.time() - self.lastLog_ > LogIntervalInSeconds:
      self.lastLog_ = time.time()
      if self.saveModelPrefix_ is not None:
        self.saveModel(self.saveModelPrefix_, epoch, saveDict)

    if self.comparison_(valLoss, self.bestValidLoss_):
      self.bestValidLoss_ = valLoss
      if self.bestModelPrefix_ is not None:
        bestModelName = self.saveModel(
          self.bestModelPrefix_, epoch, saveDict)
        if self.prevBestModel_ is not None:
          os.remove(self.prevBestModel_)
        self.prevBestModel_ = bestModelName 
      

  def summary(self):
    return self.model_.summary()

  def getInputShape(self):
    return(self.height_, self.width_)

  def getClassCount(self):
    return self.classCount_

  def trainBatchesPerEpoch(self):
    return self.batchesPerEpoch_

  def validBatchesPerEpoch(self):
    return len(self.validFilesList_) // self.validBatchSize_

  def setScaledNormalization(clrMeans, clrStds):
    assert (clrMeans is not None) == (clrStds is not None)
    self.clrMeans_ = clrMeans
    self.clrStds_ = clrStds

  def imgNormalize(self, img):
    if self.clrMeans_ is not None:
      return (img - self.clrMeans_) / self.clrStds_
    else:
      return img / 255.

  def fileToInput(self, imageFileName):
    img = UtilImageFileToArray(imageFileName)
    img = UtilImageResize(img, self.height_, self.width_)
    img = self.imgNormalize(img)
    return img

  def trainCback(self, state, lock):
    fileName = np.random.choice(self.trainFilesList_,
      p=self.trainFilesWeights_)
    print("Training file %s" % fileName)
    img = UtilImageFileToArray(fileName)
    assert img.shape[:2] == (self.height_, self.width_)
    points = None
    if self.pointsFilesDict_ is not None:
      points = [self.pointsFilesDict_[fileName]]
    augmDesc = [""]
    if self.augmenter_ is not None:
      img = self.augmenter_.augment(img, points=points, retDesc=augmDesc)
    img = self.imgNormalize(img)
    return (True, (img, self.trainFilesDict_[fileName], fileName, augmDesc[0]))


  def validCback(self, state, lock):
    lock.acquire()
    ind = state[0]
    if ind == len(self.validFilesList_):
      ind = 0
    state[0] = ind + 1
    lock.release()
    fileName = self.validFilesList_[ind]
    img = UtilImageFileToArray(fileName)
    img = self.imgNormalize(img)
    return (True, (img, self.validFilesDict_[fileName], fileName))


  def getTrainingItem(self):
    img, label, fname, augmDesc = self.mprocTraining_.getData()
    return (img, label, fname, augmDesc)


  def getValidItem(self):
    return self.mprocValidation_.getData()


  def getBatch(self, training=True):
    batchSize = self.trainBatchSize_ if training else self.validBatchSize_
    images = np.empty((batchSize, self.height_, self.width_,
      ImageChannels), dtype=np.float32)
    labelsDim = 1 if self.buildRegressor_ else self.classCount_
    labels = np.zeros((batchSize, labelsDim), dtype=np.float32)
    for i in range(batchSize):
      if training:
        images[i, :, :, :], label, _, _ = self.getTrainingItem()
      else:
        images[i, :, :, :], label, _ = self.getValidItem()
      if self.buildRegressor_:
        labels[i, 0] = (label + 0.5) / self.classCount_
      else:
        labels[i, label] = 1.

    return (images, labels)


  def train_validate(self, initEpoch, epochCount, learningRate,
    saveModelPrefix=None):

    self.saveModelPrefix_ = saveModelPrefix

    endEpoch = initEpoch + epochCount
    self.resetValidState()

    # Overwrite learning rate
    print('Setting learning rate %f' % learningRate)
    K.set_value(self.model_.optimizer.lr, learningRate)

    print('TRAINING EPOCHS %d - %d' % (initEpoch, endEpoch))

    history = self.model_.fit_generator(
      self.trainGenerator_,
      epochs=endEpoch,
      verbose=2,
      validation_data=self.validGenerator_,
      shuffle=False,
      initial_epoch=initEpoch,
      callbacks=[self.customLogger_])

    return history


  def predict(self, imageFileName):
    #TODO - get sizes from the model
    img = self.fileToInput(imageFileName)
    if img.shape[:2] != (self.height_, self.width_):
      img = UtilImageResize(img, self.height_, self.width_)
    imgArray = np.stack([img], axis=0)
    predictions = self.model_.predict(imgArray,
      batch_size=self.validBatchSize_, verbose=1)
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
  height = int(sys.argv[4])
  width = int(sys.argv[5])
  metaDataFile = None
  if len(sys.argv) >= 7:
    metaDataFile = sys.argv[6]

  c = ImageClassifKerasResnet(modelName=modelName,
    imgHeight=height, imgWidth=width)

  if metaDataFile is not None:
    with open(metaDataFile, 'rb') as fin:
      clrMeans, clrStds = dill.load(fin)
    c.setScaledNormalization(clrMeans, clrStds)

  print('PREDICTION: %d' % c.predict(imageName))
  

if __name__ == '__main__':
  if sys.argv[1] == 'summary':
    _dumpSummary()
  elif sys.argv[1] == 'predict':
    _predict()
