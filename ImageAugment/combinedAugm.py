# Class for combined image augmentation
#
# Copyright (C) 2014-2018  Author: Misha Orel
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

import dill
import random

from shared.pyutils.imageutils import *
from mlearn.ImageAugment.augmmisc import *
from mlearn.ImageAugment.augmcolor import *

class CombinedImageAugmentation(UtilObject):

  def __init__(self, height, width,
               flipProb=0.5, blurProb=0.25, sharpProb=0.25, noiseProb=0.5, lfNoiseProb=0.5, saturProb=0.5):

    self.height_ = height
    self.width_ = width

    assert blurProb + sharpProb < 1.
    sharpProb /= (1. - blurProb) # Sharpen is checked after blur

    self.flipProb_ = flipProb
    self.blurProb_ = blurProb
    self.sharpProb_ = sharpProb
    self.noiseProb_ = noiseProb
    self.lfNoiseProb_ = lfNoiseProb
    self.saturProb_ = saturProb

  def setBlur(self, blurSigmaMax):
    self.blurSigmaMax_ = blurSigmaMax

  def setSharp(self, edgeAmpMax):
    self.edgeAmpMax_ = edgeAmpMax

  def setNoise(self, noiseAmpMax):
    self.noiseAmpMax_ = 256. * noiseAmpMax

  def setLfNoise(self, sigmaMax, ampMax, cachedCount=1000, pklFile=None):
    # Don't create too mnay in cache, remmeber that amplitude is still selected randomly

    self.ampMax_ = ampMax

    if pklFile is not None:
      if os.path.isfile((pklFile)):
        with open(pklFile, 'rb') as fin:
          self.lfNoiseMats_ = dill.load(fin)
        if (self.lfNoiseMats_[0].shape[0] == self.height_) and \
          (self.lfNoiseMats_[0].shape[1] == self.width_):
          return
        else:
          print('LFNoise size mismatch')

    self.lfNoiseMats_ = []
    for _ in range(cachedCount):
      self.lfNoiseMats_.append(UtilRandomNoiseMatrix(self.height_, self.width_,
          amplitude=1., sigma=np.random.random()*sigmaMax))
    if pklFile is not None:
      with open(pklFile, 'wb') as fout:
        dill.dump(self.lfNoiseMats_, fout)

    print('Done generating LFNoise cache')

  def setSatur(self, scaleMin, scaleMax):
    self.scaleMin_ = scaleMin
    self.scaleMax_ = scaleMax

  ########## Ran time methods #############

  def lfNoiseMat(self):
    return random.choice(self.lfNoiseMats_)

  def satScale(self):
    return np.random.random() * (self.scaleMax_ - self.scaleMin_) + self.scaleMin_

  def augment(self, img, retDesc=None):
    channels = img.shape[2]
    height = img.shape[0]
    width = img.shape[1]
    assert height <= self.height_
    assert width <= self.width_

    desc = []

    def _toStr(f):
      return ('%.3f' % f)

    # Flipping
    if np.random.random() < self.flipProb_:
      desc.append('f')
      img = np.flip(img, axis=1)

    # Bluring, sharping
    do_blur, do_sharp = (False, False)
    if np.random.random() < self.blurProb_:
      desc.append('b')
      do_blur = True
    elif np.random.random() < self.sharpProb_:
      desc.append('s')
      do_sharp = True
    if do_blur or do_sharp:
      sigma = np.random.random() * self.blurSigmaMax_
      desc.append(_toStr(sigma))
      imgNew = np.dstack([scipyFilters.gaussian_filter(img[:, :, i],
          sigma=sigma) for i in range(channels)])
      if do_blur:
        img = imgNew
      else:
        edgeAmp = np.random.random() * self.edgeAmpMax_
        desc.append(_toStr(edgeAmp))
        img += edgeAmp * (img - imgNew)

    # Saturation
    if np.random.random() < self.saturProb_:
      scale = self.satScale()
      desc.append('sat' + _toStr(scale))
      img = UtilAugmScaleSaturation(img, scale)

    # Noise
    if np.random.random() < self.noiseProb_:
      amp = np.random.random() * self.noiseAmpMax_
      desc.append('n' + _toStr(amp))
      img += np.dstack([np.random.randn(height, width) * amp for _ in range(channels)])

    # Low frequency noise
    if np.random.random() < self.lfNoiseProb_:
      desc.append('lfn')
      amp = 256. * np.random.random(3) * self.ampMax_
      for a in amp:
        desc.append(_toStr(a))
      verShift = np.random.randint(low=0, high=self.height_ - height + 1)
      horShift = np.random.randint(low=0, high=self.width_ - width + 1)
      lfnMat = np.dstack([self.lfNoiseMat()[verShift:verShift+height,
                          horShift:horShift+width] for _ in range(channels)])
      lfnMat *= amp
      img += lfnMat

    if retDesc is not None:
      retDesc.append('_'.join(desc))

    return img


# Test for CombinedImageAugmentation class

if __name__ == "__main__":
  inputFilePattern = "/Users/morel/temp/*.jpg" # Put real directory here
  outputDir = "/Users/morel/temp/augm/" # Put real directory here

  cia = CombinedImageAugmentation(height=1500, width=2000)
  cia.setBlur(blurSigmaMax=1.5)
  cia.setSharp(edgeAmpMax=1.0)
  cia.setNoise(noiseAmpMax=0.12)
  cia.setLfNoise(sigmaMax=3., ampMax=0.1, cachedCount=30, pklFile='/Users/morel/temp/lfnoise.pkl')
  cia.setSatur(scaleMin=0., scaleMax=1.2)

  for fn in glob.glob(inputFilePattern):
    print('Processing %s' % fn)
    _, fnBase = os.path.split(fn)
    fnBase, ext = os.path.splitext(fnBase)

    retDesc = []
    img = UtilImageFileToArray(fn)
    img = cia.augment(img, retDesc=retDesc)

    outFile = outputDir + '/' + fnBase + '_' + retDesc[0] + ext
    UtilArrayToImageFile(img, outFile)



