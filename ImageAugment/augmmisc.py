# Miscellaneous utilities for image augmentation
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


from shared.pyutils.tensorutils import *
from shared.pyutils.imageutils import *

def UtilAugmStitchImagesMxN(imgArr, sigma=2., dist=4):
    """
    Sticth MxN images into M x N rectangle, blurring their borders
    :param imgArr: list of lists of images
    :param sigma: Blurring gaussian sigma at the border
    :param dist: Distance from teh seam at which we are blurring
    :return: Total image
    """
    def _blurSeams(img, seamsList):
        assert len(img.shape) == 3
        h,w = img.shape[:2]
        blurArea = np.zeros((h,w,1), dtype=np.bool)
        for seam in seamsList[:-1]: # The last one is a seam at the edge
            blurArea[:,seam-dist:seam+dist,0] = True
        imgBlur = np.dstack([scipyFilters.gaussian_filter(img[:, :, i], sigma=sigma) for i in range(3)])
        return np.where(blurArea, imgBlur, img)
    def _stitchHor(imgList, padMode, seamsList):
        # Blur upper and lower edges of the image to improve smoothness of reflection
        blurredImgList = []
        for img in imgList:
            imgBlur = np.dstack([scipyFilters.gaussian_filter(img[:, :, i], sigma=sigma) for i in range(3)])
            blurredImgList.append(np.concatenate([imgBlur[:dist], img[dist:-dist], imgBlur[-dist:]], axis=0))
        return UtilStitchImagesHor(blurredImgList, padMode=padMode, seamsList=seamsList)
    imgList = []
    for horImgList in imgArr:
        seamsList = []
        img = _stitchHor(horImgList, padMode='reflect', seamsList=seamsList)
        imgList.append(np.transpose(_blurSeams(img, seamsList), axes=(1,0,2)))

    seamsList=[]
    img = _stitchHor(imgList, padMode='reflect', seamsList=seamsList)
    return np.transpose(_blurSeams(img, seamsList), axes=(1,0,2))


def UtilRandomNoiseMatrix(height, width, amplitude, sigma=None):
    noise = np.random.randn(height, width)
    if sigma is not None:
        noise = scipyFilters.gaussian_filter(noise, sigma=sigma)
    mult = amplitude / (np.linalg.norm(noise) / np.sqrt(height * width))
    return noise * mult



