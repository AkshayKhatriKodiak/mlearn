# Miscellaneous utilities for image augmentation

# Copyright (C) 2016-2017  Author: Misha Orel
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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





