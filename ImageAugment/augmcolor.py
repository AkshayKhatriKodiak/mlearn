# Utilities for color augmentation

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

from shared.pyutils.imageutils import *

def UtilAugmRandomRepaintRGBMap(strength = 10., order = 2, independentAxes=False):
    """
    Creates a repainting map of 256x256x256x3
    :param strength: average distance from an old color to a new one, along one color axis
    :param order: how quickly repainting changes (it is actually sine frequency along a dimension)
    :return: tuple (mapLS, mapH)
    """
    if not isinstance(strength, tuple):
        strength = tuple([strength] * 3)
    map = np.stack([UtilRandomSinFunc((256, 256, 256), order=order, expectedStd=strength[i], \
                                      independentAxes=independentAxes) for i in range(3)], axis=3)
    map += UtilCartesianMatrixDefault(256, 256, 256)
    map = UtilReflectCoordTensor(map)
    return np.rint(map).astype(np.int).clip(min=0, max=255).astype(np.uint8)

def UtilAugmRandomRepaintHLSMap(strengthLS=10., strengthH=10., orderLS = 2, orderH=1, independentAxes=False):
    """
    Creates 2 repainting map of 256x256x2 (LS) and 181x1 (H)
    :param strengthLS: average distance from an old color to a new one, along one of LS axes
    :param strenthH: average distance from an old color to a new one, along H axis
    :param orderLS: how quickly repainting changes (it is actually sine frequency along a dimension) along LS axes
    :param orderH: how quickly repainting changes (it is actually sine frequency along a dimension) along H axis
    :return:
    """
    if not isinstance(strengthLS, tuple):
        strengthLS = tuple([strengthLS] * 2)
    mapLS = np.stack([UtilRandomSinFunc((256, 256), order=orderLS, expectedStd=strengthLS[i], \
                                        independentAxes=independentAxes) for i in range(2)], axis=2)
    mapLS += UtilCartesianMatrixDefault(256, 256)
    mapLS = UtilReflectCoordTensor(mapLS)
    mapLS = np.rint(mapLS).clip(min=0, max=255).astype(np.uint8)

    mapH = UtilRandomSinFunc((181,), order=orderH, expectedStd=strengthH).reshape(-1,1) # Max value 180. Bug in CV2 ?
    mapH += UtilCartesianMatrixDefault(181)
    mapH = np.mod(np.rint(mapH).astype(np.uint8), 180)

    return (mapLS, mapH)

def UtilAugmConstRepaintHMap(strengthH=10, gapStart=None, gapStop=None, smooth=False):
    """
    Creates a repainting map of 181x1
    :param strenthH: shift of H coordinate
    :param gapStart: start of the interval where there should be no repaint
    :param gapStop: stop of the interval where there should be no repaint
    :param smooth: if we need to smooth transitions
    :return:
    """
    shiftArr = np.full((181,), strengthH, dtype=np.int)
    if gapStart is not None:
        assert gapStop is not None
        if gapStart < 0:
            gapStart = 180 + gapStart
        if gapStop < 0:
            gapStop = 180 + gapStop
        if gapStart < gapStop:
            shiftArr[gapStart:gapStop] = 0
        else:
            shiftArr[gapStart:] = 0
            shiftArr[:gapStop] = 0
    if smooth:
        for _ in range(5):
            shiftArr = (shiftArr[list(range(1,181))+[0]] + shiftArr[[-1]+list(range(0,180))]) / 2
    shiftArr = shiftArr.reshape((-1, 1))
    mapH = UtilCartesianMatrixDefault(181) + shiftArr # Max value 180. Bug in CV2 ?
    mapH = np.mod(np.rint(mapH), 180).astype(np.uint8)
    return mapH

def UtilAugmIncrSaturLSMap(multiply):
    """
    Creates an LS repainting map of 256x256x2
    :param multiply: by how much saturaton needs to be multiplied
    :return:
    """
    mapLS = UtilCartesianMatrixDefault(256, 256) * np.array([1., multiply], dtype=np.float32)
    mapLS = np.rint(mapLS).clip(max=255).astype(np.uint8)
    return mapLS

def UtilAugmRepaintRGB(img, repaintMap):
    """
    Randomply remap picture colors
    :param img: original image
    :param repaintMap: Map of 256x256x256x3
    :return:
    """
    h,w = img.shape[:2]
    imgIndxsR, imgIndxsG, imgIndxsB = np.transpose(UtilImageToInt(img).reshape(-1,3))
    return repaintMap[imgIndxsR, imgIndxsG, imgIndxsB].reshape((h,w,3)).astype(np.float32)

def UtilAugmRepaintHLS(img, repaintMap):
    """
    Randomply remap picture colors
    :param img: original image
    :param repaintMap: Map returned from UtilAugmRandomRepaintHLSMap
    :return:
    """
    h,w = img.shape[:2]
    if isinstance(repaintMap, tuple):
        repaintMapLS, repaintMapH = repaintMap
    else:
        if repaintMap.shape == (181, 1):
            repaintMapH = repaintMap
            repaintMapLS = UtilCartesianMatrixDefault(256, 256).astype(np.uint8)
        else:
            assert repaintMap.shape == (256, 256, 2)
            repaintMapH = UtilCartesianMatrixDefault(181).astype(np.uint8)
            repaintMapLS = repaintMap
    img = cv2.cvtColor(UtilImageToInt(img), cv2.COLOR_RGB2HLS)
    imgIndxsH, imgIndxsL, imgIndxsS = np.transpose(img.reshape(-1,3))
    repaintLS = repaintMapLS[imgIndxsL, imgIndxsS]
    img = np.stack([repaintMapH[imgIndxsH][:,0], repaintLS[:,0], repaintLS[:,1]], axis=1)
    return cv2.cvtColor(img.reshape((h,w,3)), cv2.COLOR_HLS2RGB).astype(np.float32)

def UtilAugmColorReplacement(img, colorStr):
    """
    Replace colors in teh original image
    :param img:
    :param colorStr: string of type "brg" etc
    :return:
    """
    d = {'r':0, 'g':1, 'b':2}
    seq = [d[c] for c in colorStr]
    return np.stack([img[:,:,i] for i in seq], axis=2)

