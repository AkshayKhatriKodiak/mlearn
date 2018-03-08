# Geometrical distortions for image augmentation

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

import psutil
from shared.pyutils.tensorutils import *
from shared.pyutils.imageutils import *

AllowedBinaryMaskExt = [".png", ".bmp"]

class UtilAugmCachedMap(UtilObject):
    sharedMapDict = {} # Default map dictionary, not multiprocessing
    localMapDict = {} # This guy is not shared between processes, keeps maps that were initiated from
                      # this process
    lruList = UtilSimpleLinkedList()
    CheckInterval = 50
    debug = False
    counter = 0
    localHitCounter = 0
    globalHitCounter = 0
    trimCounter = 0


    @staticmethod
    def _getDefaultReservedMem():
        """
        1 GB, can be overriden with AUGMGEOM_CACHED_MAP_RESERVE_GB environment variable
        :return:
        """
        reserveGb = os.environ.get('AUGMGEOM_CACHED_MAP_RESERVE_GB')
        if reserveGb is None:
            reserveGb = 1
        else:
            reserveGb = int(reserveGb)
        return reserveGb << 30 # in GB


    def __init__(self, func, mapDict=sharedMapDict, reservedMem=None, **kwargs):
        if reservedMem is None:
            reservedMem = self._getDefaultReservedMem()

        cls = self.__class__

        cls.counter += 1
        if cls.debug and (cls.counter % 100 == 0):
            print('UtilAugmCachedMap local hits %d global hits %d misses %d trimmings %d ' \
                    'dict size %d local dict size %d' % \
                  (cls.localHitCounter, cls.globalHitCounter,
                   cls.counter - (cls.localHitCounter + cls.globalHitCounter), \
                  cls.trimCounter, len(mapDict), len(cls.localMapDict)))

        # See if we are running out of memory
        if (cls.counter % cls.CheckInterval == 0):
            assert cls.lruList.count() == len(cls.localMapDict)
            if psutil.virtual_memory().available < reservedMem:
                cls.trimCounter += 1
                trimCount = cls.lruList.count() // 20
                oldGlobalMapLen = len(mapDict)
                for _ in range(trimCount): # Removing 5% of the lru list
                    obj = cls.lruList.head
                    assert cls.localMapDict[obj.key] == obj
                    cls.lruList.dequeue(obj)
                    del cls.localMapDict[obj.key]
                    try:
                        # Might have already been removed
                        del mapDict[obj.key]
                    except KeyError:
                        pass
                if cls.debug:
                    print('Trimmed %d entries Global map %d -> %d (reduced by %d) glob mem %u' %
                          (trimCount, oldGlobalMapLen, len(mapDict), oldGlobalMapLen - len(mapDict),
                           psutil.virtual_memory().available))

        key = (func.__name__,) + tuple(kwargs.values())

        obj = None
        if key in cls.localMapDict:
            cls.localHitCounter += 1
            obj = cls.localMapDict[key]
            cls.lruList.dequeue(obj)
            cls.lruList.append(obj)
        elif key in mapDict:
            try:
                obj = mapDict[key]
                cls.globalHitCounter += 1
            except KeyError:
                pass

        if obj is not None:
            self.map = obj.map
            self.reverseMap = obj.reverseMap
            return

        self.map = func(**kwargs)
        self.reverseMap = UtilAugmReverseMapping(self.map)
        self.key = key
        mapDict[key] = self
        cls.localMapDict[key] = self
        cls.lruList.append(self)


def UtilAugmCircleMappingLeft(boundRect,center,height,width):
    """
    Map image in a circullar manner
    :param boundRect: bounding box of the image object, float 4-tuple
    :param center: module of x position of the center (y of the center is at the bottom of boundRect)
    :param height: height of the rectangle
    :param width: witdth of the rectangle
    :return: matrix of tuples (yMapped, xMapped)
    """
    yMin, xMin, yMax, xMax = boundRect
    xMidline = (xMin + xMax) / 2
    arr = np.empty((height,width,2), dtype=np.float32)
    for i in range(width):
        for j in range(height):
            # Translate to coordinates with (0,0) at the center of the circle
            x = center + i
            y = j - yMax
            # Translate into polar coordinates
            r = math.sqrt(x*x + y*y)
            phi = math.asin(y / r)
            # Translate from polar to original coordinates
            y = phi * (center + xMidline) + yMax
            x = r - center
            arr[j][i] = np.array([y,x])
    return arr

def UtilAugmCircleMappingRight(boundRect,center,height,width):
    arr = UtilAugmCircleMappingLeft(boundRect,center,height,width)
    refArray = np.tile(range(width), height).reshape((height, width))
    arr[:,:,1] = 2. * refArray - arr[:,:,1]
    return arr

def UtilAugmRandomAxisScale(size, freqCtrl=30., depthCtrl = 1.5):
    sigma = size / freqCtrl
    arr = np.random.randn(size)
    return scipyFilters.gaussian_filter1d(arr, sigma) * math.sqrt(sigma) * depthCtrl

def _augmDblSinAxisScale(size, freqCtrl=4., depthCtrl = 1.):
    ampl1 = np.random.randn() * depthCtrl
    ampl2 = np.random.randn() * depthCtrl
    freq1 = np.random.randn() * freqCtrl * np.pi
    freq2 = np.random.randn() * freqCtrl * np.pi
    phase1 = np.random.rand() * 2 * np.pi
    phase2 = np.random.rand() * 2 * np.pi
    f = np.vectorize(lambda x: ampl1*math.sin(freq1*x/size + phase1) + ampl2*math.sin(freq2*x/size + phase2))
    return f(np.array(range(size)))

def UtilAugmIndepAxes(height, width, axisFunc, **kwargs):
    def convert(arr):
        size = len(arr)
        arr = np.exp(arr)
        arr = arr / np.sum(arr) * size
        for i in range(1, size):
            # Integrate it
            arr[i] += arr[i - 1]
        return arr
    arrY = convert(axisFunc(height, **kwargs))
    arrX = convert(axisFunc(width, **kwargs))
    arr = np.empty((height, width, 2), dtype = np.float32)
    for j in range(height):
        for i in range(width):
            arr[j,i,:] = np.array([arrY[j], arrX[i]])
    return arr

def UtilAugmSimmetry1d(height, width, midCoord, isVertical, depthCtrl=0.1, lowerLimit=None, upperLimit=None):
    if not isVertical:
        height, width = (width, height)
    if lowerLimit is None:
        lowerLimit = 0
    if upperLimit is None:
        upperLimit = height
    interval = upperLimit - lowerLimit
    arr = 1. - depthCtrl * np.sin((np.array(range(height), dtype=np.float32) - lowerLimit) * np.pi / interval)
    scaledX = np.outer(arr, np.array(range(width), dtype=np.float32) - midCoord) + midCoord
    assert scaledX.shape == (height, width)
    unscaledY = np.repeat(np.array(range(height)), width).reshape((height, width))
    output = np.dstack([unscaledY, scaledX]).astype(np.float32)
    if not isVertical:
        output = np.flip(np.flip(np.rot90(output, axes=(0,1)), axis=2), axis=0)
    return output

def UtilAugmStretch1d(height, width, ratio, midCoord, isVertical):
    if not isVertical:
        height, width = (width, height)
    indDiff = UtilCartesianMatrixDefault(height, width) - [midCoord, 0.]
    output = np.stack([indDiff[:,:,0] / ratio + midCoord, indDiff[:,:,1]], axis=2)
    if not isVertical:
        output = np.flip(np.flip(np.rot90(output, axes=(0,1)), axis=2), axis=0)
    return output

def UtilAugmStretch2d(height, width, ratio, centerPoint):
    yCenter, xCenter = centerPoint
    indDiff = UtilCartesianMatrixDefault(height, width) - [yCenter, xCenter]
    return indDiff / ratio + [yCenter, xCenter]

def UtilAugmRotate(height, width, angle, centerPoint):
    yCenter, xCenter = centerPoint
    indDiff = UtilCartesianMatrixDefault(height, width) - [yCenter, xCenter]
    # Remember, image is upside down by axis Y
    s,c = (math.sin(angle), math.cos(angle)) # Minus because we go from rotated matrix to the original one
    indY = indDiff[:,:,0] * c + indDiff[:,:,1] * s
    indX = indDiff[:,:,1] * c - indDiff[:,:,0] * s
    return np.stack([indY, indX], axis=2) + [yCenter, xCenter]

def UtilAugmReverseMapping(arrMap):
    """
    Maps original pixells into the new ones
    :param mapObj: UtilAugmBidirMap with the forward map
    :return: mapObj with reverse map
    """
    height, width, coordCount = arrMap.shape
    assert coordCount == 2
    tupleArr = np.empty((height, width), dtype = object)
    tupleArr.fill(None)
    def validInd(tup):
        return ((0 <= tup[0] < height) and (0 <= tup[1] < width))
    def addToSet(tup, s):
        if validInd(tup) and (tupleArr[tup] is None):
            s.add(tup)

    # Initialize tuple array
    filled = set()
    for j in range(height):
        for i in range(width):
            y,x = np.rint(arrMap[j,i]).astype(np.int)
            if validInd((y,x)):
                tupleArr[y,x] = (j,i)
                filled.add((y,x))

    #print("Rev map: initially filled %d out of %d" % (len(filled), height * width))

    while len(filled) != 0:
        s = set()
        for j,i in filled:
            for jj in (-1,1):
                for ii in (-1,1):
                    addToSet((j+jj, i+ii), s)

        filled = set()
        for j,i in s:
            assert tupleArr[j,i] is None
            l = []
            for jj in (-1, 1):
                for ii in (-1, 1):
                    tup = (j+jj, i+ii)
                    if validInd(tup) and (tupleArr[tup] is not None):
                        l.append(tupleArr[tup])
            y = sum([v[0] for v in l]) / len(l)
            x = sum([v[1] for v in l]) / len(l)
            y = int(round(y))
            x = int(round(x))
            tupleArr[j,i] = (y,x)
            filled.add((j,i))

    assert not np.any(np.vectorize(lambda x: x is None)(tupleArr))

    # Covert array of tuples to np array
    return np.dstack([np.vectorize(operator.itemgetter(i))(tupleArr) for i in (0,1)])

def UtilAdjustBinaryMask(img, high=255., low=0.):
    if len(img.shape) == 3:
        img = UtilFromRgbToGray(img)
    boolImg = img >= 128
    return np.where(boolImg, high, low)

def UtilAdjustBinaryMaskToInt(img):
    return UtilAdjustBinaryMask(img, high=255, low=0)

def UtilSaveBinaryMask(img, fileName):
    assert os.path.splitext(fileName)[1].lower() in AllowedBinaryMaskExt
    img = UtilAdjustBinaryMask(img)
    UtilArrayToImageFile(img, fileName)

def UtilLoadBinaryMask(fileName):
    if fileName in (None, ''):
        return None
    assert os.path.splitext(fileName)[1].lower() in AllowedBinaryMaskExt
    img = UtilImageFileToArray(fileName)
    if img is None:
        return None
    return UtilAdjustBinaryMask(img)

def UtilRemapBinaryMask(imgMask, map):
    """
    For a binary mask we should remove all splining
    """
    imgMask = UtilRemapImage(imgMask, map, fillMethod='constant', fillValue=127., ky=1, kx=1)
    return UtilAdjustBinaryMask(imgMask)
