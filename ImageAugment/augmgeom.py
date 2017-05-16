__author__ = "Misha Orel"

from shared.pyutils.tensorutils import *
from shared.pyutils.imageutils import *

AllowedBinaryMaskExt = [".png", ".bmp"]

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
    arr[:,:,1] = 2. * arr[:,:,1] - refArray
    return arr

def UtilAugmRandomAxisScale(size, freqCtrl=30., depthCtrl = 1.5):
    sigma = size / freqCtrl
    arr = np.random.randn(size)
    return scipyFilters.gaussian_filter1d(arr, sigma) * math.sqrt(sigma) * depthCtrl

def UtilAugmDblSinAxisScale(size, freqCtrl=4., depthCtrl = 1.):
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

def UtilAugmSimmetry1d(height, width, midCoord, isVertical, freqCtrl=4., depthCtrl=0.1, minScale=0.7, maxScale=1.4):
    if not isVertical:
        height, width = (width, height)
    counter = 20
    while counter:
        arr = UtilAugmDblSinAxisScale(height, freqCtrl, depthCtrl) + 1.
        if (np.min(arr) > minScale) and (np.max(arr) < maxScale):
            break
        counter -= 1
    if counter == 0:
        return None
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
    indDiff = UtilCartesianMatrix(range(height), range(width)) - [midCoord, 0.]
    output = np.stack([indDiff[:,:,0] / ratio + midCoord, indDiff[:,:,1]], axis=2)
    if not isVertical:
        output = np.flip(np.flip(np.rot90(output, axes=(0,1)), axis=2), axis=0)
    return output

def UtilAugmStretch2d(height, width, ratio, centerPoint):
    yCenter, xCenter = centerPoint
    indDiff = UtilCartesianMatrix(range(height), range(width)) - [yCenter, xCenter]
    return indDiff / ratio + [yCenter, xCenter]

def UtilAugmRotate(height, width, angle, centerPoint):
    yCenter, xCenter = centerPoint
    indDiff = UtilCartesianMatrix(range(height), range(width)) - [yCenter, xCenter]
    # Remember, image is upside down by axis Y
    s,c = (math.sin(angle), math.cos(angle)) # Minus because we go from rotated matrix to the original one
    indY = indDiff[:,:,0] * c + indDiff[:,:,1] * s
    indX = indDiff[:,:,1] * c - indDiff[:,:,0] * s
    return np.stack([indY, indX], axis=2) + [yCenter, xCenter]


def UtilAugmReverseMapping(arrMap):
    """
    Maps original pixells into the new ones
    :param arrMap: mapping of the new pixels into teh original ones
    :return: matrix of [yReverseMapped, xreverseMapped]
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

def UtilAdjustBinaryMask(img):
    if len(img.shape) == 3:
        img = UtilFromRgbToGray(img)
    boolImg = img >= 128
    return np.where(boolImg, 255., 0.)

def UtilSaveBinaryMask(img, fileName):
    assert os.path.splitext(fileName)[1].lower() in AllowedBinaryMaskExt
    img = UtilAdjustBinaryMask(img)
    UtilArrayToImageFile(img, fileName)

def UtilLoadBinaryMask(fileName):
    assert os.path.splitext(fileName)[1].lower() in AllowedBinaryMaskExt
    img = UtilImageFileToArray(fileName)
    return UtilAdjustBinaryMask(img)

def UtilRemapBinaryMask(imgMask, map):
    """
    For a binary mask we should remove all splining
    """
    imgMask = UtilRemapImage(imgMask, map, fillValue = 127., ky=1, kx=1)
    return UtilAdjustBinaryMask(imgMask)
