# Utilities for color augmentation

__author__ = 'Misha Orel'

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
    mapLS = np.rint(UtilReflectCoordTensor(mapLS)).clip(min=0, max=255).astype(np.uint8)

    mapH = UtilRandomSinFunc((181,), order=orderH, expectedStd=strengthH).reshape(-1,1) # Max value 180. Bug in CV2 ?
    mapH += UtilCartesianMatrixDefault(181)
    mapH = np.mod(np.rint(mapH).astype(np.uint8), 180)

    return (mapLS, mapH)

def UtilAugmConstRepaintHMap(strengthH=10):
    """
    Creates a repainting map of 181x1
    :param strenthH: shift of H coordinate
    :return:
    """
    mapH = UtilCartesianMatrixDefault(181) + strengthH # Max value 180. Bug in CV2 ?
    mapH = np.mod(np.rint(mapH).astype(np.uint8), 180)
    return mapH

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
        repaintMapH = repaintMap
        repaintMapLS = UtilCartesianMatrixDefault(256, 256).astype(np.uint8)
    img = cv2.cvtColor(UtilImageToInt(img), cv2.COLOR_BGR2HLS) # BGR because CV2 is using BGR order
    imgIndxsH, imgIndxsL, imgIndxsS = np.transpose(img.reshape(-1,3))
    repaintLS = repaintMapLS[imgIndxsL, imgIndxsS]
    img = np.stack([repaintMapH[imgIndxsH][:,0], repaintLS[:,0], repaintLS[:,1]], axis=1)
    return cv2.cvtColor(img.reshape((h,w,3)), cv2.COLOR_HLS2BGR).astype(np.float32)

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

