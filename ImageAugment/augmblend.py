__author__ = "Misha Orel"

from mlearn.ImageAugment.augmgeom import *
from mlearn.ImageAugment.augmcolor import *

def UtilBlendForeBackGround(imgFg, imgFgMask, imgBg, yShift=0, xShift=0, blendBright=None, blendSatur=None, \
                            blurLayers=0):
    """
    Blends foreground image, limited by the binary mask, with the background
    :param imgFg:
    :param imgFgMask: greyscale image
    :param imgBg:
    :param yShift: horizontal shift of the foreground relative to the background
    :param xShift: vertical shift of the foreground relative to the background
    :param blendSigma: sharpness of the blending edge. If None, then it is taken random between 0.4 and 1.2 pixels
    :param blendBright: how much the brightness of background and foreground should be equalized
    :return:
    """
    assert len(imgFg.shape) == len(imgBg.shape) == 3
    assert imgFg.shape[:2] == imgBg.shape[:2] == imgFgMask.shape
    assert imgFg.shape[2] == imgBg.shape[2] == 3
    assert (np.max(imgFg) <= 255) and (np.max(imgBg) <= 255)
    assert (np.min(imgFg) >= 0) and (np.min(imgBg) >= 0)
    h,w,_ = imgFg.shape

    imgFgMask = UtilAdjustBinaryMaskToInt(imgFgMask)

    # TODO: sharpness equalization

    def _padWidth(shift):
        return (max(0, shift), max(0, -shift))

    yBefPad, yAftPad = _padWidth(yShift)
    xBefPad, xAftPad = _padWidth(xShift)

    imgFgMaskShifted = np.pad(imgFgMask, ((yBefPad, yAftPad), (xBefPad, xAftPad)), \
                          mode='constant', constant_values=0)[yAftPad:h + yAftPad, xAftPad:w + xAftPad]

    if blendBright is not None:
        # Brightness equalization
        imgFgShifted = np.pad(imgFg, ((yBefPad, yAftPad), (xBefPad, xAftPad), (0,0)), \
            mode='constant', constant_values=127.)[yAftPad:h+yAftPad, xAftPad:w+xAftPad, :]
        imgBg = UtilImageEqualizeBrightness(imgBg, imgFgShifted, blendBright=blendBright, kernelSize = w // 10)

    if blendSatur is not None:
        # Saturation equalization
        imgHls = cv2.cvtColor(UtilImageToInt(imgFg), cv2.COLOR_RGB2HLS)
        fgSat = np.mean(imgHls[:,:,2])
        imgHls = cv2.cvtColor(UtilImageToInt(imgBg), cv2.COLOR_RGB2HLS)
        bgSat = max(np.mean(imgHls[:,:,2]), 1.)
        mapLS = UtilAugmIncrSaturLSMap(fgSat / (bgSat * blendSatur + fgSat * (1. - blendSatur)))
        imgBg = UtilAugmRepaintHLS(imgBg, mapLS)


    # Blending
    pilImgBg = Image.fromarray(UtilImageToInt(imgBg), mode="RGB")
    pilImgFg = Image.fromarray(UtilImageToInt(imgFg), mode="RGB")
    pilImgFgMask = Image.fromarray(UtilImageToInt(imgFgMask), mode="L")
    pilImgBg.paste(pilImgFg, (xShift, yShift), mask = pilImgFgMask)
    imgBlend = np.asarray(pilImgBg, dtype=np.float32)

    imgBlend = UtilImageEqualizeBorder(imgBlend, imgFgMaskShifted, blurLayers=blurLayers)

    return (imgBlend, imgFgMaskShifted)


def UtilImageEqualizeBorder(img, imgMask, blurLayers=0):
    # To do blurring on the border

    blurList = []
    blurImg = img
    for i in range(blurLayers):
        paddedImg = np.pad(blurImg, ((1,1), (1,1), (0,0)), mode='reflect')
        blurImg = (blurImg + np.sum(UtilGenerate4WayShifted(paddedImg), axis=3)) / 5.
        blurList.append(blurImg)
    blurList = list(reversed(blurList))

    boundaries = UtilRegionBoundaries(imgMask, levelCount=blurLayers)

    for i, blurredImg in enumerate(blurList):
        cond = np.stack([boundaries == (i+1)] * 3, axis=-1)
        img = np.where(cond, blurredImg, img)

    return img


def UtilImageEqualizeBrightness(imgDst, imgSrc, blendBright=1., kernelSize=15):
    """
    Makes brightness of imgDst equal to the brightness of imgSrc, averaged over gaussian kernel
    :param imgDst: destination image
    :param imgSrc: source image
    :param kernelSize: size of the gaussian Kernel
    :return: image with equalized brightness
    """
    brDst = UtilFromRgbToGray(imgDst)
    brSrc = UtilFromRgbToGray(imgSrc)
    brDstFilt = scipyFilters.gaussian_filter(brDst, sigma=kernelSize).clip(min=1.0)
    brSrcFilt = scipyFilters.gaussian_filter(brSrc, sigma=kernelSize).clip(min=1.0)
    ratio = brSrcFilt * np.reciprocal(brDstFilt * blendBright + brSrcFilt * (1. - blendBright))
    # Smoothly limit ratio to [1/e, e]
    ratio = np.exp(np.tanh(np.log(ratio)))
    # Find maximum posible value of ratio, so that the destination color does not change
    maxRatio = 255. * np.reciprocal(np.max(imgDst, axis=2).clip(min=1.))
    ratio = np.where(ratio > maxRatio, maxRatio, ratio).reshape(ratio.shape + (1,))
    imgDst = np.multiply(imgDst, ratio).clip(min=0., max=255.)
    return imgDst



