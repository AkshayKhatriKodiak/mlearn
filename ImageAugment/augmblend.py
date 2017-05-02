__author__ = "Misha Orel"

from mlearn.ImageAugment.augmgeom import *

def UtilBlendForeBackGround(imgFg, imgFgMask, imgBg, yShift=0, xShift=0, blendSigma=None):
    """
    Blends foreground image, limited by the binary mask, with the background
    :param imgFg:
    :param imgFgMask: greyscale image
    :param imgBg:
    :param yShift: horizontal shift of the foreground relative to the background
    :param xShift: vertical shift of the foreground relative to the background
    :param blendSigma: sharpness of the blending edge. If None, then it is taken random between 0.4 and 1.2 pixels
    :return:
    """
    assert len(imgFg.shape) == len(imgBg.shape) == 3
    assert imgFg.shape[:2] == imgBg.shape[:2] == imgFgMask.shape
    assert imgFg.shape[2] == imgBg.shape[2] == 3
    assert (np.max(imgFg) <= 255) and (np.max(imgBg) <= 255)
    assert (np.min(imgFg) >= 0) and (np.min(imgBg) >= 0)
    h,w,_ = imgFg.shape

    # TODO: sharpness equalization
    # TODO: hue equalization

    # Brightness equalization
    def padWidth(shift):
        return (max(0, shift), max(0, -shift))
    yBefPad, yAftPad = padWidth(yShift)
    xBefPad, xAftPad = padWidth(xShift)
    imgFgShifted = np.pad(imgFg, ((yBefPad, yAftPad), (xBefPad, xAftPad), (0,0)), \
        mode='constant', constant_values=127.)[yAftPad:h+yAftPad, xAftPad:w+xAftPad, :]
    imgBg = UtilImageEqualizeBrightness(imgBg, imgFgShifted, kernelSize = w // 15)

    # Blending
    # TODO: Do it right, at the border
    if 0:
        if blendSigma is None:
            blendSigma = np.random.uniform(0.4, 1.2)
        imgFg = np.dstack([scipyFilters.gaussian_filter(imgFg[:,:,i], sigma=blendSigma) for i in range(3)])
        imgBg = np.dstack([scipyFilters.gaussian_filter(imgBg[:,:,i], sigma=blendSigma) for i in range(3)])
        imgFgMask = scipyFilters.gaussian_filter(imgFgMask, sigma=blendSigma)
    pilImgBg = Image.fromarray(UtilImageToUint8(imgBg), mode="RGB")
    pilImgFg = Image.fromarray(UtilImageToUint8(imgFg), mode="RGB")
    pilImgFgMask = Image.fromarray(UtilImageToUint8(imgFgMask), mode="L")
    pilImgBg.paste(pilImgFg, (xShift, yShift), mask = pilImgFgMask)
    imgBlend = np.asarray(pilImgBg, dtype=np.float32)
    # TODO: Do it right, see above
    imgBlend = np.dstack([scipyFilters.gaussian_filter(imgBlend[:,:,i], sigma=1.0) for i in range(3)])

    return imgBlend



def UtilImageEqualizeBrightness(imgDst, imgSrc, kernelSize=12):
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
    ratio = brSrcFilt * np.reciprocal(brDstFilt)
    # Smoothly limit ratio to [1/e, e]
    ratio = np.exp(np.tanh(np.log(ratio)))
    # Find maximum posible value of ratio, so that the destination color does not change
    maxRatio = 255. * np.reciprocal(np.max(imgDst, axis=2).clip(min=1.))
    ratio = np.where(ratio > maxRatio, maxRatio, ratio).reshape(ratio.shape + (1,))
    imgDst = np.multiply(imgDst, ratio).clip(min=0., max=255.)
    return imgDst



