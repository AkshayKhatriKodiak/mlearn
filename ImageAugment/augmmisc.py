__author__ = "Misha Orel"

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





