#TODO: temporary stuff
from shared.pyutils.imageutils import *


def UtilAugmCircleMapping(boundRect,center,height,width):
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
            y = math.tan(phi) * (center + xMax) * objHeight / yMax + yMax
            x = r - center
            arr[j][i] = np.array([y,x])
    return arr


def UtilAugmReverseMapping(arrMap):
    """
    Maps original pixells into the new ones
    :param arrMap: mapping of the new pixels into teh original ones
    :return: matrix of tuples (yReverseMapped, xreverseMapped)
    """
    height, width, coordCount = arrMap.shape
    assert coordCount == 2
    yValues, xValues = (np.array(a, dtype=np.float32) for a in \
        np.unravel_index(range(height * width), (height, width)))
    arrMapValues = np.reshape(arrMap, (height * width, 2))
    fy = interpolate.interp2d(arrMapValues[:,0], arrMapValues[:,1], yValues, fill_value=-1.)
    fx = interpolate.interp2d(arrMapValues[:,0], arrMapValues[:,1], xValues, fill_value=-1.)
    # Create arrays of indices


def justatemp(name, blur):
    pybgimg=Image.open("/home/morel/temp/komnata.jpg")
    pyfgimg=Image.open("/home/morel/temp/" + name + ".jpg")
    maskImg = Image.open("/home/morel/temp/" + name + ".png")
    maskImg = maskImg.convert(mode = "L")
    maskImgBin = maskImg.point(lambda p: p > 128 and 255)
    boundRect = UtilBoundingRectFromMask(maskImgBin)
    print ("boundRect %s" % str(boundRect))
    w,h=pyfgimg.size
    arrMap = UtilAugmentCircleMapping(boundRect,500.,h,w)
    pyfgimg = UtilRemapImage(pyfgimg, arrMap)
    maskImg = UtilRemapImage(maskImg, arrMap)
    pyfgimg.save("/home/morel/temp/haha1.bmp")
    maskImg.save("/home/morel/temp/haha1mask.bmp")
    pybgimg=pybgimg.resize((480, 640), resample=Image.BICUBIC)
    print ("Equalizing")
    pybgimg = UtilImageEqualizeBrightness(pybgimg, pyfgimg, 10.)
    print ("Finished equalizing")
    pybgimg.save("/home/morel/temp/komnata3.bmp")
    cvbgimg=CVImage(pybgimg)
    cvfgimg=CVImage(pyfgimg)
    kernel = 0.0
    cvbgimg.gaussian(blur)
    while False: #JUSTATEMP
        kernel += 0.1
        cvbgimg.edge()
        cvfgimg.edge()
        print ("before %f %f" % (cvbgimg.meanSharpness(), cvfgimg.meanSharpness()))
        diffSharp =  cvbgimg.meanSharpness() - cvfgimg.meanSharpness()
        cvbgimgSharp = CVImage(cvbgimg)
        cvfgimgSharp = CVImage(cvfgimg)
        if diffSharp < 0.:
            cvfgimgSharp.gaussian(kernel)
        else:
            cvbgimgSharp.gaussian(kernel)
        cvfgimgSharp.edge()
        cvbgimgSharp.edge()
        print ("after %f %f %f" % (kernel, cvbgimgSharp.meanSharpness(), cvfgimgSharp.meanSharpness()))
        newDiffSharp = cvbgimgSharp.meanSharpness() - cvfgimgSharp.meanSharpness()
        if newDiffSharp * diffSharp <= 0.:
            break
    #cvbgimg = CVImage(cvbgimgSharp)
    #cvfgimg = CVImage(cvfgimgSharp)
    cvbgimg.edge()
    cvfgimg.edge()
    print ("%f %f" % (cvbgimg.meanSharpness(), cvfgimg.meanSharpness()))
    pybgimg = cvbgimg.image("/home/morel/temp/komnata2.bmp")
    imgan=ImageAnnot(pyfgimg)
    mask = imgan.setTransparencyMask("/home/morel/temp/haha1mask.bmp", binarizeThreshold=128)
    pyimgpaste=imgan.save("/home/morel/temp/haha1.png")
    img = UtilImageSimpleBlend(pybgimg, pyimgpaste)
    img.save("/home/morel/temp/komnata4.bmp")

#JUSTATEMP
if 0:
    #justatemp("hehe", 0.8)
    #justatemp("uhuh", 0.9)
    #justatemp("haha", 0.7)
    justatemp("huhu", 0.3)
    sys.exit(0)

    #cvbgimg.gaussian(0.5)
    cvbgimg.edge()
    pybgimg=Image.open("/home/morel/temp/komnata3.bmp")
    pybgimg=pybgimg.resize((480, 640), resample=Image.BICUBIC)
    #pybgimg.save("/home/morel/temp/komnata1.jpg")
    imgan=ImageAnnot(pyfgimg)
    mask = imgan.setTransparencyMask("/home/morel/temp/haha.png", binarizeThreshold=128)
    print ("RECT %s" % str(UtilBoundingRectFromMask(mask)))
    imgan.transpImage.save("/home/morel/temp/haha1.bmp")
    pyimgpaste=imgan.save("/home/morel/temp/haha1.png")
    img = UtilImageSimpleBlend(pybgimg, pyimgpaste)
    img.save("/home/morel/temp/komnata1.bmp")
    sys.exit(0)
    bgimgcv=CVImage(pybgimg)
    pyedge=bgimgcv.edge()
    pyedge.save("/home/morel/temp/edge.bmp")
    sys.exit(0)
    img=CVImage(pyimg)
    #img.gaussian(0.5)
    #img.image(imageName="/home/morel/temp/haha1.jpg")
    img.edge()