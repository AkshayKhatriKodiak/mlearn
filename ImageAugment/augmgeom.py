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
    return np.flip(arr, axis=1)

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
    output = np.empty((height, width, 2), dtype=np.float32)
    for j in range(height):
        output[j,:] = np.transpose([np.repeat((j - midCoord) / ratio + midCoord, width), \
                                    np.array(range(width), dtype=np.float32)])
    if not isVertical:
        output = np.flip(np.flip(np.rot90(output, axes=(0,1)), axis=2), axis=0)
    return output

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
            y,x = (arrMap[j,i] + 0.5).astype(np.int)
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
            y = int(y + 0.5)
            x = int(x + 0.5)
            tupleArr[j,i] = (y,x)
            filled.add((j,i))

    assert not np.any(np.vectorize(lambda x: x is None)(tupleArr))

    # Covert array of tuples to np array
    return np.dstack([np.vectorize(operator.itemgetter(i))(tupleArr) for i in (0,1)])

def UtilAdjustBinaryMask(img):
    if len(img.shape) == 3:
        img = UtilFromRgbToGray(img)
    boolImg = img >= 128
    return np.where(boolImg, 255, 0).astype(np.uint8)

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
    UtilArrayToImageFile(pyfgimg, "/home/morel/temp/haha1.bmp")
    UtilArrayToImageFile(maskImg, "/home/morel/temp/haha1mask.bmp")
    pybgimg=pybgimg.resize((480, 640), resample=Image.BICUBIC)
    print ("Equalizing")
    pybgimg = UtilImageEqualizeBrightness(pybgimg, pyfgimg, 10.)
    print ("Finished equalizing")
    UtilArrayToImageFile(pybgimg, "/home/morel/temp/komnata3.bmp")
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
    UtilArrayToImageFile(img, "/home/morel/temp/komnata4.bmp")

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
    UtilArrayToImageFile(img, "/home/morel/temp/komnata1.bmp")
    sys.exit(0)
    bgimgcv=CVImage(pybgimg)
    pyedge=bgimgcv.edge()
    UtilArrayToImageFile(pyedge, "/home/morel/temp/edge.bmp")
    sys.exit(0)
    img=CVImage(pyimg)
    #img.gaussian(0.5)
    #img.image(imageName="/home/morel/temp/haha1.jpg")
    img.edge()


