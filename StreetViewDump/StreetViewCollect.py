#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This app collects 512 x 512 driver's view tile images from Google
# Street View
#
# Usage:
# StreetViewCollect.py <lower> <upper> <left> <right> <max> <destDir> <start>
# where
# lower, upper, left, right - rectangle of interest, in lattitude / longtitude
# coordinates
# max - maximum number of files that will be downloaded, <= 100000
# destDir - destination directory
# start - starting file number
#
# Files are downloaded as 00001.jpg, 00002.jpg, etc

import sys
import os
import random
from Queue import Queue
import tempfile

from UrlRead import urlRead
from StreetViewXMLParse import streetViewXmlParse
from utils import UtilWrapper, UtilStitchImagesHor

GRID_SIZE = 0.0002  # In degrees
MIN_RECT = (2 * GRID_SIZE)
MAX_RECT = 3.0

XML_URL_BASE = "http://cbk0.google.com/cbk?output=xml"
TILE_URL_BASE = "http://cbk0.google.com/cbk?output=tile"

f1 = tempfile.NamedTemporaryFile(mode="wb", prefix="StreetViewCollect",
    delete=True)
f2 = tempfile.NamedTemporaryFile(mode="wb", prefix="StreetViewCollect",
    delete=True)
tempFileName1 = f1.name
tempFileName2 = f2.name
f1.close()
f2.close()
print("Temp files %s and %s" % (tempFileName1, tempFileName2))

def graceExit():
    try:
        os.remove(tempFileName1)
    except:
        pass
    try:
        os.remove(tempFileName2)
    except:
        pass
    sys.exit()

def usage():
    print "USAGE:"
    print """
StreetViewCollect.py <lower> <upper> <left> <right> <max> <destDir> <start>
where
lower, upper, left, right - rectangle of interest, in lattitude / longtitude
coordinates. Length and height of this rectangle should be no more than
%f degrees each, and more that %f degrees.
max - maximum number of files that will be downloaded, <= 100000
destDir - destination directory for JPEG files; must exist
start - starting file number
""" % (MAX_RECT, MIN_RECT)
    graceExit()

horCells = None
lower = None
upper = None
left = None
right = None
maxCount = None
destDir = None
fileCount = 0
grid = []

def gridElem(x, y):
    global grid, horCells
    try:
        obj = grid[y * horCells + x]
    except IndexError as e:
        obj = UtilWrapper(False)
    return obj


if len(sys.argv) != 8:
    usage()

try:
    lower = float(sys.argv[1])
    upper = float(sys.argv[2])
    if lower > upper:
        lower, upper = (upper, lower)

    left = float(sys.argv[3])
    right = float(sys.argv[4])
    if left > right:
        left, right = (right, left)

    maxCount = int(sys.argv[5])
    destDir = sys.argv[6]
    fileCount = int(sys.argv[7])

    if (right - left > MAX_RECT) or (upper - lower > MAX_RECT) or \
            (right - left < MIN_RECT) or (upper - lower < MIN_RECT):
        raise ValueError("Wrong rectangle size")

    if maxCount > 100000:
        raise ValueError("Maximum count is too high")

    if not os.path.isdir(destDir):
        raise ValueError("Directory %s does not exist" % destDir)

except ValueError as e:
    print e
    usage()

horSize = right - left
horCells = int(horSize / GRID_SIZE)
vertSize = upper - lower
vertCells = int(vertSize / GRID_SIZE)
print("Grid size %d X %d\n" % (horCells, vertCells))

for i in range(horCells * vertCells):
    grid.append(UtilWrapper(True))

queue = Queue(0)

while True:
    rx = random.uniform(left, right)
    ry = random.uniform(lower, upper)
    urlStr = XML_URL_BASE + "&ll=" + str(ry) + "," + str(rx)
    print("First URL: %s\n" % urlStr)
    xml = urlRead(urlStr)
    if not xml:
        continue
    tup = streetViewXmlParse(xml)
    if not tup:
        continue
    queue.put(tup.pano_id)
    break

seenPanoIds = set()

while not queue.empty():
    if fileCount >= maxCount:
        print("Downloaded %d files\n" % fileCount)
        graceExit()
    panoid = queue.get()
    print("Processing pano_id %s\n" % panoid)
    urlStr = XML_URL_BASE + "&panoid=" + panoid
    xml = urlRead(urlStr)
    if not xml:
        continue        
    tup = streetViewXmlParse(xml)
    if not tup:
        continue
    print("Parsed %s\n" % repr(tup))

    if (tup.longitude < left) or (tup.longitude > right) or \
        (tup.latitude < lower) or (tup.latitude > upper):
        print("Outside of rectangle\n")
        continue

    assert (tup.pano_id == panoid)
    for linkid in tup.linkids:
        if linkid not in seenPanoIds:
            seenPanoIds.add(linkid)
            queue.put(linkid)

    x = int((tup.longitude - left) / GRID_SIZE)
    y = int((tup.latitude - lower) / GRID_SIZE)
    print("Coord %d, %d     Queued panoids %d seen %d\n" % \
          (x, y, queue.qsize(), len(seenPanoIds)))
    if gridElem(x, y).value:
        for i in range(-1,2):
            for j in range(-1,2):
                gridElem(x+i,y+j).value = False

        fileName = destDir + ("/%05u.jpg" % fileCount)
        urlStr1 = TILE_URL_BASE + "&panoid=" + panoid + "&zoom=4&x=6&y=3"
        urlStr2 = TILE_URL_BASE + "&panoid=" + panoid + "&zoom=4&x=7&y=3"
        print("Saving to file %s\n" % fileName)
        if (urlRead(urlStr1, tempFileName1) is None) or \
            (urlRead(urlStr2, tempFileName2) is None):
            print("File read failed\n")
            continue
        UtilStitchImagesHor([tempFileName1, tempFileName2], fileName)
        fileCount += 1

graceExit()




