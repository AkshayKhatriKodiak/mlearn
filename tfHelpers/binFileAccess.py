# Classes to write, and randomly read, from a bin file containing training records
#
# Copyright (C) 2017  Author: Misha Orel
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


# Unfortunately, if we use Tensorflow Queues mechanism, we have to give up feed_dict interface.
# This file allows you to load random training entries at real time, while keeping feed_dict.
# It also allows to do boosting, by assigning different probabilities for loading different entries



__author__ = "Misha Orel"

import sys
import numpy as np
from multiprocessing import Array, Process, cpu_count
import psutil
import functools
import time
from shared.pyutils.utils import *
from shared.pyutils.tensorutils import *

def _worker(fileName, offset, entrySize, entryCount, array, weightsFileName=None):
    entriesToRead = len(array) // entrySize
    readingArray = np.frombuffer(array, dtype=np.uint8)
    assert entriesToRead <= entryCount
    if weightsFileName is None:
        weights = None
    else:
        try:
            with open(weightsFileName, 'rb') as fweights:
                fweights.seek(np.dtype(np.float32).itemsize * offset // entrySize, os.SEEK_SET)
                weights = np.fromfile(fweights, dtype=np.float32, count = entryCount)
                # Normalize them
                weights = weights / np.sum(weights)
        except IOError as ex:
            print('NOT LOADING WEIGHTS, got %s' % str(ex))
            weights = None

    # See which entries to take
    entriesToTake = sorted(list(np.random.choice(range(entryCount), size=entriesToRead, replace=False, p=weights)))
    with open(fileName, 'rb') as finput:
        finput.seek(offset, os.SEEK_SET)
        prevFileIndex = 0
        arrayOffset = 0
        for fileIndex in entriesToTake:
            finput.seek((fileIndex - prevFileIndex) * entrySize, os.SEEK_CUR)
            prevFileIndex = fileIndex + 1
            readingArray[arrayOffset : arrayOffset + entrySize] = \
                np.fromfile(finput, dtype=np.uint8, count = entrySize)
            arrayOffset += entrySize



class BinFileWriter(UtilObject):
    """
    Writer of numpy ndarrays into a binary file as fixed size entries
    """

    def __init__(self, fileName):
        """
        Creates an object that will write to fileName
        :param fileNa:
        :param typeShapeList: list of tuples (np.type, shape) for all numpy objects in one entry
        """
        self.fileName = fileName

    def open(self):
        self.fd = open(self.fileName, 'wb')
        self.count = 0

    def close(self):
        self.fd.close()
        self.fd = None

    def newEntry(self, arrList):
        for arr in arrList:
            arr.tofile(self.fd)


class BinFileParRandBuffer(UtilObject):
    """
    A helper class for BinFileReader. Implements a buffer, and worker process invocation that fills the buffer
    """

    def __init__(self, fileName, bufferEntryCount, bufferCount, entrySize, weightsFileName=None, bufferSizeRampUp=0):
        self.fileName = fileName
        self.weightsFileName = weightsFileName
        self.entrySize = entrySize
        self.targetBufferEntryCount = bufferEntryCount
        self.bufferEntryCount = None
        self.bufferCount = bufferCount
        self.fileSize = os.path.getsize(fileName)
        self.buffers = None
        self.memViews = None
        self.empty = True
        self.rampUpCounter = bufferSizeRampUp
        self.inProgress = False

    def fetchFromFile(self):
        assert self.empty and (not self.inProgress)

        # Allocate arrays if needed
        if self.rampUpCounter >= 0:
            self.bufferEntryCount = self.targetBufferEntryCount // (1 << self.rampUpCounter)
            self.buffers = [Array('B', self.bufferEntryCount * self.entrySize, lock=False)
                            for _ in range(self.bufferCount)]
            self.memViews = [memoryview(b) for b in self.buffers]
            self.rampUpCounter -= 1

        self.empty = False
        self.inProgress = True
        self.readBufPos = 0
        entryCount = self.fileSize // self.bufferCount // self.entrySize
        argDict = {'fileName':self.fileName, 'entrySize':self.entrySize, 'entryCount':entryCount}
        if self.weightsFileName is not None:
            argDict['weightsFileName'] = self.weightsFileName
        self.procs = [Process(target = _worker, kwargs = UtilMergeDicts({'offset':entryCount * self.entrySize * i, \
            'array':self.buffers[i]}, argDict)) for i in range(self.bufferCount)]
        for p in self.procs:
            p.start()
        self.readSeq = np.random.permutation(self.bufferEntryCount * self.bufferCount)

    def wait(self):
        assert (not self.empty) and self.inProgress
        for p in self.procs:
            p.join()
        retCodes = np.array([p.exitcode for p in self.procs])
        if not np.all(retCodes == 0):
            raise IOError("Some worker processes failed: %s" % str(retCodes))
        self.inProgress = False

    def read(self):
        index = self.readSeq[self.readBufPos]
        self.readBufPos += 1
        if self.readBufPos == (self.bufferEntryCount * self.bufferCount):
            self.empty = True
        startPosInBuffer = (index % self.bufferEntryCount) * self.entrySize
        return (self.memViews[index // self.bufferEntryCount])[startPosInBuffer : startPosInBuffer + self.entrySize]


class BinFileParRandReader(UtilObject):
    """
    Reads random fixed size entries from a file, passes them back to the caller. The idea is pretty much the same
    as with FixedLengthRecordReader in Tensorflow, but here we are аctually getting the samples, and can look at them
    or modify them.
    """

    minBufferEntryCount = 32
    assert minBufferEntryCount & (minBufferEntryCount - 1) == 0

    def __init__(self, fileName, typeShapeList, batchSize, weightsFileName=None, procCount=None, mem=None):
        self.fileSize = os.path.getsize(fileName)
        if procCount is None:
            procCount = cpu_count()

        self.batchSize = batchSize
        self.typeShapeList = typeShapeList
        self.itemSizes = [UtilNumpyEntryItemSize(t) for t in typeShapeList]
        entrySize = sum(self.itemSizes)

        # Let's grab memory
        fourGig = 0x100000000
        memorySize = fourGig if mem is None else mem
        memorySize = min(memorySize, self.fileSize)

        while True:
            # Should not take more than half of the whole memory
            if memorySize >= psutil.virtual_memory().available // 2:
                raise MemoryError('Not enough available physical memory to get %u' % memorySize)
            bufferEntryCount = memorySize // 2 // procCount // entrySize // batchSize * batchSize
            if bufferEntryCount >= self.minBufferEntryCount:
                break
            if memorySize >= self.fileSize:
                if procCount > 1:
                    procCount //= 2
                else:
                    raise IOError('File %s got too few entries' % fileName)
            else:
                memorySize *= 2

        bufferSizeRampUp = bufferEntryCount.bit_length() - BinFileParRandReader.minBufferEntryCount.bit_length()

        self.buffers = [BinFileParRandBuffer(fileName, entrySize=entrySize, bufferEntryCount=bufferEntryCount, \
            bufferCount=procCount, weightsFileName=weightsFileName, bufferSizeRampUp=bufferSizeRampUp) for _ in (0, 1)]
        self.currentBuffer = 1 # We start with the "other" buffer so we are never fetching 2 buffers at the same time
        self.buffers[0].fetchFromFile()

    def batch(self):
        """
        :return: batch of numpy ndarrays with teh same dtype and shape as been defined in typeShapeList
        """
        currentBuffer = self.buffers[self.currentBuffer]
        otherBuffer = self.buffers[1 - self.currentBuffer]
        if currentBuffer.empty:
            self.currentBuffer = 1 - self.currentBuffer
            currentBuffer, otherBuffer = (otherBuffer, currentBuffer)
            assert not currentBuffer.empty

        if currentBuffer.inProgress:
            currentBuffer.wait()

        if otherBuffer.empty:
            otherBuffer.fetchFromFile()

        l = []
        for _ in range(self.batchSize):
            mem = currentBuffer.read()
            offset = 0
            n = []
            for i, isize in enumerate(self.itemSizes):
                dtype, shape = self.typeShapeList[i]
                n.append(np.frombuffer(mem[offset:offset+isize], dtype=dtype).reshape(shape))
                offset += isize
            l.append(n)

        return l

    def sync(self):
        """
        Finish all asynchroneous operations
        :return:
        """
        for b in self.buffers:
            b.wait()


class BinFileSimpleReader(UtilObject):
    """
    Reads fixed size entries from a file in sequntial order, passes them back to the caller. Has to be reset to start
    from the beginning
    """
    def __init__(self, fileName, typeShapeList, batchSize, batchesAtOnce=1):
        # batchesAtOnce - the total number of batches must be divisible by this number
        self.fileName = fileName
        self.fileSize = os.path.getsize(fileName)
        self.batchSize = batchSize
        self.typeShapeList = typeShapeList
        entrySize = UtilNumpyEntriesSize(typeShapeList)
        self.maxCount = self.fileSize // (entrySize * batchSize) // batchesAtOnce * batchesAtOnce
        if self.maxCount == 0:
            raise ValueError('This file size %u cannot be broken in %d chuncks of batches of size %d' % \
                             (self.fileSize, batchesAtOnce, batchSize))
        self.fd = None

    def open(self):
        self.close()
        self.fd = open(self.fileName, 'rb')
        self.count = 0

    def close(self):
        if self.fd is not None:
            self.fd.close()
            self.fd = None

    def eof(self):
        return (self.count >= self.maxCount)

    def batch(self):
        if self.count >= self.maxCount:
            self.close()
            return None
        assert self.fd is not None
        self.count += 1
        l = []
        for _ in range(self.batchSize):
            n = []
            for dtype, shape in self.typeShapeList:
                n.append(np.fromfile(self.fd, dtype=dtype, \
                    count=functools.reduce(lambda x, y: x*y, list(shape))).reshape(shape))
            l.append(n)
        return l

