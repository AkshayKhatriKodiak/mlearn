# Utils to help with Tensorflow interface
#
# Copyright (C) 2016-2017  Author: Misha Orel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.


from shared.pyutils.utils import *
from shared.pyutils.tensorutils import *
_minTfVersion = '1.4'
import tensorflow as tf
if forwardCompat.VersionCompare(tf.__version__, _minTfVersion) < 0:
    raise ValueError('Tensorflow too old: required %s, actual %s' % (_minTfVersion, tf.__version__))

def tfUtilLoadGraph(fileName, prefix=''):

    with tf.gfile.GFile(fileName, "rb") as f:
        graphDef = tf.GraphDef()
        graphDef.ParseFromString(f.read())

    g = tf.Graph()
    with g.as_default():
        tf.import_graph_def(graphDef, name=prefix)

    return g


def tfUtliDumpGraph(g):
    return g.get_operations()


def tfGetTrainableCount(scope=None):
    ret = 0
    for var in tf.trainable_variables(scope=scope):
        ret += functools.reduce(lambda x, y: x * y, var.get_shape())
    return ret


def tfUtilCentroids(imgs, axes):
    """
    Calculates centroid values for a set of 2D maps
    :param imgs: tensor, containing 2D maps among its dimensions
    :param axes: 2-tuple indicating which dimensions correpond to the height and width of the 2D maps
    :return: Array of centroids, with number of dimen
    """
    shape = tf.shape(imgs)
    shapeLen = shape.get_shape()[0].value
    tileShapeList = [shape[i] if i not in axes else 1 for i in range(shapeLen)]
    meshReshapeList = [shape[i] if i in axes else 1 for i in range(shapeLen)]
    height = shape[axes[0]]
    width = shape[axes[1]]

    yMesh, xMesh = (tf.reshape(tf.cast(x, tf.float32), meshReshapeList) \
                    for x in tf.meshgrid(tf.range(height), tf.range(width)))
    yMesh = tf.tile(yMesh, tileShapeList)
    xMesh = tf.tile(xMesh, tileShapeList)

    m0 = tf.reduce_sum(imgs, axis=axes) + UtilNumpyClippingValue(np.float32)
    return tf.reduce_sum(yMesh * imgs, axis=axes) / m0, tf.reduce_sum(xMesh * imgs, axis=axes) / m0
