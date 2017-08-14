# Utils to help with Tensorflow interface
#
# Copyright (C) 2016-2017  Author: Misha Orel
#
# # This program is free software: you can redistribute it and/or modify
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

import tensorflow as tf
from shared.pyutils.utils import *

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
