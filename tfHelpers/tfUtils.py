# Utils to help with Tensorflow interface

__author__ = "Misha Orel"

import tensorflow as tf

def tfUtilLoadGraph(fileName):

    with tf.gfile.GFile(fileName, "rb") as f:
        graphDef = tf.GraphDef()
        graphDef.ParseFromString(f.read())

    g = tf.Graph()
    with g.as_default():
        tf.import_graph_def(graphDef, name='')

    return g

def tfUtliDumpGraph(g):
    return g.get_operations()
