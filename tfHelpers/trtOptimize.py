# TRT optimizing Tensorflow frozen graph.
# Inspired by 
# https://github.com/ardianumam/Tensorflow-TensorRT/blob/master/7_optimizing_YOLOv3_using_TensorRT.ipynb                                             
#                                                                                                   
# Copyright (C) 2019  Author: Misha Orel                                                            
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

# Import the needed libraries
import cv2
import time
import arggparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile
from PIL import Image


def ParseArgs():                                                                                    
  parser = argparse.ArgumentParser(description='Optimizing TF model with TRT.')
                                                                                                    
  parser.add_argument("-i",                                                                         
                      dest="input",
                      type=str,                                                                     
                      required=True,                                                                
                      help="Name of the input PB file")

  parser.add_argument("-o",                                                                         
                      dest="output",                                                                 
                      type=str,                                                                     
                      required=True,                                                                
                      help="Name of the output PB file")      
                                                                                                    
  parser.add_argument("-n",                                                                         
                      dest="output_names",
                      type=str,                                                                     
                      required=True,                                                                
                      nargs="+",
                      help="Output names")                                                        

  parser.add_argument("-p",                                                                         
                      dest="precision_mode",
                      type=str,                                                                     
                      default="FP32",                                                                
                      help="Precision mode, FP16 or FP32")      

  parser.add_argument("-b",                                                                         
                      dest="max_batch_size",                                                        
                      type=int,                                                                     
                      default=1,                                                               
                      help="Maximum inference batch size")     

  parser.add_argument("-w",                                                                         
                      dest="workspace",                                                        
                      type=int,                                                                     
                      default=4,                                                               
                      help="Workspace in GB, default is 4")     
                                                                                                    
  return parser.parse_args()


# function to read a ".pb" model 
# (can be used to read frozen model or TensorRT model)
def read_pb_graph(model):
  with gfile.FastGFile(model,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


def _main():
  args = ParseArgs()         
  frozen_graph = read_pb_graph(args.input)

  # convert (optimize) frozen model to TensorRT model
  trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=args.output_names,
    max_batch_size=args.max_batch_size,
    max_workspace_size_bytes=args.workspace*(10**9),
    precision_mode=args.precision_mode)

  #write the TensorRT model to be used later for inference
  with gfile.FastGFile(args.output, 'wb') as f:
    f.write(trt_graph.SerializeToString())
  print("TensorRT model is successfully stored!")

  # check how many ops of the original frozen model
  all_nodes = len([1 for n in frozen_graph.node])
  print("numb. of all_nodes in frozen graph:", all_nodes)

  # check how many ops that is converted to TensorRT engine
  trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
  print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
  all_nodes = len([1 for n in trt_graph.node])
  print("numb. of all_nodes in TensorRT graph:", all_nodes)


if __name__ == "__main__":                                                                          
  _main()
