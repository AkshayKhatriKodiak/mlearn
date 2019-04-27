# Utils to help with Keras interface                                                                
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

# Convert Keras h5 frozen model to TF pb.

import os
import argparse

import tensorflow as tf

from keras import backend as K
from keras.models import load_model, model_from_json
from tensorflow.python.framework.graph_util import convert_variables_to_constants

def ParseArgs():                                                                                    
  parser = argparse.ArgumentParser(description='Storing H5 as PB.')                          
                                                                                                    
  parser.add_argument("-k",                                                                         
                      dest="h5",                                                                    
                      type=str,                                                                     
                      required=True,                                                                
                      help="Output H5 file")                                                        

  parser.add_argument("-p",                                                                         
                      dest="pb",                                                                    
                      type=str,                                                                     
                      required=True,                                                                
                      help="Output PB file")     
                                                                                                    
  return parser.parse_args()


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
  """
  Freezes the state of a session into a pruned computation graph.

  Creates a new computation graph where variable nodes are replaced by
  constants taking their current value in the session. The new graph will be
  pruned so subgraphs that are not necessary to compute the requested
  outputs are removed.
  @param session The TensorFlow session to be frozen.
  @param keep_var_names A list of variable names that should not be frozen,
                        or None to freeze all the variables in the graph.
  @param output_names Names of the relevant graph outputs.
  @param clear_devices Remove the device directives from the graph for better portability.
  @return The frozen graph definition.
  """

  graph = session.graph
  with graph.as_default():
    freeze_var_names = \
      list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
    output_names = output_names or []
    output_names += [v.op.name for v in tf.global_variables()]
    # Graph -> GraphDef ProtoBuf
    input_graph_def = graph.as_graph_def()
    if clear_devices:
      for node in input_graph_def.node:
        node.device = ""

  frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                output_names, freeze_var_names)
  return frozen_graph




def _main():
  args = ParseArgs()

  #K.set_learning_phase(0)

  model = load_model(args.h5)
  print("Outputs: %s" % str(model.outputs))
  print("Inputs: %s" % str(model.inputs))

  frozen_graph = freeze_session(K.get_session(),
    output_names=[out.op.name for out in model.outputs])

  folder, file_name = os.path.split(args.pb)
  assert(file_name[-3:] == ".pb")
  tf.train.write_graph(frozen_graph, folder, file_name, as_text=False)


if __name__ == "__main__":
  _main()
