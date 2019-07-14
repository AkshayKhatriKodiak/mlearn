# Miscellaneous utils to help DL
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

from shared.pyutils.imageutils import *
from shared.pyutils.tensorutils import *
import matplotlib.pyplot as plt
import tensorflow as tf


def UtilTimeSeqRandomSelect(len, minSelect, count):
    """
    Selects random numbor of events, not more than count, not less that minSelect, dsitributed by Poisson
    If too few entries left, teh whole count is consumed
    :param len:
    :param minSelect:
    :param count: period of exponential decay
    :return:
    """
    assert len >= minSelect
    assert count > minSelect

    if len <= count // 2:
        return len
    if len <= 2 * minSelect:
        return len

    while True:
        n = int(np.rint(np.random.standard_exponential((1,)) * count))
        if (n >= minSelect) and (n < len):
            break

    if (len - n) < minSelect:
        n = len

    return n


def UtilDisplayAccuracyLoss(accuracy, loss, averaging=None):
    accuracy = np.array(accuracy)
    loss = np.array(loss)
    if averaging is not None:
        accuracy = scipyFilters.gaussian_filter1d(accuracy, sigma=averaging, axis=0)
        loss = scipyFilters.gaussian_filter1d(loss, sigma=averaging, axis=0)

    accuracy = accuracy / np.min(accuracy).clip(min=UtilNumpyClippingValue(accuracy.dtype))
    loss = loss / np.min(loss).clip(min=UtilNumpyClippingValue(loss.dtype))

    accuracy = np.log(accuracy)
    loss = np.log(loss)

    plt.plot(range(loss.shape[0]), loss, 'gs', range(accuracy.shape[0]), accuracy, 'ro')
    plt.show()


def TfPrint(tensor, *args, message="", summarize=None, output_stream=sys.stderr):
    kwargs = {"output_stream": output_stream}
    if summarize is not None:
        kwargs["summarize"] = summarize
    print_op = tf.print(*(("\n" + message,) + args), ** kwargs)

    # Create dummy dependency on print_op
    with tf.control_dependencies([print_op]):
        tensor = tf.identity(tensor, name="TfPrint_dummy")
    return tensor
