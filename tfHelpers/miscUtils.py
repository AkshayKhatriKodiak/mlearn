# Miscellaneous utils to help DL
#
# Copyright (C) 2016-2017  Author: Misha Orel
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

from shared.pyutils.imageutils import *
from shared.pyutils.tensorutils import *
import matplotlib.pyplot as plt


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
