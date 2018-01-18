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

from shared.pyutils.utils import *
from shared.pyutils.tensorutils import *


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