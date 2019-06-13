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

import numpy as np


class PermutRandFileInput:
    """
    This class reads from several text files, number of lines in specified proportions.
    Then it permutates the lines, and presents it to a reader.
    """

    def __init__(self, txt_file_list, proportions_list):
        """
        :param txt_file_list: list of input text files
        :param proportions_list: proportions in which we want to load those files
        """

        self.file_dict_ = {}
        line_count = 0
        for ind, fn in enumerate(txt_file_list):
            with open(fn, "r") as fin:
                lines = np.array(fin.readlines())
            select_count = int(len(lines) * proportions_list[ind])
            line_count += select_count
            self.file_dict_[fn] = (lines, select_count)
        self.line_count_ = line_count

    def line_count(self):
        return self.line_count_

    def get_permuted(self):
        array_list = []
        for lines, select_count in self.file_dict_.values():
            length = len(lines)
            array_list += ([lines] * (select_count // length))
            array_list.append(np.random.choice(lines, size=(select_count % length), replace=False))
        total_lines = np.concatenate(array_list)
        np.random.shuffle(total_lines)
        return total_lines


if __name__ == "__main__":
    # test
    prfi = PermutRandFileInput(["test1.txt", "test2.txt"],
                               [0.6, 2.4])
    print("LINES:\n %s" % str(prfi.get_permuted())) 
