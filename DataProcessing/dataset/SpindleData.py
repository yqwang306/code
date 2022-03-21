# -*- coding: utf-8 -*-
# Author: Yiqiao Wang
# Date: 2022/3/9

import os
import glob
import pandas as pd
import numpy as np
import math


class SpindleData:
    """:class:`SpindleData` stores the paths of original sleep data and corresponding labels in memory.

        Attributes:

            dataset_path (str): Local file path of this dataset.

            paths (list): List of all files in the file path.

            labels (list): Corresponding labels of all files in the file path.

    """

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.paths, self.labels = self.get_paths_and_labels()

    def get_paths_and_labels(self):
        """
        :return: The path and label of the data to be fetched.
        """

        path = self.dataset_path
        cate = [(os.path.join(path, x)) for x in os.listdir(path)]
        paths = []
        labels = []
        for i, p in enumerate(cate):
            path_tmps = glob.glob(os.path.join(p, "*.csv"))
            for p in path_tmps:
                paths.append(p)
                labels.append(i)
        labels = np.asarray(labels)
        return paths, labels

    @classmethod
    def trans_list_str(self, list_a):
        """
        convert array to string

        :param list_a: the array to be converted
        :return: converted string
        """

        str_a = ""
        for a in list_a:
            str_a += str(a)
        return str_a


















