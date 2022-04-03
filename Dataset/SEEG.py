# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 06.03.2022

from Dataset.Base import BaseDataset
from util.seeg_utils import get_all_file_path

import numpy as np
import os

class SEEGData(BaseDataset):
    data_normal = None # processed normal data that can be used directly
    data_cases = None # processed preseizure data that can be used directly
    channel_num = None # the no.channels

    def __init__(self, root, path_normal, path_cases):
        """
        directly laod in splitted npy data for training
        :param path_normal:
        :param path_cases:
        """

        map_cases = get_all_file_path(self.path_cases, 'npy')
        map_normal = get_all_file_path(self.path_normal, 'npy')
        data_map_cases = {}
        for d_map in map_cases.items():
            data = [np.load(x) for x in d_map[1]]
            data_map_cases[d_map[0]] = data
        data_map_normal = {}
        for d_map in map_normal.items():
            data = [np.load(x) for x in d_map[1]]
            data_map_normal[d_map[0]] = data

        self.root = root
        self.channel_num = data_map_cases[list(data_map_cases.keys())[0]][0].shape[0]
        self.data_normal = data_map_normal
        self.data_cases = data_map_cases

    def get_all_path_by_keyword(self, keyword):
        name_dir = os.listdir(self.path_dir)
        if keyword in name_dir:
            temp_path = os.path.join(self.path_dir, keyword)
            path_all = get_all_file_path(temp_path, 'npy')
            return path_all
        else:
            print("please check your keyword, this keyword doesn't exist!")
            return None
