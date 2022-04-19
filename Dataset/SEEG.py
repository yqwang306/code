# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 06.03.2022

from Dataset.Base import BaseDataset
from util.seeg_utils import get_all_file_path

import os

class SEEGData(BaseDataset):
    root = None # the root path of the dataset
    data_train = {} # processed train data that can be used directly
    data_test = {} # processed test data that can be used directly
    data_val = {} # processed val data that can be used directly
    tree_file = None # the full dirs of the dataset
    channel_info = None # information in terms of included channels

    def __init__(self, root):
        """
        directly laod in splitted npy data for training
        :param path_normal:
        :param path_cases:
        """

        self.root = root
        self.data_train = get_all_file_path(os.path.join(self.root, 'split/train'), 'npy')
        self.data_test = get_all_file_path(os.path.join(self.root, 'split/test'), 'npy')
        self.data_val = get_all_file_path(os.path.join(self.root, 'split/val'), 'npy')

    def get_dataset_tree(self):
        tree_file = {}  # 生成目录的结构树
        path = os.path.join(self.root, 'transform')
        tree_file["root_path"] = path
        for index, dirs in enumerate(os.listdir(path)):
            now_dir_name = {}
            path_dir = os.path.join(path, dirs)
            for name in os.listdir(path_dir):
                p = os.path.join(path_dir, name)
                patient_files = [p for p in os.listdir(p)]
                now_dir_name[name] = patient_files
            tree_file[dirs] = now_dir_name
        self.tree_file = tree_file
        return tree_file

    def get_statistic_info(self):

        normal_sleep = self.tree_file["sleep"]
        pre_seizure = self.tree_file["preseizure"]
        normal_number = {}
        preseizure_number = {}

        print("---" * 5 + "NORMAL_SLEEP INFORMATION" + "---" * 5)
        count = []
        for meta_data in normal_sleep.items():
            print("name :{}; data_number:{}".format(meta_data[0], len(meta_data[1])))
            count.append(len(meta_data[1]))
            normal_number[meta_data[0]] = len(meta_data[1])

        print("average number:{} ".format(sum(count) // len(count)))
        normal_number["average number"] = sum(count) // len(count)

        print("---" * 5 + "Pre_Seizure INFORMATION" + "---" * 5)
        count = []
        for meta_data in pre_seizure.items():
            print("name :{}; data_number:{}".format(meta_data[0], len(meta_data[1])))
            count.append(len(meta_data[1]))
            preseizure_number[meta_data[0]] = len(meta_data[1])
        print("average number:{} ".format(sum(count) // len(count)))
        preseizure_number["average number"] = sum(count) // len(count)
