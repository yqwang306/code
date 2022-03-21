# -*- coding: utf-8 -*-
# Author: Yiqiao Wang
# Date: 27.02.2022

from DataProcessing.dataset.SpindleData import SpindleData
from DataProcessing.PreProcess.Base import BasePreProcessor
from util.eeg_utils import *
import os
import pandas as pd


class EEGPreProcessor(BasePreProcessor):

    def filter_noise(self, filter_length):
        """
        :param filter_length: minimum threshold that will be filtered out
        :return: labels and paths after filtering
        """

        spindle = SpindleData(self.data_path)
        labels = spindle.labels
        paths = spindle.paths
        del_list = []

        for i, p in enumerate(paths):
            if self.data_path == "../../dataset/mesa_dataset/":
                data = pd.read_csv(p, sep=",")
            else:
                data = pd.read_csv(p, skiprows=(0, 1), sep=",")
            if data.__len__() < filter_length:
                del_list.append(i)
                print("过滤掉了第%d个文件!" % (i + 1))

        labels = [x for i, x in enumerate(labels) if i not in del_list]
        paths = [x.split("\\")[-1] for i, x in enumerate(paths) if i not in del_list]
        spindle.labels = labels
        spindle.paths = paths
        return labels, paths

    def bi_class_handle(self):
        """
        Perform pre-processing for bi class data
        :return:
        """
        print("call bi_class_handle() from EEGPreProcessor")
        pass

    def multi_class_handle(self):
        """
        Perform pre-processing for multi class data
        :return:
        """
        print("call multi_class_handle() from EEGPreProcessor")
        pass
