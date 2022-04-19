# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 09.01.2022

""" A Basic Preprocessing Pipeline"""

from abc import ABC, abstractmethod
import pandas as pd

class BasePreProcessor(ABC):
    """
    A BasePreProcessor for EEG and SEEG data

    :param data_root: string, the root of the dataset
    :param data_name: string, the name of dataset
    :param data_info: pd.DataFrame(header = ["raw_path", "start_time", "end_time", "gap_time", "name", "flag"]),
                        the information of every signal
    :param data_type: string, 'bi_class' or 'multi_class'
    """

    data_root = None # the root path of the data
    data_name = None  # the name of the dataset
    data_info = pd.DataFrame(columns=['patient', 'raw_path', 'start_time', 'end_time', 'gap_time', 'name', 'flag', 'type'])  # a df contains raw_path, endtime and name respectively

    def __init__(self, data_root, data_name, data_info):
        self.data_root = data_root
        self.data_name = data_name
        self.data_info = data_info

    def perform_preprocessing(self):
        print("start pre-processing")
        # TODO: uncomment this line when executing
        data_info_bi_class = self.data_info.loc[self.data_info['type'] == 'bi_class']
        data_info_multi_class = self.data_info.loc[self.data_info['type'] == 'multi_class']
        self.bi_class_handle(data_info_bi_class)
        self.multi_class_handle(data_info_multi_class)
        if self.data_info['type'].value_counts().length > 2:
            print("ERROR: Unknown Data Type Cannot Be Recognized.")
        print("finish pre-processing")

    @abstractmethod
    def bi_class_handle(self):
        print("call bi_class_handle() from base")
        pass

    @abstractmethod
    def multi_class_handle(self):
        print("call multi_class_handle() from base")
        pass





