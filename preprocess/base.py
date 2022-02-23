# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 09.01.2022

""" A Basic Preprocessing Pipeline"""

from abc import ABC, abstractmethod

class BasePreProcessor(ABC):
    """
    A BasePreProcessor for EEG and SEEG data

    :param data_name: string, the name of dataset
    :param data_info: pd.DataFrame(header = ["raw_path", "start_time", "end_time", "gap_time", "name"]),
                        or pd.DataFrame(header = ["raw_path", "start_time", "end_time", "gap_time", "within_warning_name", "before_warning_name"]),
                        the information of every singal
    :param data_type: string, 'bi_class' or 'multi_class'
    """

    data_name = None # the name of the dataset
    data_info = None  # a df contains raw_path, endtime and name respectively
    data_type = None # bi class or not

    def __init__(self, data_name, data_info, data_type):
        self.data_name = data_name
        self.data_info = data_info
        self.data_type = data_type

    def load_data(self):
        print("call load_data() from base")
        # TODO
        if self.data_type == 'bi_class':
            self.bi_class_handle()
        elif self.data_type == 'multi_class':
            self.multi_class_handle()
        else:
            print("ERROR: data_type not given.")
            pass

    @abstractmethod
    def bi_class_handle(self):
        print("call bi_class_handle() from base")
        pass

    @abstractmethod
    def multi_class_handle(self):
        print("call multi_class_handle() from base")
        pass

    @staticmethod
    def filter_noise():
        print("call filter_noise() from base")
        # TODO
        pass

    @staticmethod
    def normalise():
        # TODO
        pass





