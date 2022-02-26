# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 24.02.2022

from util.seeg_utils import *

from abc import ABC, abstractmethod
import pandas as pd
import uuid


class BaseTransformer(ABC):

    data_name = None # the name of the dataset
    data_type = None # bi class or multi class
    data_path = None # the path of all the raw data

    @abstractmethod
    def generate_data(self):
        pass

    @abstractmethod
    def save_data(self):
        pass

    @abstractmethod
    def save_segment(self):
        pass

    @abstractmethod
    def generate_save_path(self):
        pass

class SEEGBaseTransformer(BaseTransformer):

    flag = None # 0: pre-seizure, 1: attacking stage, 2: normal sleep
    path_common_channel = None # the names of common channels
    high_pass = 0
    low_pass = 30
    resample_freq = 100
    time = 2

    def __init__(self, data_name, data_type, data_path, flag, path_common_channel,
                 high_pass=0, low_pass=30, resample_freq=100, time=2):
        self.data_name = data_name
        self.data_type = data_type
        self.data_path = data_path
        self.flag = flag
        self.path_common_channel = path_common_channel
        self.high_pass = high_pass
        self.low_pass = low_pass
        self.resample_freq = resample_freq
        self.time = time

    def generate_data(self, flag_duration = 0, isfilter = True):
        data = pd.read_csv(self.path_commom_channel, sep=',')
        d_list = data['chan_name']
        common_channels = list(d_list)
        self.save_data(common_channels, flag_duration=flag_duration, isfilter=isfilter)

    def save_data(self, common_channels, flag_duration, isfilter):
        raw = read_raw(self.data_path)
        if isfilter:
            raw = filter_hz(raw, self.high_pass, self.low_pass)
        raw.resample(self.resample_freq, npad="auto")
        raw = select_channel_data_mne(raw, common_channels)  # 根据特定的信道顺序来选择信道
        raw.reorder_channels(common_channels)  # 更改信道的顺序

        raw_split_data = split_data(raw, self.time)  # 进行2秒的切片
        print("split time {}".format(len(raw_split_data)))
        self.save_segment(raw_split_data)

    def save_segment(self, data_split):
        path = self.generate_save_path()
        if not os.path.exists(path):
            os.makedirs(path)
        for d in data_split:
            name = str(uuid.uuid1()) + "-" + str(self.flag)
            path_all = os.path.join(path, name)
            save_numpy_info(d, path_all)
        print("File save successfully {}".format(path))

    def generate_save_path(self):
        # TODO
        # path_dir = save_split_data_test__path_dir  # 通过配置文件来读取
        path_dir = ''
        if self.flag == 0:  # pre-seizure
            if self.flag_duration == 0:
                dir = 'preseizure'
            else:
                if self.flag_duration == 1:
                    dir = 'preseizure/within_warning_time'
                else:
                    dir = 'preseizure/before_warning_time'
        else:
            if self.flag == 1:
                dir = "cases"  # attacking
            else:
                if self.flag == 2:
                    dir = "sleep"
                else:
                    dir = "awake"

        path_dir = os.path.join(path_dir, dir)
        if os.path.exists(path_dir) is not True:
            os.makedirs(path_dir)
            print("create dir:{}".format(path_dir))
        path_person = os.path.join(path_dir, self.name)
        if os.path.exists(path_person) is not True:
            os.makedirs(path_person)
            print("create dir:{}".format(path_person))
        return path_person