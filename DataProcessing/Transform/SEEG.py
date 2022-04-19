# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 25.02.2022

from DataProcessing.Transform.Base import BaseTransformer
from util.seeg_utils import *

import os
import pandas as pd
import uuid

class SEEGTransformer(BaseTransformer):

    high_pass = 0
    low_pass = 30
    resample_freq = 100
    time = 2

    def __init__(self, data_root, map_path_common_channel, data_info=None,
                 high_pass=0, low_pass=30, resample_freq=100, time=2):
        self.data_root = data_root
        self.data_info = data_info
        self.high_pass = high_pass
        self.low_pass = low_pass
        self.resample_freq = resample_freq
        self.time = time
        self.map_path_common_channel = map_path_common_channel

    def perform_transform(self, data_info=None):
        print("start transforming")

        if data_info != None: # user specify the files need transformation
            self.data_info - data_info

        if self.data_info == None: # if data_info not give, by default transform all the preprocessed data
            all_patients = os.listdir(os.path.join(self.data_root, 'preprocess'))
            for p in all_patients:
                all_file = get_all_file_path(os.path.join(self.data_root, 'preprocess/{}'.format(p)))
                for path in all_file:
                    flag, flag_duration = get_flag_from_path(path)
                    self.generate_data(path, self.map_path_common_channel[p], p, flag, flag_duration)
        else: # if data_info given, only transform specified data
            for idx, row in self.data_info.iterrows():
                self.generate_data(row['path_after_preprocess'], row['path_commom_channel'], row['patient_name'],
                                   row['flag'], row['flag_duration'])

        print("finish transforming")

    def generate_data(self, path, path_commom_channel, name, flag, flag_duration, isfilter = True):
        data = pd.read_csv(self.map_path_common_channel[name], sep=',')
        d_list = data['chan_name']
        common_channels = list(d_list)

        raw = read_raw(path)
        if isfilter:
            raw = filter_hz(raw, self.high_pass, self.low_pass)
        raw.resample(self.resample_freq, npad="auto")
        raw = select_channel_data_mne(raw, common_channels)  # 根据特定的信道顺序来选择信道
        raw.reorder_channels(common_channels)  # 更改信道的顺序

        raw_split_data = split_data(raw, self.time)  # 进行2秒的切片
        print("split time {}".format(len(raw_split_data)))

        save_path = self.generate_save_path(flag, flag_duration, name)
        self.save_segment(raw_split_data, save_path, flag)

    def save_segment(self, data_split, save_path, flag):
        for d in data_split:
            name = str(uuid.uuid1()) + "-" + str(flag)
            path_all = os.path.join(save_path, name)
            save_numpy_info(d, path_all)
        print("File save successfully {}".format(save_path))

    def generate_save_path(self, flag, flag_duration, name):
        # TODO
        path_dir = os.path.join(self.data_root, 'transform')
        path_dir = os.path.join(path_dir, generate_path_by_flag(flag, flag_duration))
        if os.path.exists(path_dir) is not True:
            os.makedirs(path_dir)
            print("create dir:{}".format(path_dir))
        path_person = os.path.join(path_dir, name)
        if os.path.exists(path_person) is not True:
            os.makedirs(path_person)
            print("create dir:{}".format(path_person))
        return path_person

    @classmethod
    def get_normal_sleep_info(cls):
        map_path_common_channel = {"LK": "../data/data_slice/channels_info/LK_seq.csv",
                                   "BDP": "../data/data_slice/channels_info/BDP_seq.csv",
                                   "SYF": "../data/data_slice/channels_info/SYF_seq.csv",
                                   "WSH": "../data/data_slice/channels_info/WSH_seq.csv",
                                   "ZK": "../data/data_slice/channels_info/ZK_seq.csv"}

        df_LK_info = pd.DataFrame({
            'patient_name': ['LK', 'LK', 'LK', 'LK', 'LK'],
            'data_path': ["../data/raw_data/LK/LK_SLEEP/LK_Sleep_raw-0.fif",
                             '../data/raw_data/LK/LK_SLEEP/LK_Sleep_raw-1.fif',
                             '../data/raw_data/LK/LK_SLEEP/LK_Sleep_raw-2-0.fif',
                             '../data/raw_data/LK/LK_SLEEP/LK_Sleep_raw-4-0.fif',
                             '../data/raw_data/LK/LK_SLEEP/LK_Sleep_raw-6-0.fif'
                          ],
            'flag': [2, 2, 2, 2, 2, 2],
            'flag_duration': [-1, -1, -1, -1, -1, -1]
        })

        df_BDP_info = {
            'patient_name': ['BDP'],
            'data_path': ['../data/raw_data/BDP/BDP_SLEEP/BDP_Sleep_raw.fif'],
            'flag': [2]
        }

        df_SYF_info = {
            'patient_name': ['SYF'],
            'data_path': ['../data/raw_data/SYF/SYF_SLEEP/SYF_Sleep_raw.fif'],
            'flag': [2]
        }

        df_WSH_info = {
            'patient_name': ['WSH'],
            'data_path': ['../data/raw_data/WSH/WSH_SLEEP/WSH_Sleep_raw.fif'],
            'flag': [2]
        }

        df_ZK_info = {
            'patient_name': ['ZK', 'ZK', 'ZK', 'ZK', 'ZK', 'ZK', 'ZK', 'ZK', 'ZK'],
            'data_path': ['../data/raw_data/ZK/ZK_SLEEP/ZK_Sleep_raw.fif',
                             '../data/raw_data/ZK/ZK_SLEEP/ZK_Sleep_raw-11.fif',
                             '../data/raw_data/ZK/ZK_SLEEP/ZK_Sleep_raw-22.fif',
                             '../data/raw_data/ZK/ZK_SLEEP/ZK_Sleep_raw-33.fif',
                             '../data/raw_data/ZK/ZK_SLEEP/ZK_Sleep_raw-44.fif',
                             '../data/raw_data/ZK/ZK_SLEEP/ZK_Sleep_raw-55.fif',
                             '../data/raw_data/ZK/ZK_SLEEP/ZK_Sleep_raw-66.fif',
                             '../data/raw_data/ZK/ZK_SLEEP/ZK_Sleep_raw-77.fif',
                             '../data/raw_data/ZK/ZK_SLEEP/ZK_Sleep_raw-88.fif'
                             ],
            'flag': [2, 2, 2, 2, 2, 2, 2, 2, 2],
            'flag_duration': [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        }

        return map_path_common_channel, pd.concat([df_ZK_info, df_LK_info, df_SYF_info, df_BDP_info, df_WSH_info])

    @classmethod
    def get_pre_seizure_biclass_info(cls):
        map_path_common_channel = {"LK": "../data/data_slice/channels_info/LK_seq.csv",
                                   "BDP": "../data/data_slice/channels_info/BDP_seq.csv",
                                   "SYF": "../data/data_slice/channels_info/SYF_seq.csv",
                                   "WSH": "../data/data_slice/channels_info/WSH_seq.csv",
                                   "ZK": "../data/data_slice/channels_info/ZK_seq.csv",
                                   "JWJ": "../data/data_slice/channels_info/JWJ_seq.csv"}

        df_LK_info = {
            'patient_name': ['LK'],
            'data_path': get_all_path("../data/raw_data/LK/LK_Pre_seizure"),
            'flag': [0]
        }

        df_ZK_info = {
            'patient_name': ['ZK'],
            'data_path': get_all_path("../data/raw_data/ZK/ZK_Pre_seizure"),
            'flag': [0]
        }

        df_WSH_info = {
            'patient_name': ['WSH'],
            'data_path': get_all_path("../data/raw_data/WSH/WSH_Pre_seizure"),
            'flag': [0]
        }

        df_SYF_info = {
            'patient_name': ['SYF'],
            'data_path': get_all_path("../data/raw_data/SYF/SYF_Pre_seizure"),
            'flag': [0]
        }

        df_BDP_info = {
            'patient_name': ['BDP'],
            'data_path': get_all_path("../data/raw_data/BDP/BDP_Pre_seizure"),
            'flag': [0]
        }

        df_JWJ_info = {
            'patient_name': ['JWJ'],
            'data_path': get_all_path("../data/raw_data/JWJ/JWJ_Pre_seizure"),
            'flag': [0]
        }

        return map_path_common_channel, pd.concat([df_ZK_info, df_LK_info, df_SYF_info, df_BDP_info, df_WSH_info, df_JWJ_info])

    @classmethod
    def get_awake_info(cls, path_common_channel):
        # TODO: why path_common_channel not given?
        return {
            'path_commom_channel': path_common_channel,
            'data_path': get_all_path("../data/raw_data/LK/LK_Awake"),
            'flag': 3,
            'data_name': "LK"
        }

    @classmethod
    def get_pre_seizure_multiclass_info(cls, path_common_channel):
        # TODO: why path_common_channel not given?
        df_LK_before_warning_info = {
            'path_commom_channel': path_common_channel,
            'data_path': get_all_path("../data/raw_data/LK/multiPre_seizure/before_warning_time"),
            'flag': 0,
            'data_name': "LK",
            'flag_duration': 2
        }

        df_LK_within_warning_info = {
            'path_commom_channel': path_common_channel,
            'data_path': get_all_path("../data/raw_data/LK/multiPre_seizure/within_warning_time"),
            'flag': 0,
            'data_name': "LK",
            'flag_duration': 2
        }

        return {
            'LK_before_warning': df_LK_before_warning_info,
            'LK_within_warning': df_LK_within_warning_info
        }
