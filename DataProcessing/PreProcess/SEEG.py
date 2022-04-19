# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 09.01.2022

""" Preprocessing Pipeline For SEEG Dataset """

from DataProcessing.PreProcess import BasePreProcessor
from util.seeg_utils import *
import os
import pandas as pd

class SEEGPreProcessor(BasePreProcessor):

    @classmethod
    def get_bi_class_data_info(self):
        """
        return the metadata of our bi class dataset

        :return: df_info(pd.DataFrame())
        """

        data_info_LK = pd.DataFrame({
            'patient': ["LK", "LK", "LK", "LK", "LK", "LK"],
            'raw_path': [
                "../data/raw_data/LK_SZ/LK_SZ1_seeg_raw.fif",
                "../data/raw_data/LK_SZ/LK_SZ2_seeg_raw.fif",
                "../data/raw_data/LK_SZ/LK_SZ3_seeg_raw.fif",
                "../data/raw_data/LK_SZ/LK_SZ4_seeg_raw.fif",
                "../data/raw_data/LK_SZ/LK_SZ5_seeg_raw.fif",
                "../data/raw_data/LK_SZ/LK_SZ6_seeg_raw.fif"
            ],
            'start_time': [
                0, 0, 0, 0, 0, 0
            ],
            'end_time': [
                546, 564, 733, 995, 1535, 702
            ],
            'gap_time': [
                30, 30, 30, 30, 30, 30
            ],
            'name': [
                "LK_SZ1_pre_seizure",
                "LK_SZ2_pre_seizure",
                "LK_SZ3_pre_seizure",
                "LK_SZ4_pre_seizure",
                "LK_SZ5_pre_seizure",
                "LK_SZ6_pre_seizure"
            ],
            'flag': [
                0, 0, 0, 0, 0, 0
            ],
            'type': [
                'bi_class', 'bi_class', 'bi_class', 'bi_class', 'bi_class', 'bi_class'
            ]
        })
        return data_info_LK

    @classmethod
    def get_extended_bi_class_info(self):
        """
        return the metadata of our bi class dataset

        :return: df_info(pd.DataFrame())
        """

        data_info_WSH = pd.DataFrame({
            'patient': ["WSJ"],
            'raw_path': ['../data/raw_data/WSH/WSH_SZ/WSH_SZ1_raw.fif'],
            'start_time': [0],
            'end_time': [1446],
            'gap_time': [30],
            'name': ["WSH_SZ1_pre_seizure"],
            'flag': [0],
            'type': ['bi_class']
        })

        data_info_SYF = pd.DataFrame({
            'patient': ['SYF', 'SYF', 'SYF'],
            'raw_path': [
                '../data/raw_data/SYF/SYF_SZ/SYF_SZ1_raw.fif',
                '../data/raw_data/SYF/SYF_SZ/SYF_SZ2_raw.fif',
                '../data/raw_data/SYF/SYF_SZ/SYF_SZ3_raw.fif'
            ],
            'start_time': [
                0, 0, 0
            ],
            'end_time': [
                2581, 2703, 3272
            ],
            'gap_time': [
                30, 30, 30
            ],
            'name': [
                "SYF_SZ1_pre_seizure",
                "SYF_SZ2_pre_seizure",
                "SYF_SZ3_pre_seizure"
            ],
            'flag': [0, 0, 0],
            'type': ['bi_class', 'bi_class', 'bi_class']
        })

        data_info_SJ = pd.DataFrame({
            'patient': ['SJ', 'SJ', 'SJ'],
            'raw_path': [
                '../data/raw_data/SJ/SJ_SZ/SJ_SZ1_raw.fif',
                '../data/raw_data/SJ/SJ_SZ/SJ_SZ2_raw.fif',
                '../data/raw_data/SJ/SJ_SZ/SJ_SZ3_raw.fif'
            ],
            'start_time': [
                0, 0, 0
            ],
            'end_time': [
                1040, 790, 899
            ],
            'gap_time': [
                30, 30, 30
            ],
            'name': [
                "SJ_SZ1_pre_seizure",
                "SJ_SZ2_pre_seizure",
                "SJ_SZ3_pre_seizure"
            ],
            'flag': [0, 0, 0],
            'type': ['bi_class', 'bi_class', 'bi_class']
        })

        data_info_BDP = pd.DataFrame({
            'patient': ['BDP', 'BDP'],
            'raw_path': [
                "../data/raw_data/BDP/BDP_Pre_seizure",
                '../data/raw_data/BDP/BDP_SZ/BDP_SZ2_raw.fif'
            ],
            'start_time': [
                0, 696
            ],
            'end_time': [
                651, 4296
            ],
            'gap_time': [
                30, 30
            ],
            'name': [
                "BDP_SZ1_pre_seizure",
                "BDP_SZ2_pre_seizure"
            ],
            'flag': [0, 0],
            'type': ['bi_class', 'bi_class']
        })

        data_info_JWJ = pd.DataFrame({
            'patient': ['JWJ', 'JWJ'],
            'raw_path': [
                "../data/raw_data/JWJ/JWJ_SZ/JWJ_SZ1_raw.fif",
                "../data/raw_data/JWJ/JWJ_SZ/JWJ_SZ2_raw.fif"
            ],
            'start_time': [
                0, 0
            ],
            'end_time': [
                349, 349
            ],
            'gap_time': [
                30, 30
            ],
            'name': [
                "JWJ_SZ1_pre_seizure",
                "JWJ_SZ2_pre_seizure"
            ],
            'flag': [0, 0],
            'type': ['bi_class', 'bi_class']
        })

        return pd.concat([self.get_bi_class_data_info(), data_info_WSH, data_info_SYF, data_info_SJ, data_info_BDP, data_info_JWJ], ignore_index=True)

    @classmethod
    def get_multi_class_data_info(self):
        """
        return the metadata of our multi class dataset

        :return: df_info(pd.DataFrame())
        """

        data_info_LK_SZ = pd.DataFrame({
            'patient': ['LK_SZ', 'LK_SZ', 'LK_SZ', 'LK_SZ', 'LK_SZ', 'LK_SZ'],
            'raw_path': [
                "../data/raw_data/LK_SZ/LK_SZ1_seeg_raw.fif",
                "../data/raw_data/LK_SZ/LK_SZ2_seeg_raw.fif",
                "../data/raw_data/LK_SZ/LK_SZ3_seeg_raw.fif",
                "../data/raw_data/LK_SZ/LK_SZ4_seeg_raw.fif",
                "../data/raw_data/LK_SZ/LK_SZ5_seeg_raw.fif",
                "../data/raw_data/LK_SZ/LK_SZ6_seeg_raw.fif"
            ],
            'start_time': [
                0, 0, 0, 0, 0, 0
            ],
            'end_time': [
                546, 564, 733, 995, 1535, 702
            ],
            'gap_time': [
                30, 30, 30, 30, 30, 30
            ],
            'name': [
                "LK_SZ1_pre_seizure",
                "LK_SZ2_pre_seizure",
                "LK_SZ3_pre_seizure",
                "LK_SZ4_pre_seizure",
                "LK_SZ5_pre_seizure",
                "LK_SZ6_pre_seizure"
            ],
            'flag': [0, 0, 0, 0, 0, 0],
            'type': ['multi_class', 'multi_class', 'multi_class', 'multi_class', 'multi_class', 'multi_class']
        })
        return data_info_LK_SZ

    def get_duration_data(self, raw_path, name, save_dir, start, end_times, gap_time=30):
        '''
        :param raw_path: the path of raw data
        :param name: name to be stored
        :param save_dir: the directory for saving processed data
        :param start: start time of SEEG signal
        :param end_times: end time of SEEG signal
        :param gap_time: lasted time of SEEG signal
        :return: a signal segment, ranging from the start time to the end time and lasting for gap time long
        '''

        if end_times - gap_time > start:
            raw_data = read_raw(raw_path)
            channel_names = get_channels_names(raw_data)
            duration_data = get_duration_raw_data(raw_data, start, end_times - gap_time)
            if os.path.exists(save_dir) is not True:
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, name)
            save_path += '_raw.fif'

            rewrite(duration_data, channel_names, save_path)
            return duration_data
        else:
            print("ERROR: Time span is invalid.")
            return None


    def bi_class_handle(self, data_info):
        """
        Perform pre-processing for bi class data
        :return:
        """
        for idx, row in data_info.iterrows():
            save_dir = self.data_root + "/preprocess/{}".format(row['patient'])
            subdir = generate_path_by_flag(row['flag'], -1)
            path = os.path.join(save_dir, subdir)
            self.get_duration_data(row['raw_path'], row['name'], path, row['start_time'], row['end_time'], row['gap_time'])

    def multi_class_handle(self, data_info):
        """
        Perform pre-processing for multi class data
        :return:
        """
        warning_time = 300  # 设计的预警时间为300
        within_warning_time_dir = "within_warning_time"
        before_warning_time_dir = "before_warning_time"

        for idx, row in self.data_info.iterrows():
            save_dir = "../data/preprocess/{}/preseizure".format(row['patient'])
            save_dir_wwt = os.path.join(save_dir, within_warning_time_dir)
            save_dir_bwt = os.path.join(save_dir, before_warning_time_dir)
            if os.path.exists(save_dir_bwt) is not True:
                os_mkdir(save_dir_bwt)
            if os.path.exists(save_dir_wwt) is not True:
                os_mkdir(save_dir_wwt)

            before_warning_end_time = row['end_time'] - warning_time
            self.get_duration_data(row['raw_path'], row['name'] + '_' + within_warning_time_dir, save_dir_bwt, row['start_time'], before_warning_end_time, row['gap_time'])
            self.get_duration_data(row['raw_path'], row['within_warning_name'] + '_' + before_warning_time_dir, save_dir_wwt, before_warning_end_time, row['end_time'], row['gap_time'])






