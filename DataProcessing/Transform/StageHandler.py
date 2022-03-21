# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 25.02.2022

from DataProcessing.Transform.Base import SEEGBaseTransformer
from util.seeg_utils import get_all_path


class SEEGStageHandler(SEEGBaseTransformer):

    def perform_stage_handle(self):
        for path_raw in self.data_path:
            self.generate_data(path_raw, self.flag, self.data_name, self.path_common_channel)
        print("The current stage's signal of {} has already been done!".format(self.data_name))

    @classmethod
    def get_normal_sleep_info(cls):
        df_LK_info = {
            'path_commom_channel': "../data/data_slice/channels_info/LK_seq.csv",
            'data_path': ["../data/raw_data/LK/LK_SLEEP/LK_Sleep_raw-0.fif",
                          '../data/raw_data/LK/LK_SLEEP/LK_Sleep_raw-1.fif',
                          '../data/raw_data/LK/LK_SLEEP/LK_Sleep_raw-2-0.fif',
                          '../data/raw_data/LK/LK_SLEEP/LK_Sleep_raw-4-0.fif',
                          '../data/raw_data/LK/LK_SLEEP/LK_Sleep_raw-6-0.fif'
                          ],
            'data_name': "LK",
            'flag': 2
        }

        df_BDP_info = {
            'path_commom_channel': "../data/data_slice/channels_info/BDP_seq.csv",
            'data_path': ['../data/raw_data/BDP/BDP_SLEEP/BDP_Sleep_raw.fif'],
            'data_name': "BDP",
            'flag': 2
        }

        df_SYF_info = {
            'path_commom_channel': "../data/data_slice/channels_info/SYF_seq.csv",
            'data_path': ['../data/raw_data/SYF/SYF_SLEEP/SYF_Sleep_raw.fif'],
            'data_name': "SYF",
            'flag': 2
        }

        df_WSH_info = {
            'path_commom_channel': "../data/data_slice/channels_info/WSH_seq.csv",
            'data_path': ['../data/raw_data/WSH/WSH_SLEEP/WSH_Sleep_raw.fif'],
            'data_name': "WSH",
            'flag': 2
        }

        df_ZK_info = {
            'path_commom_channel': "../data/data_slice/channels_info/ZK_seq.csv",
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
            'data_name': "ZK",
            'flag': 2
        }

        return {
            'LK': df_LK_info,
            'BDP': df_BDP_info,
            'SYF': df_SYF_info,
            'WSH': df_WSH_info,
            'ZK': df_ZK_info
        }

    @classmethod
    def get_pre_seizure_biclass_info(cls):
        df_LK_info = {
            'path_commom_channel': "../data/data_slice/channels_info/LK_seq.csv",
            'data_path': get_all_path("../data/raw_data/LK/LK_Pre_seizure"),
            'flag': 0,
            'data_name': "LK"
        }

        df_ZK_info = {
            'path_commom_channel': "../data/data_slice/channels_info/ZK_seq.csv",
            'data_path': get_all_path("../data/raw_data/ZK/ZK_Pre_seizure"),
            'flag': 0,
            'data_name': "ZK"
        }

        df_WSH_info = {
            'path_commom_channel': "../data/data_slice/channels_info/WSH_seq.csv",
            'data_path': get_all_path("../data/raw_data/WSH/WSH_Pre_seizure"),
            'flag': 0,
            'data_name': "WSH"
        }

        df_SYF_info = {
            'path_commom_channel': "../data/data_slice/channels_info_back/SYF_seq.csv",
            'data_path': get_all_path("../data/raw_data/SYF/SYF_Pre_seizure"),
            'flag': 0,
            'data_name': "SYF"
        }

        df_BDP_info = {
            'path_commom_channel': "../data/data_slice/channels_info/BDP_seq.csv",
            'data_path': get_all_path("../data/raw_data/BDP/BDP_Pre_seizure"),
            'flag': 0,
            'data_name': "BDP"
        }

        df_BDP_info = {
            'path_commom_channel': "../data/data_slice/channels_info/JWJ_seq.csv",
            'data_path': get_all_path("../data/raw_data/JWJ/JWJ_Pre_seizure"),
            'flag': 0,
            'data_name': "JWJ"
        }

        return {
            'LK': df_LK_info,
            'BDP': df_BDP_info,
            'SYF': df_SYF_info,
            'WSH': df_WSH_info,
            'ZK': df_ZK_info,
            'BDP': df_BDP_info
        }

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
