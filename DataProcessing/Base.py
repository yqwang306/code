# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 10.03.2022

from DataProcessing.PreProcess.SEEG import SEEGPreProcessor
from DataProcessing.Transform.SEEG import SEEGTransformer
from DataProcessing.Split.SEEG import SEEGSpliter


class SEEGDataProcessor:
    preProcessor = None
    transformer = None
    splitter = None
    SEEGData = None

    def __init__(self, data_name, data_root, data_info_raw, map_path_common_channel,
                 high_pass=0, low_pass=30, resample_freq=100, time=2, train_ratio=0.7, val_ratio=0.3,
                 resampling_base_size=3000, split_mode=0, patient_test=None):
        self.preProcessor = SEEGPreProcessor(data_root, data_name, data_info_raw)
        self.transformer = SEEGTransformer(data_root, map_path_common_channel, high_pass=high_pass,
                                           low_pass=low_pass, resample_freq=resample_freq, time=time)
        self.splitter = SEEGSpliter(data_root, patient_test=patient_test, mode=split_mode, train_ratio=train_ratio, val_ratio=val_ratio, resampling_base_size=resampling_base_size)

    def perform_data_processing(self):
        print("start processing raw data...")
        self.preProcessor.perform_preprocessing()
        self.transformer.perform_transform()
        self.SEEGData = self.splitter.perform_splitting()
        print("data processing is done!")

