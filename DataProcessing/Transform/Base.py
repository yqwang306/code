# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 24.02.2022


from abc import ABC, abstractmethod

class BaseTransformer(ABC):

    data_root = None # the root of the dataset
    data_info = None # pd.DataFrame(patient_name, path_after_preprocess, flag, flag_duration)
                     # if not give, then transform all the preprocessed data
    map_path_common_channel = None

    @abstractmethod
    def generate_data(self):
        pass

    @abstractmethod
    def save_segment(self):
        pass

    @abstractmethod
    def generate_save_path(self):
        pass

    @abstractmethod
    def perform_transform(self):
        pass

