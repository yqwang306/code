# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 06.03.2022

from abc import ABC, abstractmethod

class BaseDataset(ABC):

    @abstractmethod
    def load_data_by_class(self):
        pass

    @abstractmethod
    def get_statistic_info(self):
        pass
