# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 06.03.2022

from abc import ABC, abstractmethod

class BaseDataset(ABC):

    @abstractmethod
    def get_dataset_tree(self):
        pass

    @abstractmethod
    def get_statistic(self):
        pass
