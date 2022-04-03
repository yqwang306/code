# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 03.03.2022

""" Perform splitting train/test for EEG and SEEG Data."""

from abc import ABC, abstractmethod
import os

class BaseSplitter(ABC):

    train_ratio = None
    val_ratio = None
    patient_name = None # a list containing the patients' names
    root = None # the path of generated data after preprocessing and transforming
    train_path = None
    test_path = None
    val_path = None
    resampling_base_size = None

    def __init__(self, root, patient_name, train_ratio = 0.7, val_ratio = 0.3, resampling_base_size = 3000):

        self.patient_name = patient_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.resampling_base_size = resampling_base_size

        self.root = "{}/{}/".format(root, patient_name)
        self.train_path = self.root + "train"
        self.test_path = self.root + "test"
        self.val_path = self.root + "val"

        dirs_to_be_created = [self.train_path, self.test_path, self.val_path]
        for dir in dirs_to_be_created:
            if os.path.exists(dir) is not True:
                os.makedirs(dir)
            else:
                os.system("rm -r {}/*".format(dir))

    @abstractmethod
    def save_splitted_data(self, sleep_label0, sleep_label1):
        pass

    @abstractmethod
    def perform_splitting(self):
        pass

    @abstractmethod
    def perform_splitting_n_1(self):
        pass
    