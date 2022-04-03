# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 5.03.2022

from DataProcessing.Split.Base import BaseSplitter

import numpy as np
import os
import random

class SEEGSpliter(BaseSplitter):

    def save_splitted_data(self, sleep_label0, sleep_label1):

        train_folder_dir_normal = os.path.join(self.train_path, "sleep_normal")
        train_folder_dir_pre = os.path.join(self.train_path, "pre_zeizure")
        test_folder_dir_normal = os.path.join(self.test_path, "sleep_normal")
        test_folder_dir_pre = os.path.join(self.test_path, "pre_zeizure")
        val_folder_dir_normal = os.path.join(self.val_path, "sleep_normal")
        val_folder_dir_pre = os.path.join(self.val_pathl, "pre_zeizure")
        dirs_to_be_created = [train_folder_dir_normal, train_folder_dir_pre, test_folder_dir_normal,
                              test_folder_dir_pre, val_folder_dir_normal, val_folder_dir_pre]
        for dir in dirs_to_be_created:
            if os.path.exists(dir) is not True:
                os.makedirs(dir)

        train_num = int(self.train_ratio * len(sleep_label0))
        test_num = int(self.val_ratio * len(sleep_label0))

        for (i, p) in enumerate(sleep_label0):
            name = p.split('/')[-1]
            d = np.load(p)
            if i <= int(train_num):
                save_path = os.path.join(train_folder_dir_normal, name)
            else:
                if i < (train_num + test_num):
                    save_path = os.path.join(test_folder_dir_normal, name)
                else:
                    save_path = os.path.join(val_folder_dir_normal, name)
            np.save(save_path, d)
        print("Successfully write for normal sleep data!!!")

        for (i, p) in enumerate(sleep_label1):
            name = p.split('/')[-1]
            d = np.load(p)
            if i <= int(train_num):
                save_path = os.path.join(train_folder_dir_pre, name)
            else:
                if i <= (train_num + test_num):
                    save_path = os.path.join(test_folder_dir_pre, name)
                else:
                    save_path = os.path.join(val_folder_dir_pre, name)
            np.save(save_path, d)
        print("Successfully write for pre seizure sleep data!!!")

    def perform_splitting(self):

        # TODO: seeg data model need to be built
        seeg = seegdata()

        sleep_normal = [] # paths of normal sleep data collected
        all_path = seeg.get_all_path_by_keyword('sleep')
        for dir_path in all_path.values():
            for path in dir_path:
                sleep_normal.append(path)

        sleep_pre_seizure = [] # paths of pre-seizure sleep data collected
        all_path = seeg.get_all_path_by_keyword("preseizure")
        for dir_path in all_path.values():
            for path in dir_path:
                sleep_pre_seizure.append(path)

        print("normal sleep:{} pre seizure:{}".format(len(sleep_normal), len(sleep_pre_seizure)))
        random.shuffle(sleep_normal)
        random.shuffle(sleep_pre_seizure)
        min_data = min(len(sleep_normal), len(sleep_pre_seizure))  # get the minimum size of these two datasets
        sleep_label1 = sleep_pre_seizure[:min_data]
        sleep_label0 = sleep_normal[:min_data]
        train_num = int(self.train_ratio * len(sleep_label0))
        test_num = int(self.val_ratio * len(sleep_label0))
        print("train number:{}, test number:{}, val number:{}".format(train_num, test_num,
                                                                      len(sleep_label0) - test_num - train_num))
        self.save_splitted_data(sleep_label0, sleep_label1)

    def perform_splitting_n_1(self):

        # TODO: seeg data model need to be built
        seeg = seegdata()
        all_path_normal_sleep = seeg.get_all_path_by_keyword('sleep')
        all_path_preseizure = seeg.get_all_path_by_keyword('preseizure')

        m_normal = len(all_path_normal_sleep) - 1
        n_preseizure = len(all_path_preseizure) - 1
        resampling_size_normal = int((n_preseizure / m_normal) * self.resampling_base_size)
        resampling_size_preseizure = self.resampling_base_size
        print("size of sampling normal sleep:{}  size of sampling preseizure:{}".format(resampling_size_normal,
                                                                                        resampling_size_preseizure))

        sleep_normal = []  # paths of normal sleep data collected
        val_sleep_normal = []
        sleep_pre_seizure = []  # paths of pre-seizure sleep data collected
        val_sleep_pre_seizure = []

        for key, dp in all_path_normal_sleep.items():
            if key not in self.patient_name:
                result = up_sampling(dp, resampling_size_normal)
                for p in result.items():
                    sleep_normal.append(p)
            else:
                result = up_sampling(dp, self.resampling_base_size)
                for p in result.items():
                    val_sleep_normal.append(p)

        for key, dp in all_path_preseizure.items():  # 获取的是所有的癫痫发前数据
            # 使用重采样的方法

            if key not in self.patient_name:
                result = up_sampling(dp, resampling_size_preseizure)
                for p in result.items():
                    sleep_pre_seizure.append(p)  # 加入的是字典
            else:
                result = up_sampling(dp, self.resampling_base_size)
                for p in result.items():
                    val_sleep_pre_seizure.append(p)

        random.shuffle(sleep_normal)
        random.shuffle(sleep_pre_seizure)  # 用于训练的数据集
        random.shuffle(val_sleep_normal)  # 用于验证的数据集
        random.shuffle(val_sleep_pre_seizure)