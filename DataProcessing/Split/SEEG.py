# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 5.03.2022

from DataProcessing.Split.Base import BaseSplitter
from Dataset.SEEG import SEEGData
from util.seeg_utils import get_all_file_path, relocate_file_in_dic

import numpy as np
import os
import random
from collections import Counter

class SEEGSpliter(BaseSplitter):

    def perform_splitting(self):
        root = os.path.join(self.root, 'transform')
        train_path = os.path.join(root, 'train')
        test_path = os.path.join(root, 'test')
        val_path = os.path.join(root, 'val')
        dirs_to_be_created = [train_path, test_path, val_path]
        for dir in dirs_to_be_created:
            if os.path.exists(dir) is not True:
                os.makedirs(dir)
            else:
                os.system("rm -r {}/*".format(dir))

        if self.mode == 0:
            self.split_data_n()
        elif self.mode == 1:
            self.perform_splitting_n_1()

    def get_transformed_data(self):

        sleep_normal = []  # paths of normal sleep data collected
        all_path = get_all_file_path(os.path.join(self.root, 'tranform/sleep'))
        for dir_path in all_path.values():
            for path in dir_path:
                sleep_normal.append(path)

        sleep_pre_seizure = []  # paths of pre-seizure sleep data collected
        all_path = get_all_file_path(os.path.join(self.root, 'tranform/preseizure'))
        for dir_path in all_path.values():
            for path in dir_path:
                sleep_pre_seizure.append(path)

        print("normal sleep:{} pre seizure:{}".format(len(sleep_normal), len(sleep_pre_seizure)))
        return sleep_normal, sleep_pre_seizure

    def up_sampling(self, paths, upsampling_size):
        raw_data_size = len(paths)
        result = {}
        if raw_data_size > upsampling_size:
            print("Raw data size {} is bigger than up sampling size {}, down sampling ".format(raw_data_size,
                                                                                            upsampling_size))
            data_index = random.sample(range(raw_data_size), upsampling_size)
            for d in data_index:
                name = paths[d].split('/')[-1] #TODO:might has problems
                result[name] = paths[d]
        else:
            print("Up sampling, the repetition rate is :{:.2f}%".format(
                (upsampling_size - raw_data_size) * 100 / raw_data_size))
            data_index = np.random.randint(0, raw_data_size, upsampling_size - raw_data_size)
            data_index = list(range(raw_data_size)) + data_index.tolist()
            bit_map = np.zeros(raw_data_size)  # 位图， 查看重采样的重复个数
            for d in data_index:
                path = paths[d]
                bit_map[d] += 1  # 修改位图
                pre_name = path[:-4]  # 获得名称的前缀
                if bit_map[d] > 1:
                    pre_name = pre_name + "-ups-{}".format(int(bit_map[d]))
                    full_path = pre_name + ".npy"
                else:
                    full_path = path
                name = full_path.split('/')[-1] #TODO:might has problems
                result[name] = path
            sa = Counter(bit_map)
            print(sa)
        return result

    def save_splitted_data(self, sleep_label0, sleep_label1):

        train_folder_dir_normal = os.path.join(self.root, "split/train/sleep")
        train_folder_dir_pre = os.path.join(self.root, "split/train/prezeizure")
        test_folder_dir_normal = os.path.join(self.root, "split/test/sleep")
        test_folder_dir_pre = os.path.join(self.root, "split/test/prezeizure")
        val_folder_dir_normal = os.path.join(self.val_path, "split/val/sleep")
        val_folder_dir_pre = os.path.join(self.val_pathl, "split/val/prezeizure")
        dirs_to_be_created = [train_folder_dir_normal, train_folder_dir_pre, test_folder_dir_normal,
                              test_folder_dir_pre, val_folder_dir_normal, val_folder_dir_pre]
        for dir in dirs_to_be_created:
            if os.path.exists(dir) is not True:
                os.makedirs(dir)

        train_num = int(self.train_ratio * len(sleep_label0))
        test_num = int(self.val_ratio * len(sleep_label0))
        print("train number:{}, test number:{}, val number:{}".format(train_num, test_num,
                                                                      len(sleep_label0) - test_num - train_num))

        for (i, p) in enumerate(sleep_label0):
            name = p.split('/')[-1] # might has problem
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

    def save_resampling_data(self, sleep_normal, sleep_pre_seizure, val_normal, val_pre_seizure):

        train_folder_dir_normal = os.path.join(self.root, "split/train/sleep")
        train_folder_dir_pre = os.path.join(self.root, "split/train/prezeizure")
        test_folder_dir_normal = os.path.join(self.root, "split/test/sleep")
        test_folder_dir_pre = os.path.join(self.root, "split/test/prezeizure")
        val_folder_dir_normal = os.path.join(self.val_path, "split/val/sleep")
        val_folder_dir_pre = os.path.join(self.val_pathl, "split/val/prezeizure")
        dirs_to_be_created = [train_folder_dir_normal, train_folder_dir_pre, test_folder_dir_normal,
                              test_folder_dir_pre, val_folder_dir_normal, val_folder_dir_pre]
        for dir in dirs_to_be_created:
            if os.path.exists(dir) is not True:
                os.makedirs(dir)

        train_normal_number = int(self.train_ratio * len(sleep_normal))
        train_preseizure_number = int(self.train_ratio * len(sleep_pre_seizure))

        train_normal_data_dict = sleep_normal[:train_normal_number]
        test_normal_data_dict = sleep_normal[train_normal_number:]
        train_preseizure_data_dict = sleep_pre_seizure[:train_preseizure_number]
        test_preseizure_data_dict = sleep_pre_seizure[train_preseizure_number:]

        relocate_file_in_dic(train_normal_data_dict, train_folder_dir_normal)
        relocate_file_in_dic(test_normal_data_dict, test_folder_dir_normal)

        relocate_file_in_dic(train_preseizure_data_dict, train_folder_dir_pre)
        relocate_file_in_dic(test_preseizure_data_dict, test_folder_dir_pre)

        # 验证集的采样重写
        relocate_file_in_dic(val_normal, val_folder_dir_normal)
        relocate_file_in_dic(val_pre_seizure, val_folder_dir_pre)

        print("-" * 5 + "statistic information" + "-" * 5)
        print("training data number of normal sleep :{} testing data number of normal sleep :{}\n"
              "training data number of preseizure: {}  testing data number of preseizure: {}\n"
              "validation data number of preseizure: {}  validation data number of preseizure: {}\n"
              .format(len(train_normal_data_dict), len(test_normal_data_dict), len(train_preseizure_data_dict),
                      len(test_preseizure_data_dict), len(val_normal), len(val_pre_seizure)))

        processedData = SEEGData(self.root)
        return processedData

    def split_data_n(self):
        # TODO: seeg data model need to be built

        sleep_normal, sleep_pre_seizure = self.get_transformed_data()
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

        all_path_normal_sleep, all_path_preseizure = self.get_transformed_data()

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
        if self.patient_test == None: # if not given, all the patients are included
            self.patient_test = os.listdir(os.path.join(self.root, 'preseizure'))

        for key, dp in all_path_normal_sleep.items():
            if key not in self.patient_test:
                result = self.up_sampling(dp, resampling_size_normal)
                for p in result.items():
                    sleep_normal.append(p)
            else:
                result = self.up_sampling(dp, self.resampling_base_size)
                for p in result.items():
                    val_sleep_normal.append(p)

        for key, dp in all_path_preseizure.items():  # 获取的是所有的癫痫发前数据
            # 使用重采样的方法
            if key not in self.patient_test:
                result = self.up_sampling(dp, resampling_size_preseizure)
                for p in result.items():
                    sleep_pre_seizure.append(p)  # 加入的是字典
            else:
                result = self.up_sampling(dp, self.resampling_base_size)
                for p in result.items():
                    val_sleep_pre_seizure.append(p)

        random.shuffle(sleep_normal)
        random.shuffle(sleep_pre_seizure)  # 用于训练的数据集
        random.shuffle(val_sleep_normal)  # 用于验证的数据集
        random.shuffle(val_sleep_pre_seizure)

        self.save_resampling_data(sleep_normal, sleep_pre_seizure, val_sleep_normal, val_sleep_pre_seizure)
