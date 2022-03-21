# -*- coding: utf-8 -*-
# Author: Nuo Ma, Yiqiao Wang
# Date: 24.02.2022, 09.03.2022

from util.seeg_utils import *
from util.eeg_utils import *
from DataProcessing.dataset.SpindleData import SpindleData
from DataProcessing.PreProcess.EEG import EEGPreProcessor

from abc import ABC, abstractmethod
import pandas as pd
import uuid
import keras.preprocessing as preprocessing


class BaseTransformer(ABC):
    data_name = None  # the name of the dataset
    data_type = None  # bi class or multi class
    data_path = None  # the path of all the raw data

    @abstractmethod
    def generate_data(self):
        pass

    @abstractmethod
    def save_data(self):
        pass

    @abstractmethod
    def save_segment(self):
        pass

    @abstractmethod
    def generate_save_path(self):
        pass


class SEEGBaseTransformer(BaseTransformer):
    flag = None  # 0: pre-seizure, 1: attacking stage, 2: normal sleep
    path_common_channel = None  # the names of common channels
    high_pass = 0
    low_pass = 30
    resample_freq = 100
    time = 2

    def __init__(self, data_name, data_type, data_path, flag, path_common_channel,
                 high_pass=0, low_pass=30, resample_freq=100, time=2):
        self.data_name = data_name
        self.data_type = data_type
        self.data_path = data_path
        self.flag = flag
        self.path_common_channel = path_common_channel
        self.high_pass = high_pass
        self.low_pass = low_pass
        self.resample_freq = resample_freq
        self.time = time

    def generate_data(self, flag_duration=0, isfilter=True):
        data = pd.read_csv(self.path_commom_channel, sep=',')
        d_list = data['chan_name']
        common_channels = list(d_list)
        self.save_data(common_channels, flag_duration=flag_duration, isfilter=isfilter)

    def save_data(self, common_channels, flag_duration, isfilter):
        raw = read_raw(self.data_path)
        if isfilter:
            raw = filter_hz(raw, self.high_pass, self.low_pass)
        raw.resample(self.resample_freq, npad="auto")
        raw = select_channel_data_mne(raw, common_channels)  # 根据特定的信道顺序来选择信道
        raw.reorder_channels(common_channels)  # 更改信道的顺序

        raw_split_data = split_data(raw, self.time)  # 进行2秒的切片
        print("split time {}".format(len(raw_split_data)))
        self.save_segment(raw_split_data)

    def save_segment(self, data_split):
        path = self.generate_save_path()
        if not os.path.exists(path):
            os.makedirs(path)
        for d in data_split:
            name = str(uuid.uuid1()) + "-" + str(self.flag)
            path_all = os.path.join(path, name)
            save_numpy_info(d, path_all)
        print("File save successfully {}".format(path))

    def generate_save_path(self):
        # TODO
        # path_dir = save_split_data_test__path_dir  # 通过配置文件来读取
        path_dir = ''
        if self.flag == 0:  # pre-seizure
            if self.flag_duration == 0:
                dir = 'preseizure'
            else:
                if self.flag_duration == 1:
                    dir = 'preseizure/within_warning_time'
                else:
                    dir = 'preseizure/before_warning_time'
        else:
            if self.flag == 1:
                dir = "cases"  # attacking
            else:
                if self.flag == 2:
                    dir = "sleep"
                else:
                    dir = "awake"

        path_dir = os.path.join(path_dir, dir)
        if os.path.exists(path_dir) is not True:
            os.makedirs(path_dir)
            print("create dir:{}".format(path_dir))
        path_person = os.path.join(path_dir, self.name)
        if os.path.exists(path_person) is not True:
            os.makedirs(path_person)
            print("create dir:{}".format(path_person))
        return path_person


class EEGBaseTransformer(BaseTransformer):
    flag = None  # 0: insomnia, 1: normal
    step = 0.002
    paths = []

    def __init__(self, data_name, data_type, data_path, flag, filter_length, is_align, step=0.002):
        self.data_name = data_name
        self.data_type = data_type
        self.data_path = data_path
        self.flag = flag
        self.filter_length = filter_length
        self.is_align = is_align
        self.step = step

    def generate_data(self, isfilter=True):
        path = self.save_data(isfilter=isfilter)
        return path

    def save_data(self, isfilter):
        spindle = SpindleData(self.data_path)
        labels = spindle.labels
        paths = spindle.paths
        processor = EEGPreProcessor(data_path=self.data_path, data_name=self.data_name, data_info=None,
                                    data_type=self.data_type)
        if isfilter:
            labels, paths = processor.filter_noise(filter_length=self.filter_length)

        all_data = []
        num_cases = 0
        num_controls = 0

        for i, p in enumerate(paths):
            if self.data_path == "../dataset/mesa_dataset/":
                if self.flag == 0:
                    if labels[i] == 0:
                        data = pd.read_csv(self.data_path + "cases/" + p, sep=",")
                        num_cases += 1
                    else:
                        continue
                else:
                    if labels[i] == 1:
                        data = pd.read_csv(self.data_path + "controls/" + p, sep=",")
                        num_controls += 1
                    else:
                        continue
            else:
                if self.flag == 0:
                    if labels[i] == 0:
                        data = pd.read_csv(self.data_path + "cases/" + p, skiprows=(0, 1), sep=",")
                        num_cases += 1
                    else:
                        continue
                else:
                    if labels[i] == 1:
                        data = pd.read_csv(self.data_path + "controls/" + p, skiprows=(0, 1), sep=",")
                        num_controls += 1
                    else:
                        continue

            data = data['Time_of_night']
            all_data.append(data)

        path = self.save_segment(all_data, num_cases, num_controls, is_align=self.is_align, paths=paths)
        return path

    def save_segment(self, data, num_cases, num_controls, is_align, paths):
        path = self.generate_save_path()
        if not os.path.exists(path):
            os.makedirs(path)

        coding_q = []
        for d in data:
            code = bit_coding(d, step=self.step)
            coding_q.append(code)
        if is_align:
            max_length = max([len(x) for x in coding_q])
            code_q = preprocessing.sequence.pad_sequences(coding_q, maxlen=max_length)
            coding_q = np.asarray(code_q)

        path = path + "/encoded_str.txt"
        f = open(path, 'w', encoding="UTF-8")

        for index, p in enumerate(coding_q):
            str_a = SpindleData.trans_list_str(p)
            if(len(data) == num_cases):
                f.write(paths[index] + ":")
            else:
                f.write(paths[len(paths)-num_controls+index] + ":")
            f.writelines(str_a)
            f.write("\n")

        f.close()
        print("File save successfully {}".format(path))
        return path

    def generate_save_path(self):
        # TODO
        # path_dir = save_split_data_test__path_dir  # 通过配置文件来读取
        path_dir = '../processed_data/'
        if self.flag == 0:  # insomnia
            dir = "cases"
        else:  # normal
            dir = "controls"

        path_dir = os.path.join(path_dir, dir)
        if os.path.exists(path_dir) is not True:
            os.makedirs(path_dir)
            print("create dir:{}".format(path_dir))
        path_dataset = os.path.join(path_dir, self.data_name)
        if os.path.exists(path_dataset) is not True:
            os.makedirs(path_dataset)
            print("create dir:{}".format(path_dataset))
        return path_dataset
