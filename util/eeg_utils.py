# -*- coding: utf-8 -*-
# Author: Yiqiao Wang
# Date: 27.02.2022


import mne
import os
import numpy as np
import scipy.io


def read_raw_edf(path):
    """
    load in raw data that stored in the given path

    :param path: string, path of raw data
    :return:
    """

    raw = mne.io.read_raw_edf(path, preload=True)
    return raw

def read_raw_mat(path, ch_names):
    """
    load in raw data that stored in the given path

    :param path: string, path of raw data
    :param ch_names: list, names of channels
    :return:
    """

    data = scipy.io.loadmat(path)
    eeg_data = list(data.values())[3]

    eeg_data = np.array(eeg_data)
    eeg_max = np.amax(eeg_data, 1)
    eeg_min = np.min(eeg_data, 1)
    eeg_mean = np.mean(eeg_data, 1)

    for i in range(62):
        eeg_data[i] = (eeg_data[i] - eeg_min[i]) / (eeg_max[i] - eeg_min[i])

    ch_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6'
        , 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5'
        , 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2'
        , 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7'
        , 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

    info = mne.create_info(
        ch_names,
        ch_types=['eeg' for _ in range(62)],
        sfreq=200
    )

    raw = mne.io.RawArray(eeg_data, info)

    return raw

def get_channels_names(raw):
    """
    :param raw: raw data loaded from the disk
    :return: the name of every channel
    """

    channel_names = raw.info['ch_names']
    return channel_names

def rewrite(raw, include_names, save_path):
    """
    rewrite raw data by extracting some specific channels

    :param raw: raw data loaded from the disk
    :param include_names: the name of every included channels
    :param save_path: a path for saving the processed data
    :return: extract the data from some specific channels
    """

    picks = mne.pick_types(raw.info, include=include_names, exclude='bads')
    print("included channel names:{}".format(include_names))

    raw.save(save_path, picks=picks, overwrite=True)
    print("successfully written!")
    return True

def os_mkdir(save_dir, dir):
    """
    create some directories

    :param save_dir: string, root dir for saving processed data
    :param dir: string, new dir to be created
    :return:
    """

    new_path = os.path.join(save_dir, dir)
    if os.path.exists(new_path) is not True:
        os.makedirs(new_path)
        print("new dir has been created! {}".format(new_path))
    else:
        print("{} dir is existed!".format(new_path))

def filter_hz(raw, high_pass, low_pass):
    """
    Perform signal filtering, extract signal that ranges from (high_pass, low_pass)

    :param raw: raw data
    :param high_pass: float, maximum threshold
    :param low_pass: float, minimum threshold
    :return:
    """

    raw.filter(high_pass, low_pass, fir_design='firwin')
    return raw

def get_duration_raw_data(raw, start, stop):
    """
    :param raw: raw data loaded from the disk
    :param start: start time
    :param stop: end time
    :return: a segment of data, ranging from start time to end time
    """

    end = max(raw.times)
    if stop > end:
        print("out of range!!!")
        return None
    else:
        duration_data = raw.crop(start, stop)
        return duration_data

def split_data(raw, time_step):
    """
    Split the signal data into segments

    :param raw:
    :param time_step:
    :return:
    """

    data_split = []
    end = max(raw.times)
    epoch = int(end // time_step)
    fz = int(len(raw) / end)  # 采样频率
    for index in range(epoch - 1):
        start = index * fz * time_step
        stop = (index + 1) * fz * time_step
        data, time = raw[:, start:stop]
        data_split.append(data)
    return data_split

def save_numpy_info(data, path):
    """
    save numpy file

    :param data:
    :param path:
    :return:
    """
    if os.path.exists(path):
        print("File is exist!!!")
        return False
    else:
        np.save(path, data)
        print("Successfully save!")
        return True

def get_all_path(path_dir):
    all_path = []
    for p in os.listdir(path_dir):
        all_path.append(os.path.join(path_dir, p))
    return all_path

def bit_coding(data, step):
    """
    Binary encoding based on the number of spindles.

    :param data: serial information of a person
    :param step: window size
    :return: encoded sequence
    """
    code = []
    pre_data = 0
    count = 0
    length = len(data)
    while count < length:
        n = (data[count] - pre_data) / step
        if n > 0:
            if n > int(n):
                n = int(n)
                code += [0] * n + [1]
            else:
                n = int(n)
                code += [0] * (n - 1) + [1]
        pre_data = data[count]
        count += 1
    return code

def num_coding(data, step):
    """
    Coding based on number distribution.

    :param data: serial information of a person
    :param step: window size
    :return: encoded sequence
    """
    code = []
    pre_flag = step
    count = 0
    write_count = 0
    length = len(data)
    while count < length:
        if data[count] > pre_flag:
            code.append(write_count)
            pre_flag += step
            write_count = 0
        else:
            write_count += 1
            count += 1
    if write_count != 0:
        code.append(write_count)
    return code

def multiply(data1, data2):
    length = len(data1)
    sum = 0
    for index in range(length):
        sum += data1[index] * data2[index]
    return sum

def cos(data1, data2):
    d1 = multiply(data1, data2)
    d2 = math.sqrt(multiply(data1, data1)) * math.sqrt(multiply(data2, data2))
    result = d1 / d2
    return result