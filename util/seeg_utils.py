# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 09.01.2022

import mne
import os
import numpy as np
import glob
import random

def read_raw(path):
    """
    :param path: the path of raw data
    :return: the loaded raw data
    """

    raw = mne.io.read_raw_fif(path, preload=True)
    return raw

def get_channels_names(raw):
    """
    :param raw: raw data loaded from the disk
    :return: the name of every channel
    """

    channel_names = raw.info['ch_names']
    return channel_names

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

def rewrite(raw, include_names, save_path):
    """
    rewrite raw data by extracting some specific channels

    :param raw: raw data loaded from the disk
    :param include_names: the name of every included channels
    :param save_path: a path for saving the processed data
    :return: extract the data from some specific channels
    """

    want_meg = True
    want_eeg = False
    want_stim = False

    picks = mne.pick_types(raw.info, meg=want_meg, eeg=want_eeg, stim=want_stim,
                           include=include_names, exclude='bads')
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

def read_raw(path):
    """
    load in raw data that stored in the given path

    :param path: string, path of raw data
    :return:
    """
    raw = mne.io.read_raw_fif(path, preload=True)
    return raw

def select_channel_data_mne(raw, select_channel_name):
    """
    reselect channel based on its order

    :param raw:
    :param select_channel_name:
    :return:
    """

    chan_name = select_channel_name
    specific_chans = raw.copy().pick_channels(chan_name)
    return specific_chans

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

def get_all_file_path(path, suffix='fif'):
    '''
    get the paths of all files under every sub-directory

    :param path: 存储对应文件的路径
    :return: map, {dir_path:[file1, file2, file3...]}文件夹下面对应的文件路径
    '''

    if os.path.exists(path) == False:
        return None
    file_map = {}
    path_dir = []
    dirs = os.listdir(path)
    for d in dirs:
        path_dir.append(os.path.join(path, d))
        new_suffix = '*.' + suffix
        file_p = glob.glob(os.path.join(d, new_suffix))
        file_map[d] = file_p
    return file_map

def generate_path_by_flag(flag, flag_duration):
    if flag == 0:  # pre-seizure
        if flag_duration == 0:
            dir = 'preseizure'
        else:
            if flag_duration == 1:
                dir = 'preseizure/within_warning_time'
            else:
                dir = 'preseizure/before_warning_time'
    else:
        if flag == 1:
            dir = "cases"  # attacking
        else:
            if flag == 2:
                dir = "sleep"
            else:
                dir = "awake"
    return dir

def get_flag_from_path(path):
    tmp = path.split('/')
    if tmp[-1] == 'preseizure':
        flag = 0
        flag_duration = 0
    elif tmp[-1] == 'within_warning_time':
        flag = 0
        flag_duration = 1
    elif tmp[-1] == 'before_warning_time':
        flag = 0
        flag_duration = 2
    elif tmp[-1] == 'cases':
        flag = 1
    elif tmp[-1] == 'sleep':
        flag = 2
    elif tmp[-1] == 'awake':
        flag = 3
    return flag, flag_duration

def relocate_file_in_dic(result_dic, save_dir):
    for name, path_f in result_dic:
        save_path = os.path.join(save_dir, name)
        data = np.load(path_f)
        np.save(save_path, data)
    print("Successfully writen sampling data to {}".format(save_dir))

def dir_create_check(path_dir):
    if os.path.exists(path_dir) is False:
        os.makedirs(path_dir)
        print("{} has been created!".format(path_dir))
    else:
        print("{} has existed!".format(path_dir))

def get_label_data(path):  # get data include label
    '''

    :param path:
    :return: {"path":1, "path2":2}
    '''
    class_name = os.listdir(path)
    data_name = []
    data_label = []
    for i, name in enumerate(class_name):
        new_path = os.path.join(path, name)
        data_file = os.listdir(new_path)
        path_file = [os.path.join(new_path, x) for x in data_file]
        data_name += path_file
        data_label += [i] * len(data_file)
    result_data_label = dict(zip(data_name, data_label))
    return result_data_label

def matrix_normalization(data, resize_shape=(130, 200)):
    '''
    矩阵的归一化，主要是讲不通形状的矩阵变换为特定形状的矩阵, 矩阵的归一化主要是更改序列
    也就是主要更改行
    eg:(188, 200)->(130, 200)   归一化的表示
    :param data:
    :param resize_shape:
    :return:
    '''
    data_shape = data.shape  # 这个必须要求的是numpy的文件格式
    if data_shape[0] != resize_shape[0]:
        if resize_shape[0] > data_shape[0]:  # 做插入处理
            '''
            扩大原来的矩阵
            '''
            d = resize_shape[0] - data_shape[0]
            channels_add = random.sample(range(1, data_shape[0] - 1), d)
            fake_channel = []  # 添加信道列表的值
            for c in channels_add:
                tmp = (data[c - 1] + data[c]) * 1.0 / 2
                fake_channel.append(tmp)
            data = np.insert(data, channels_add, fake_channel, axis=0)
        else:
            if resize_shape[0] < data_shape[0]:  # 做删除处理
                '''
                删除掉原来的矩阵
                '''
                d = data_shape[0] - resize_shape[0]
                channels_del = random.sample(range(1, data_shape[0] - 1), d)
                data = np.delete(data, channels_del, axis=0)
    return data
