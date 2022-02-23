# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 09.01.2022

import mne
import os

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