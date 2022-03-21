# @Time   : 2022/3/9
# @Author : Yiqiao Wang


"""
config.configurator
################################
"""

import re
import os
import sys
import yaml
from logging import getLogger


class Config(object):
    """ Configurator module that load the defined parameters.

    """

    def __init__(self, model=None, dataset=None, config_file_list=None, config_dict=None):
        """
        Args:
            model (str/AbstractClass): the model name or the model class, default is None, if it is None, config
            will search the parameter 'model' from the external input as the model name or model class.
            dataset (str): the dataset name, default is None, if it is None, config will search the parameter 'dataset'
            from the external input as the dataset name.
            config_file_list (list of str): the external config file, it allows multiple config files, default is None.
            config_dict (dict): the external parameter dictionaries, default is None.
        """

