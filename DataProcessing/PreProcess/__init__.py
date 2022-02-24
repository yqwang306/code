# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 09.01.2022

"""Preprocessing Pipeline for EEG and SEEG Data."""

from DataProcessing.PreProcess.Base import BasePreProcessor
from DataProcessing.PreProcess.EEG import EEGPreProcessor
from DataProcessing.PreProcess.SEEG import SEEGPreProcessor

__all__ = [
    'SEEGPreProcessor',
    'EEGPreProcessor'
]
