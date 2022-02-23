# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 09.01.2022

"""Preprocessing Pipeline for EEG and SEEG Data."""

from preprocess.SEEG.PreProcessor import SEEGPreProcessor
from preprocess.EEG.PreProcessor import EEGPreProcessor

# from preprocess.EEG.EEGPreProcessor import EEGPreProcessor
__all__ = [
    'SEEGPreProcessor',
    'EEGPreProcessor'
]

print("import PreProcessor, load SEEGPreProcessor and EEGPreProcessor")
