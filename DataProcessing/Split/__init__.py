# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 03.03.2022

""" Perform splitting train/test for EEG and SEEG Data."""

from DataProcessing.Split.Base import BaseSplitter
from DataProcessing.Split.SEEG import SEEGSpliter

__all__ = [
    'SEEGSpliter'
]
