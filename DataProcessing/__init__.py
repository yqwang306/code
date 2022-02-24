# -*- coding: utf-8 -*-
# Author: Nuo Ma
# Date: 24.02.2022

from DataProcessing import (
    PreProcess
)

from DataProcessing.PreProcess.Base import BasePreProcessor
from DataProcessing.PreProcess.EEG import EEGPreProcessor
from DataProcessing.PreProcess.SEEG import SEEGPreProcessor

# TODO: ADD Transfer part

__all__ = [
    'BasePreProcessor',
    'SEEGPreProcessor',
    'EEGPreProcessor'
]