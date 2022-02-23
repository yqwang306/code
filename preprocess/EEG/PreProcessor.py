from preprocess.base import BasePreProcessor

class EEGPreProcessor(BasePreProcessor):

    @classmethod
    def filter_noise(cls):
        print("call filter_noise() from EEGPreProcessor")
        # TODO
        pass