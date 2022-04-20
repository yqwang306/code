
from abc import ABC, abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def evaluate_model(self):
        pass

    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def calculate_loss(self):
        pass

    @abstractmethod
    def __init__(self):
        pass


