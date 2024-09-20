from abc import ABC, abstractmethod
class learner(ABC):

    @abstractmethod
    def __init__(self, query_function, initial_x, initial_y):
        pass

    @abstractmethod
    def teach(self, dataset_x, dataset_y):
        #appends data to the train set
        pass
    
    @abstractmethod
    def query(self, dataset, howmany):
        pass

    @abstractmethod
    def estimate(self, dataset):
        pass
    

    def getName(self):
        return self.name