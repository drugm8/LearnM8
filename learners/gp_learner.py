from learners.learner_abc import learner
from learners.sklearn_learner import sklearn_learner
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor

class gp_learner(sklearn_learner):

    def __init__(self, query_function, dataset_x, dataset_y, batch_size):
        super().__init__(query_function, dataset_x, dataset_y, GaussianProcessRegressor(kernel=WhiteKernel() + RBF()), batch_size)
        self.name = "Gaussian Process Regressor"

    