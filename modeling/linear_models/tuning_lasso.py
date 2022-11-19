import sys
import numpy as np
from sklearn.linear_model import Lasso

sys.path.insert(0, "..")
from tuning import tuning_estimator


if __name__ == "__main__":
    estimator = Lasso()
    alphas_tested = np.arange(0.001, 0.1, step=0.001)
    grid = {"lasso__alpha": alphas_tested}
    path = "./results_tuning"
    filename = "tuning_lasso"
    tuning_estimator(estimator, grid, path, filename)
