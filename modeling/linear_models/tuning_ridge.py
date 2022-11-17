import sys
import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, "..")
from tuning import tuning_estimator


if __name__ == "__main__":
    estimator = Ridge()
    alphas_tested = np.arange(0.01, 1, step=0.01)
    grid = {"ridge__alpha": alphas_tested}
    path = "./results_tuning"
    filename = "tuning_ridge"
    tuning_estimator(estimator, grid, path, filename)
