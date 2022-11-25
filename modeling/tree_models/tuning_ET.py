import sys
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, "..")
from tuning import tuning_estimator


if __name__ == "__main__":
    estimator = ExtraTreesRegressor()

    grid = {
        "extratreesregressor__n_estimators": [10, 15],
        "extratreesregressor__max_samples": [0.7, 0.8, 0.9],
        "extratreesregressor__bootstrap": [True],
        "extratreesregressor__max_depth": [10, 20, 30, 40, 50],
        "extratreesregressor__max_features": [0.1, 0.2, 0.3, 0.4, 0.5],
    }

    path = "./results_tuning"
    filename = "tuning_et"
    tuning_estimator(estimator, grid, path, filename, n_jobs=4)
