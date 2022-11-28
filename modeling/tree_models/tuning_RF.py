import sys
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, "..")
from tuning import tuning_estimator


if __name__ == "__main__":
    estimator = RandomForestRegressor()
    grid = {
        "randomforestregressor__n_estimators": [300],
        "randomforestregressor__max_samples": [0.4, 0.5, 0.6],
        "randomforestregressor__max_depth": [30, 40],
        "randomforestregressor__max_features": ["sqrt", 0.15, 0.25],
    }
    path = "./results_tuning"
    filename = "tuning_RF"
    tuning_estimator(estimator, grid, path, filename, n_jobs=4)
