import sys
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, "..")
from tuning import tuning_estimator


if __name__ == "__main__":
    estimator = RandomForestRegressor()
    grid = {
        "randomforestregressor__n_estimators": [200, 400],
        "randomforestregressor__max_samples": [0.4, 0.6, 0.8],
        "randomforestregressor__max_depth": [20, 40, 60, 80],
        "randomforestregressor__max_features": ["sqrt", 0.2],
    }
    path = "./results_tuning"
    filename = "tuning_RF"
    tuning_estimator(estimator, grid, path, filename, n_jobs=4)
