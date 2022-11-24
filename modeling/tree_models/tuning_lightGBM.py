import sys
from lightgbm import LGBMRegressor

sys.path.insert(0, "..")
from tuning import tuning_estimator


if __name__ == "__main__":
    estimator = LGBMRegressor()

    grid = {
        "lgbmregressor__n_estimators": [400, 600, 800],
        "lgbmregressor__learning_rate": [0.1, 0.05],
        "lgbmregressor__num_leaves": [30, 50, 70],
        "lgbmregressor__min_child_samples": [10, 20, 30],
        "lgbmregressor__subsample": [0.6, 0.8, 1],
        "lgbmregressor__subsample_freq": [1, 10],
        "lgbmregressor__colsample_bytree": [0.4, 0.6, 0.8, 1],
    }

    path = "./results_tuning"
    filename = "tuning_lightGBM"
    tuning_estimator(estimator, grid, path, filename, n_jobs=6)
