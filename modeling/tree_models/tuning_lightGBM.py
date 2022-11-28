import sys
from lightgbm import LGBMRegressor

sys.path.insert(0, "..")
from tuning import tuning_estimator


if __name__ == "__main__":
    estimator = LGBMRegressor()

    grid = {
        "lgbmregressor__n_estimators": [600, 800, 1000],
        "lgbmregressor__learning_rate": [0.1],
        "lgbmregressor__num_leaves": [10, 20, 30],
        "lgbmregressor__min_child_samples": [5, 10, 20],
        "lgbmregressor__subsample": [0.6, 0.7, 0.8],
        "lgbmregressor__subsample_freq": [1, 5],
        "lgbmregressor__colsample_bytree": [0.1, 0.2, 0.3, 0.4],
    }

    path = "./results_tuning"
    filename = "tuning_lightGBM_v2data"
    tuning_estimator(estimator, grid, path, filename, n_jobs=6)
