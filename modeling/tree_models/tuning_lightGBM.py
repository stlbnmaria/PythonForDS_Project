import sys
from lightgbm import LGBMRegressor

sys.path.insert(0, "..")
from tuning import tuning_estimator


if __name__ == "__main__":
    estimator = LGBMRegressor()

    grid = {
        "lgbmregressor__n_estimators": [300],
        "lgbmregressor__subsample": [.6, .7, .8, .9, 1],

        "lgbmregressor__learning_rate": [0.1],
        "lgbmregressor__colsample_bytree": [.5, .6, .7, .8],
        "lgbmregressor__colsample_bynode": [.6, .8, .9],
        "lgbmregressor__max_depth": [4, 6, 8, 10],

        "lgbmregressor__min_child_weight": [0.1, 1],
    }

    path = "./results_tuning"
    filename = "tuning_lightGBM"
    tuning_estimator(estimator, grid, path, filename, n_jobs=4)
