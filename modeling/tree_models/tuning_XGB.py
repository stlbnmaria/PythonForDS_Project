import sys
from xgboost import XGBRegressor

sys.path.insert(0, "..")
from tuning import tuning_estimator


if __name__ == "__main__":
    estimator = XGBRegressor(tree_method="hist")

    grid = {
        "xgbregressor__n_estimators": [600],
        "xgbregressor__learning_rate": [0.1],
        "xgbregressor__colsample_bytree": [.4, .5, .6,],
        "xgbregressor__colsample_bylevel": [.6, .8],
        "xgbregressor__colsample_bynode": [.8, .9],
        "xgbregressor__max_depth": [4, 6, 8, 10],
        "xgbregressor__subsample": [.6, .7, .8, .9, 1],
        "xgbregressor__min_child_weight": [1, 10],
    }

    path = "./results_tuning"
    filename = "tuning_xgb_reduced_higher"
    tuning_estimator(estimator, grid, path, filename, n_jobs=6)
