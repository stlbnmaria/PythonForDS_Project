import sys
from xgboost import XGBRegressor

sys.path.insert(0, "..")
from tuning import tuning_estimator


if __name__ == "__main__":
    estimator = XGBRegressor()

    grid = {
        "xgbregressor__n_estimators": [400],
        "xgbregressor__learning_rate": [0.1],
        "xgbregressor__colsample_bytree": [0.5, 0.6, 0.7],
        "xgbregressor__colsample_bylevel": [0.5, 0.6, 0.7],
        "xgbregressor__colsample_bynode": [0.5, 0.6, 0.7],
        "xgbregressor__max_depth": [9, 10],
        "xgbregressor__subsample": [0.8],
    }

    path = "./results_tuning"
    filename = "tuning_xgb"
    tuning_estimator(estimator, grid, path, filename)
