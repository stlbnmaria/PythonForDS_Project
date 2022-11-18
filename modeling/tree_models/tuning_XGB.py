import sys
from xgboost import XGBRegressor

sys.path.insert(0, "..")
from tuning import tuning_estimator


if __name__ == "__main__":
    estimator = XGBRegressor()
    grid = {
        "xgbregressor__n_estimators": [100, 200],
        "xgbregressor__colsample_bytree": [0.8, 1],
        "xgbregressor__colsample_bylevel": [0.8, 1],
        "xgbregressor__learning_rate": [0.01, 0.1, 0.2],
        "xgbregressor__max_depth": [4, 6, 8],
        "xgbregressor__reg_alpha": [0, 0.01],
        "xgbregressor__subsample": [0.6, 0.8, 1],
    }
    path = "./results_tuning"
    filename = "tuning_xgb"
    tuning_estimator(estimator, grid, path, filename)
