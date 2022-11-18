import sys
from xgboost import XGBRegressor

sys.path.insert(0, "..")
from tuning import tuning_estimator


if __name__ == "__main__":
    estimator = XGBRegressor()
    grid = {
        "xgbregressor__n_estimators": [100, 200],        
        "xgbregressor__colsample_bytree": [.8, 1],
        "xgbregressor__colsample_bylevel": [.8, 1],
        "xgbregressor__learning_rate": [.01, .1, .2],
        "xgbregressor__max_depth": [4, 6, 8],
        "xgbregressor__reg_alpha": [0, .01],
        "xgbregressor__subsample": [.6, .8, 1],
    }
    path = "./results_tuning"
    filename = "tuning_xgb"
    tuning_estimator(estimator, grid, path, filename)
