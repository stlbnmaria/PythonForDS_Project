import sys
import xgboost as xgb

sys.path.insert(0, "..")
from tuning import tuning_estimator


if __name__ == "__main__":
    estimator = xgb.XGBRegressor()
    grid = {
        "": [],
    }
    path = "./results_tuning"
    filename = "tuning_xgb"
    tuning_estimator(estimator, grid, path, filename, n_jobs=4)
