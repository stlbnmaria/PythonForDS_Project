import sys
from catboost import CatBoostRegressor

sys.path.insert(0, "..")
from tuning import tuning_estimator


if __name__ == "__main__":
    estimator = CatBoostRegressor()

    grid = {
        "catboostregressor__iterations": [500, 750, 1000, 1200, 1400],
        "catboostregressor__subsample": [0.5, 0.6, 0.7, 0.8, 0.9],
        "catboostregressor__sampling_frequency": ["PerTree"],
        "catboostregressor__depth":  [6, 7, 8, 9, 10],
        "catboostregressor__rsm":  [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    }

    path = "./results_tuning"
    filename = "tuning_cat_no_monkey"
    tuning_estimator(estimator, grid, path, filename, n_jobs=-1)
