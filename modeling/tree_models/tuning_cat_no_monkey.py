import sys
from catboost import CatBoostRegressor

sys.path.insert(0, "..")
from tuning import tuning_estimator


if __name__ == "__main__":
    estimator = CatBoostRegressor()

    grid = {
        "catboostregressor__iterations": [1000, 1200],
        "catboostregressor__subsample": [0.6, 0.7, 0.8],
        "catboostregressor__sampling_frequency": ["PerTree"],
        "catboostregressor__depth":  [9, 10, 11],
        "catboostregressor__rsm":  [0.05, 0.1, 0.15, 0.2, 0.25],
    }

    path = "./results_tuning"
    filename = "tuning_cat_no_mo_v2data"
    tuning_estimator(estimator, grid, path, filename, n_jobs=-1)
