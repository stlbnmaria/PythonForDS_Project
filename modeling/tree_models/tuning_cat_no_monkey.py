import sys
from catboost import CatBoostRegressor

sys.path.insert(0, "..")
from tuning import tuning_estimator


if __name__ == "__main__":
    estimator = CatBoostRegressor()

    grid = {
        "catboostregressor__iterations": [500, 750, 1000],
        "catboostregressor__learning_rate": [0.05, 0.1, 0.2],
        "catboostregressor__subsample": [0.4, 0.5, 0.6, 0.7],
        "catboostregressor__sampling_frequency": ["PerTree"],
        "catboostregressor__depth": [4, 5, 6, 7],
        "catboostregressor__rsm": [0.4, 0.5, 0.6, 0.7],
    }

    path = "./results_tuning"
    filename = "tuning_cat_no_monkey"
    tuning_estimator(estimator, grid, path, filename, n_jobs=6)
