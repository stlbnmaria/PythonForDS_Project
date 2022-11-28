import os
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from catboost import CatBoostRegressor
from sklearn.preprocessing import FunctionTransformer
from joblib import dump
import pandas as pd
from datetime import datetime

sys.path.insert(0, "../..")
import problem


def tuning_estimator(estimator, grid, path, filename: str, n_jobs: int = 1):
    # get timestamp for file name in the end
    timestamp = datetime.today().strftime("%Y%m%d_%H%M")

    # date variables specifications
    date_encoder = FunctionTransformer(
        problem._encode_dates, kw_args={"drop_date": False}
    )

    # additional data variables specifications
    add_date_encoder = FunctionTransformer(
        problem._additional_date_variables, kw_args={"drop_date": True}
    )

    # numerical variables in X
    num_cols = ["temp", "dwpt", "rhum", "prcp", "wspd", "pres"]

    # categorical variables in X
    categorical_cols = [
        "year",
        "month",
        "weekday",
        "hour",
        "counter_name",
        "wdir",
        "season",
    ]

    # create pipe incl. merging weather data
    pipe = make_pipeline(
        FunctionTransformer(problem._merge_external_data, validate=False),
        date_encoder,
        add_date_encoder,
        FunctionTransformer(lambda x: x[num_cols + categorical_cols]),
        estimator,
    )

    # get training data from problem, testing is reserved for trials in notebook
    X_train, y_train = problem.get_train_data(path="../..")

    # get cv time series split
    cv = problem.get_cv(X_train, y_train)

    # performing grid search on pipe
    clf = GridSearchCV(
        pipe,
        grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        return_train_score=True,
        n_jobs=n_jobs,
        verbose=10,
    )
    clf.fit(X_train, y_train, catboostregressor__cat_features=categorical_cols)

    # saving cv_results
    results = pd.DataFrame(clf.cv_results_)
    results.to_csv(
        os.path.join(path, f"{timestamp}_{filename}_cv_results.csv"), index=False
    )


if __name__ == "__main__":
    estimator = CatBoostRegressor()

    grid = {
        "catboostregressor__iterations": [300],
        "catboostregressor__subsample": [0.6, 0.7, 0.8, 0.9, 1],
        "catboostregressor__sampling_frequency": ["PerTree", "PerTreeLevel"],
        "catboostregressor__depth": [6, 7, 8, 9, 10],
        "catboostregressor__rsm": [0.6, 0.7, 0.8, 0.9, 1],
    }

    path = "./results_tuning"
    filename = "tuning_cat"
    tuning_estimator(estimator, grid, path, filename, n_jobs=6)
