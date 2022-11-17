from joblib import dump
import os
import sys
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
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
    date_cols = ["year", "month", "day", "weekday", "hour"]

    # additional data variables specifications
    add_date_encoder = FunctionTransformer(
        problem._additional_date_variables, kw_args={"drop_date": True}
    )
    add_date_cols = ["season"]

    # transformer to drop certain columns of the dataframe
    drop_cols = [
        "counter_id",
        "site_id",
        "counter_installation_date",
        "counter_technical_id",
    ]
    drop_transformer = FunctionTransformer(
        problem._drop_cols, kw_args={"cols": drop_cols}, validate=False
    )

    # categorical variables in X specifications
    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name", "wdir"]

    # create column transformer with all one hot encoders
    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols + add_date_cols),
            ("cat", categorical_encoder, categorical_cols),
        ],
        remainder="passthrough",
    )

    # create pipe incl. merging weather data
    pipe = make_pipeline(
        FunctionTransformer(problem._merge_external_data, validate=False),
        date_encoder,
        add_date_encoder,
        drop_transformer,
        preprocessor,
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
    clf.fit(X_train, y_train)

    # saving cv_results
    results = pd.DataFrame(clf.cv_results_)
    results.to_csv(
        os.path.join(path, f"{timestamp}_{filename}_cv_results.csv"), index=False
    )

    # saving the best model
    dump(
        clf.best_estimator_,
        os.path.join(path, f"{timestamp}_{filename}_best_estimator.pkl"),
    )
