import sys
import os

from datetime import datetime
from joblib import dump
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

sys.path.insert(0, "../..")
import problem


def main(path="./results_tuning"):
    path = "./results_tuning"
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

    # categorical variables in X specifications
    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name", "wdir"]

    # create column transformer with all one hot encoders
    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols + add_date_cols),
            ("cat", categorical_encoder, categorical_cols),
        ]
    )

    # used algorithm for grid search
    regressor = RandomForestRegressor()

    # create pipe incl. merging weather data
    pipe = make_pipeline(
        FunctionTransformer(problem._merge_external_data, validate=False),
        date_encoder,
        add_date_encoder,
        preprocessor,
        regressor,
    )

    # get training data from problem, testing is reserved for trials in notebook
    X_train, y_train = problem.get_train_data(path="../..")

    # get cv time series split
    cv = problem.get_cv(X_train, y_train)

    # define grid for parameter tuning
    grid = {
        "randomforestregressor__n_estimators": [200, 400],
        "randomforestregressor__max_samples": [0.4, 0.6, 0.8],
        "randomforestregressor__max_depth": [20, 40, 60, 80],
        "randomforestregressor__max_features": ["sqrt", 0.2],
    }

    # perform grid search
    clf = GridSearchCV(
        pipe,
        grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        return_train_score=True,
        n_jobs=4,
        verbose=10,
    )
    clf.fit(X_train, y_train)

    # saving cv_results
    results = pd.DataFrame(clf.cv_results_)
    results.to_csv(
        os.path.join(path, f"{timestamp}_tuning_RF_cv_results.csv"), index=False
    )

    # saving the best model
    dump(
        clf.best_estimator_,
        os.path.join(path, f"{timestamp}_tuning_RF_best_estimator.pkl"),
    )


if __name__ == "__main__":
    main()
