import sys

from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

sys.path.insert(0, "../..")
import problem


def main():
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
        "randomforestregressor__n_estimators": [200, 300],
        "randomforestregressor__max_samples": [0.5, 0.6, 0.7],
        "randomforestregressor__max_depth": [10, 20, 40],
        "randomforestregressor__max_features": ["sqrt", "log2"],
    }

    # perform grid search
    clf = GridSearchCV(
        pipe,
        grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        return_train_score=True,
        n_jobs=4,
        verbose=1,
    )
    search = clf.fit(X_train, y_train)

    # saving the best model
    dump(search, "tuning_RF_grid_search.pkl")

    # saving the full pipeline
    dump(search.best_estimator_, "tuning_RF_best_estimator.pkl")


if __name__ == "__main__":
    main()
