import sys

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

sys.path.insert(0, "../..")
import problem


def main():
    X_train, y_train = problem.get_train_data(path="../..")

    cv = problem.get_cv(X_train, y_train)

    grid = {
        "randomforestregressor__n_estimators": [100, 200],
        "randomforestregressor__max_samples": [0.5],
    }  # "max_depth": [50, 100], "min_samples_leaf": [50, 100, 200],

    # 
    date_encoder = FunctionTransformer(
        problem._encode_dates, kw_args={"drop_date": False}
    )
    date_cols = ["year", "month", "day", "weekday", "hour"]

    add_date_encoder = FunctionTransformer(
        problem._additional_date_variables, kw_args={"drop_date": True}
    )
    add_date_cols = ["season"]

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name", "wdir"]

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols + add_date_cols),
            ("cat", categorical_encoder, categorical_cols),
        ]
    )

    regressor = RandomForestRegressor(max_features="sqrt")

    pipe = make_pipeline(
        FunctionTransformer(problem._merge_external_data, validate=False),
        date_encoder,
        add_date_encoder,
        preprocessor,
        regressor,
    )

    clf = GridSearchCV(
        pipe, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=4, verbose=1
    )
    search = clf.fit(X_train, y_train)

    print(search.cv_results_)
    


if __name__ == "__main__":
    main()
