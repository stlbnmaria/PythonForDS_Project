from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline

from vacances_scolaires_france import SchoolHolidayDates
from jours_feries_france import JoursFeries

categorical_feat = [
    "year",
    "month",
    "weekday",
    "hour",
    "season",
    "counter_name",
    "wdir",
]

num_cols = ["temp", "dwpt", "rhum", "prcp", "wspd", "pres"]


from sklearn.pipeline import Pipeline

if not hasattr(Pipeline, "_original_fit"):
    # Here, we replace the `fit` method with a monkey patch (a modification of it at runtime)
    # This allows modifying the behavior of `fit` (here we inject the categorical features
    # information) while respecting the API contract used by RAMP.

    # Store the previous implementation of `fit` in a private method.
    Pipeline._original_fit = Pipeline.fit

    # Define the new behavior for `fit`
    def monkey_patch_for_fit(self, X, y, *args):
        # Injecting the information of categorical before calling the original fit method.
        # categorical_feat is not passed as a parameter but should be in the closure
        # of the method.
        return self._original_fit(
            X, y, catboostregressor__cat_features=categorical_feat
        )

    # Patch the `fit` method at runtime.
    Pipeline.fit = monkey_patch_for_fit


def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # add seasons
    seasons = {
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "autumn",
        10: "autumn",
        11: "autumn",
        12: "winter",
    }
    X.loc[:, "season"] = X["date"].dt.month.map(seasons)

    public_holidays = []
    school_holidays = {}
    for year in X["date"].dt.year.unique():
        public_holidays.extend(JoursFeries.for_year(year).values())
        school_holidays.update(
            SchoolHolidayDates().holidays_for_year_and_zone(year, "C")
        )

    # add public holidays
    X.loc[:, "public_holiday"] = X["date"].isin(public_holidays)

    # add school holidays
    school_holidays_bool = [
        k for k, v in school_holidays.items() if v["vacances_zone_c"]
    ]
    X.loc[:, "school_holiday"] = X["date"].isin(school_holidays_bool)

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


def _merge_external_data(X):
    file_path = Path(__file__).parent / "external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(X.sort_values("date"), df_ext.sort_values("date"), on="date")
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X


def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)

    regressor = CatBoostRegressor()

    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder,
        FunctionTransformer(lambda x: x[num_cols + categorical_feat]),
        regressor,
    )

    return pipe
