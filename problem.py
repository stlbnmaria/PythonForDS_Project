import os
from pathlib import Path
from jours_feries_france import JoursFeries
from vacances_scolaires_france import SchoolHolidayDates
import re
import numpy as np
import pandas as pd
import rampwf as rw
from sklearn.model_selection import TimeSeriesSplit

problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow
workflow = rw.workflows.EstimatorExternalData()

score_types = [
    rw.score_types.RMSE(name="rmse", precision=3),
]


def get_cv(X, y, random_state=0):
    cv = TimeSeriesSplit(n_splits=8)
    rng = np.random.RandomState(random_state)

    for train_idx, test_idx in cv.split(X):
        # Take a random sampling on test_idx so it's that samples are not consecutives.
        yield train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)


def _read_data(path, f_name):
    data = pd.read_parquet(os.path.join(path, "data", f_name))
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array


def get_train_data(path="."):
    f_name = "train.parquet"
    return _read_data(path, f_name)


def get_test_data(path="."):
    f_name = "test.parquet"
    return _read_data(path, f_name)


def _encode_dates(X, drop_date: bool = True):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    if drop_date:
        # Finally we can drop the original columns from the dataframe
        return X.drop(columns=["date"])
    else:
        return X


def _additional_date_variables(X, drop_date: bool = True, holiday_names=False):
    X = X.copy()  # modify a copy of X

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

    if holiday_names:
        # add school holidays names
        school_holidays_name = {
            k: re.sub("\s+|'", "_", re.sub("[éë]", "e", v["nom_vacances"].lower()))
            for k, v in school_holidays.items()
            if v["vacances_zone_c"]
        }
        X.loc[:, "school_holiday_name"] = X["date"].map(school_holidays_name)
    else:
        # add school holidays
        school_holidays_bool = [
            k for k, v in school_holidays.items() if v["vacances_zone_c"]
        ]
        X.loc[:, "school_holiday"] = X["date"].isin(school_holidays_bool)

    if drop_date:
        # Finally we can drop the original columns from the dataframe
        return X.drop(columns=["date"])
    else:
        return X


def merge_external_data(X, imputed_data=True):
    if imputed_data:
        f_name = "weather_data_imp.csv"
    else:
        f_name = "weather_data.csv"

    file_path = Path(__file__).parent / "data" / f_name
    df_ext = pd.read_csv(file_path, parse_dates=["date"])

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(X.sort_values("date"), df_ext.sort_values("date"), on="date")
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X
