from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline

from vacances_scolaires_france import SchoolHolidayDates
from jours_feries_france import JoursFeries


def get_covid_data(X_col):
    # dates were retrieved from here: https://en.wikipedia.org/wiki/COVID-19_pandemic_in_France
    first_lockdown = pd.date_range(start="2020-10-30", end="2020-12-15")
    second_lockdown = pd.date_range(start="2021-03-20", end="2021-06-09")
    combined = first_lockdown.union(second_lockdown)
    return X_col.dt.date.isin(combined.date)


def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
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

    # get covid lockdown data
    X.loc[:, "covid_lockdown"] = get_covid_data(X["date"])

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


def _merge_external_data(X):
    # external weather data was downloaded here:
    # https://meteostat.net/pt/place/fr/paris?s=07156&t=2022-10-25/2022-11-01
    # and here https://github.com/CSSEGISandData/COVID-19
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
    # prepare pipeline steps for data merge and date columns creation
    merge_data = FunctionTransformer(_merge_external_data)
    date_encoder = FunctionTransformer(_encode_dates)

    # define all columns that are used for the model
    date_cols = ["year", "month", "weekday", "hour", "season"]
    num_cols = ["temp", "prcp", "wspd", "daily_covid_cases"]
    categorical_cols = ["counter_name", "wdir"]
    bin_cols = ["public_holiday", "school_holiday", "covid_lockdown"]

    # column transformer to one hot encode and standardize columns
    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("binary", "passthrough", bin_cols),
        ],
    )

    # catboost regressor with parameters obtained by gridsearch
    regressor = CatBoostRegressor(
        depth=9,
        iterations=1000,
        rsm=0.25,
        sampling_frequency="PerTree",
        subsample=0.7,
        verbose=0,
    )

    # define final pipeline
    pipe = make_pipeline(
        merge_data,
        date_encoder,
        preprocessor,
        regressor,
    )

    return pipe
