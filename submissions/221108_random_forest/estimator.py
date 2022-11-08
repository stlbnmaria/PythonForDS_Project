from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

from vacances_scolaires_france import SchoolHolidayDates
from jours_feries_france import JoursFeries


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


def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["year", "month", "day", "weekday", "hour", "season"]

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name"]

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", categorical_encoder, categorical_cols),
        ]
    )
    regressor = RandomForestRegressor(max_features="sqrt", n_jobs=4, max_samples=0.5)

    pipe = make_pipeline(
        date_encoder,
        preprocessor,
        regressor,
    )

    return pipe
