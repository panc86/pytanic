import numpy as np
import pandas as pd
# PIPELINES
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
# MODELS
from sklearn.linear_model import LogisticRegression
# CUSTOM
import features


def transform_titanic(X):
    """
    Transform Titanic data into ready-to-train dataset.
    """
    # impute missing Ages
    numeric = X.select_dtypes(exclude="category").columns
    X.loc[:, numeric] = KNNImputer().fit_transform(X[numeric])
    # add cabin indicator of missingness
    X["CabinInd"] = features.cabin_indicator(X)
    # frequency encoding ticket
    ticket_map = X["Ticket"].value_counts()
    X["TicketFreq"] = X["Ticket"].map(ticket_map)
    # transform name into name title and size
    X["Title"] = features.passenger_title(X)
    X["NameSize"] = features.passenger_name_size(X)
    # capture marriage
    X["IsMarried"] = X["Title"] == "mrs"
    # simplify sibsp and parch into binary
    # capture family size
    family_size = X["SibSp"] + X["Parch"]
    X["IsAlone"] = family_size == 0
    X["IsSmallFamily"] = (0 < family_size) & (family_size < 3)
    X["IsLargeFamily"] = family_size > 2
    # simplify fares as the digits
    X["FareAboveStd"] = X.Fare > X.Fare.std()
    X["FareDigits"] = X["Fare"].astype(int).astype(str).apply(len)
    # capture children in variable
    X["IsChild"] = X["Age"] < 15
    # map sex to binary
    X["IsFemale"] = X["Sex"] == "female"
    # prepare for training
    X = X.drop(
        columns=["Age", "Sex", "Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin"]
    ).pipe(pd.get_dummies, ["Pclass", "Embarked", "Title"], dummy_na=True)
    return X


class TitanicProcessor(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self, X):
        # assume X as the dataset from read_csv() with the followings:
        #  - updated feature types
        #  - removed target Survived
        return transform_titanic(X)


def titanic_pipeline(random_state=42):
    """
    Return a Pipeline skeleton and connected grid parameters.
    TN: returned Pipeline is tuned and unfitted.
    """
    # grid search hyperparameters
    hyper_params = [
        {
            "rfe__estimator": [LogisticRegression()],
            "rfe__estimator__C": np.logspace(-3, 1, 11),
            "rfe__n_features_to_select": np.linspace(.1, .8, 11),
            "rfe__estimator__random_state": [random_state],
            "clf": [LogisticRegression()],
            "clf__C": np.logspace(-3, 1, 11),
            "clf__solver": ["liblinear"],
            "clf__penalty": ["l1", "l2"],
            "clf__random_state": [random_state],
        },
    ]
    titanic_pipeline = Pipeline(steps=[
        ("prep", TitanicProcessor()),
        ("std", StandardScaler()),
        ("rfe", RFE(LogisticRegression())),
        ("clf", LogisticRegression())
    ])
    return titanic_pipeline, hyper_params


def tuning(pipeline, params, X, y, cv=3, verbose=0):
    gs = GridSearchCV(pipeline, params, cv=cv, n_jobs=-1, verbose=verbose).fit(X, y)
    pipeline.set_params(**gs.best_params_)
    print("GridSearchCV best score:", gs.best_score_)
