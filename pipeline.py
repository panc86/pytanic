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
    tmp = X.copy(deep=True)
    # impute missing cat Embarked with mode
    tmp.loc[tmp["Embarked"].isna(), "Embarked"] = tmp["Embarked"].value_counts().idxmax()
    # impute missing numeric
    numeric = tmp.select_dtypes(exclude="category").columns
    tmp.loc[:, numeric] = KNNImputer().fit_transform(tmp[numeric])
    # add cabin indicator of missingness
    tmp["CabinInd"] = features.cabin_indicator(tmp)
    # frequency encoding ticket
    ticket_map = tmp["Ticket"].value_counts()
    tmp["TicketFreq"] = tmp["Ticket"].map(ticket_map)
    # transform name into name title and size
    tmp["Title"] = features.passenger_title(tmp)
    tmp["NameSize"] = features.passenger_name_size(tmp)
    # capture marriage
    tmp["IsMarried"] = tmp["Title"] == "mrs"
    # simplify sibsp and parch into binary
    # capture family size
    family_size = tmp["SibSp"] + tmp["Parch"]
    tmp["IsAlone"] = family_size == 0
    tmp["IsSmallFamily"] = (0 < family_size) & (family_size < 3)
    tmp["IsLargeFamily"] = family_size > 2
    # simplify fares as the digits
    tmp["FareAboveStd"] = tmp.Fare > tmp.Fare.std()
    tmp["FareDigits"] = tmp["Fare"].astype(int).astype(str).apply(len)
    # capture children in variable
    tmp["IsChild"] = tmp["Age"] < 15
    # map sex to binary
    tmp["IsFemale"] = tmp["Sex"] == "female"
    # prepare for training
    X_tr = tmp.drop(
        columns=["Age", "Sex", "Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin"]
    ).pipe(pd.get_dummies, ["Pclass", "Embarked", "Title"])
    return X_tr


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
