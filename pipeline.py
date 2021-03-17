import numpy as np
import pandas as pd
# PIPELINES
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler
)
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# MODELS
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_validate, learning_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC


def get_column_types(X):
    # get categorical features
    cat_cols = X.select_dtypes(include="category").columns
    # get non categorical (numeric)
    num_cols = X.select_dtypes(exclude="category").columns
    return num_cols, cat_cols


def get_cv_scores(pipeline, X, y, cv=10):
    print("CV scoring...")
    cv_scores = cross_validate(
        pipeline, X, y, cv=cv, return_train_score=True,
        scoring=["accuracy", "roc_auc"], n_jobs=-1
    )
    return pd.DataFrame(cv_scores)


def tune(pipeline, X, y, params, cv=10):
    print("Running RandomizedSearchCV...")
    rs = RandomizedSearchCV(
        pipeline, params, cv=cv, n_jobs=-1, refit=False
    ).fit(X, y)
    # set best hyper-parameters
    pipeline.set_params(**rs.best_params_)
    return pipeline, rs.best_params_


def build_pipeline(numeric, categorical, random_state=42):
    """
    Return a Pipeline skeleton and connected grid parameters.
    """
    print("Building pipeline...")
    # grid search hyperparameters
    grid_params = [
        {
            "transformer__num__imputer__strategy": ["mean", "median"],
            "transformer__cat__imputer__strategy": ["most_frequent"],
            "transformer__cat__onehot__drop": ["if_binary"],
            "feature_selection__C": np.logspace(-2, 2, 20),
            "feature_selection__penalty": ["l1"],
            "feature_selection__dual": [True, False],
            "classifier": [LogisticRegression()],
            "classifier__C": np.logspace(-2, 2, 20),
            "classifier__random_state": [random_state]
        },
        {
            "transformer__num__imputer__strategy": ["mean", "median"],
            "transformer__cat__imputer__strategy": ["most_frequent"],
            "transformer__cat__onehot__drop": ["if_binary"],
            "feature_selection__C": np.logspace(-2, 2, 20),
            "feature_selection__penalty": ["l1"],
            "feature_selection__dual": [True, False],
            "classifier": [SVC()],
            "classifier__probability": [True],
            "classifier__gamma": np.logspace(-3, 1, 30),
            "classifier__C": np.logspace(-2, 2, 20),
            "classifier__random_state": [random_state]
        }
    ]
    # numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="mean")),
        ('scaler', StandardScaler())
    ])
    # categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('onehot', OneHotEncoder(drop="if_binary")),
    ])
    # features transformer
    transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric),
            ('cat', categorical_transformer, categorical)
        ]
    )
    # final pipeline
    pipeline = Pipeline(steps=[
            ("transformer", transformer),
            ('feature_selection', SelectFromModel(LinearSVC())),
            ("classifier", LogisticRegression(random_state=random_state))
        ])
    return pipeline, grid_params


def init_pipeline(X, y, cv=10, random_state=42):
    """
    Return a tuned Pipeline, CV scores report, and best GridSearchCV parameters.
    The tuned pipeline is made of the following steps:
      - transformer: to prepare raw data into ML ingestable format
      - classifier: estimator that implements fit() and predict() methods.
    TN: returned Pipeline is tuned and unfitted.
    """
    # extract columns by types
    numeric, categorical = get_column_types(X)
    # build pipeline skeleton and hyperparams
    pl, hp = build_pipeline(numeric, categorical, random_state=random_state)
    # tune pipeline steps
    pl, best_hp = tune(pl, X, y, hp, cv=cv)
    # evaluate pipeline perfs
    scores = get_cv_scores(pl, X, y, cv=cv)
    return pl, scores, best_hp


def run_transformer(X, y, pl: Pipeline):
    """
    Return transformed X as Pandas DataFrame, i.e. X goes through all transformer steps, `transformer` Pipeline step for X_test transformation, and names of new features.
    TN: Pipeline argument is un unfitted Pipeline created with init_pipeline().
    """
    print("Use `transformer` pipeline step for dataset transformation...")
    # fit to data to enable transform() method
    transformer = pl["transformer"].fit(X, y)
    # extract columns by types
    numeric, categorical = get_column_types(X)
    # get transformed feature names (only after fit)
    onehot_cols = (transformer
                .named_transformers_["cat"]
                .named_steps["onehot"]
                .get_feature_names(categorical))
    cols = [*numeric, *onehot_cols]
    X_tr = pd.DataFrame(transformer.transform(X), columns=cols)
    return X_tr, transformer, cols


# def run_classifier(X, y, cv=3, n_jobs=-1, random_state=42):
#     clf_pl, grid_params = init_training_pipeline(random_state=random_state)
#     # tune with best parameters
#     best_params, score = tune(clf_pl, grid_params, X, y, cv=cv, n_jobs=n_jobs)
#     clf_pl.set_params(**best_params)
#     # fit to data to enable prediction related methods
#     return clf_pl["classifier"], best_params, score

