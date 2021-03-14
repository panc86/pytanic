# PIPELINES
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, KBinsDiscretizer
)
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
# MODELS
from sklearn.linear_model import LogisticRegression


def baseline_pipeline(numeric, categorical, random_state=42):
    # grid search hyperparameters
    hyperparams = [
        {
            "transformer__num__imputer__strategy": ["mean", "median"],
            "transformer__cat__imputer__strategy": ["most_frequent"],
            "transformer__cat__onehot__drop": ["if_binary", "first"]
        },
        {
            "transformer__num__imputer__strategy": ["mean", "median"],
            "transformer__cat__imputer__strategy": ["constant"],
            # If left to the default, fill_value will be 0 when imputing numerical data and “missing_value” for strings or object data types.
            "transformer__cat__onehot__drop": ["if_binary", "first"]
        }
    ]
    # numeric
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    # categorical
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='if_binary')),
    ])
    # transformer
    transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric),
            ('cat', categorical_transformer, categorical)
        ]
    )
    # define pipeline
    pipeline = Pipeline(steps=[
            ("transformer", transformer),
            ("classifier", LogisticRegression(random_state=random_state))
        ])
    # run pipeline assessment
    return pipeline, hyperparams
