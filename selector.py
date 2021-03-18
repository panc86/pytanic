import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
from sklearn.model_selection import (
    cross_validate, RepeatedKFold, learning_curve
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV # default model for coefficient importance
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import rfpimp


def show_missing_values(train, test):
    """
    Return concatenated count of missing values for train and test sets.
    """
    na_dist = pd.concat([train.isna().sum(), test.isna().sum()], axis=1)
    na_dist.columns = ['train', 'test']
    return na_dist


def get_column_types(X):
    """
    Return numeric and categorical features into two separate lists.
    """
    # get categorical features
    cat_cols = X.select_dtypes(include="category").columns
    # get non categorical (numeric)
    num_cols = X.select_dtypes(exclude="category").columns
    return num_cols, cat_cols


def get_transformed_features(transformer, X):
    """
    Return the names of the transformed features.
    """
    # extract columns by types
    numeric, categorical = get_column_types(X)
    # get transformed feature names (only after fit)
    onehot_cols = (transformer
                .named_transformers_["cat"]
                .named_steps["onehot"]
                .get_feature_names(categorical))
    return [*numeric, *onehot_cols]


def get_cv_scores(model, X, y, cv=5, scoring=["accuracy", "roc_auc"], train_score=True, fit=False):
    """
    Return CV Accuracy and ROC scores as Pandas DataFrame for a given model.
     - `fit` allows to return the fitted model for each CV split
     - `train_score` return the train score together with the CV score.
    """
    cv_scores = cross_validate(
        model, X, y, cv=cv, scoring=scoring,
        return_train_score=train_score, return_estimator=fit, n_jobs=-1
    )
    return pd.DataFrame(cv_scores)


def plot_features_correlation(X, figsize=(7,6), info=""):
    """
    Plot Parwise correlation heatmap.
    """
    print("Plot the features correlation")
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        X.corr(), annot=True, fmt='.2f', ax=ax, vmin=-1, vmax=1, cmap='RdBu_r'
    )
    fig.savefig('img/correlation_'+info, bbox_inches='tight')
    plt.close(fig)


def plot_features_dependence(X, figsize=(7,6), info=""):
    """
    The scikit-learn Random Forest feature importances strategy is mean decrease in impurity (or gini importance) mechanism, which is unreliable. To get reliable results, use permutation importance, provided in the rfpimp package. Given training observation independent variables in a dataframe, compute the feature importance using each var as a dependent variable using a RandomForestRegressor or RandomForestClassifier. We retrain a random forest for each var as target using the others as independent vars. Only numeric columns are considered.
    # The dependence heatmap can be read as follows:
    # The features on the X axis predict the features on the Y axis, the higher the score, the higher the correlation.
    For more info, see https://github.com/parrt/random-forest-importances
    """
    print("Plot the features dependence")
    fig = plt.figure(figsize=figsize)
    rfpimp.plot_dependence_heatmap(rfpimp.feature_dependence_matrix(X))
    fig.savefig('img/rfpimp_dependence_'+info, bbox_inches='tight')
    plt.close(fig)


def plot_learning_curves(pipeline, X, y, cv=5, figsize=(10,5), info=""):
    """
    How the model learns on a growing number of training examples?
    Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """
    print("Plot the learning curves")
    intervals, train_scores, test_scores = learning_curve(
        pipeline, X, y, cv=cv, train_sizes=np.linspace(.1, 1., 5), n_jobs=-1
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # setup plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f"Learning Curves " + info)
    ax.set_xlabel('No. Training Examples')
    ax.set_ylabel('Scores')
    ax.fill_between(intervals, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.2,
                        color="r")
    ax.fill_between(intervals, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.2,
                        color="g")
    ax.plot(intervals, train_scores_mean, 'o-', color="r", label='Training score')
    ax.plot(intervals, test_scores_mean, 'o-', color="g", label='Cross-validation score')
    ax.legend(loc='lower right')
    ax.grid(True)
    fig.savefig('img/learning_curves_'+info, bbox_inches='tight')
    plt.close(fig)


def plot_coefficient_importance(X, y, cv=5, random_state=42, figsize=(7,6), info=""):
    """
    Plot the coefficient importance and its variability on CV repeated folds.
    X is the transformed dataset.
    """
    print("Plot the coefficient importance")
    logregCV = LogisticRegressionCV(
        Cs=np.logspace(-4, 1, 20), random_state=random_state).fit(X, y)
    best_C = logregCV.C_[0]
    cv_models = get_cv_scores(
        LogisticRegression(C=best_C, random_state=random_state),
        X, y, cv=RepeatedKFold(n_splits=cv, n_repeats=cv), fit=True
    )
    # compute coefficients by variability as DataFrame
    coefs = pd.DataFrame(
        [model.coef_[0] for model in cv_models['estimator']],
        columns=X.columns
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.stripplot(data=coefs, orient='h', color='k', alpha=0.5, ax=ax)
    sns.boxplot(data=coefs, orient='h', color='cyan', saturation=0.5, ax=ax)
    ax.axvline(x=0, color='.5')
    ax.set_xlabel('Coefficient importance')
    ax.set_title('Coefficient importance and variability')
    plt.tight_layout()
    fig.savefig('img/coef_importance_'+info, bbox_inches='tight')
    plt.close(fig)


def accuracy(clf, X_train, y_train, X_test, y_test):
    """
    Evaluate classifier accuracy against leaked y_test set.
    """
    clf = clf.fit(X_train, y_train)
    print(f"training score:     {clf.score(X_train, y_train)*100:.4f}")
    print(f"leaked test score:  {clf.score(X_test, y_test)*100:.4f}")


def tune(pipeline, X, y, params, cv=10):
    """
    Run RandomizedSearchCV to select the best hyper-parameters from a list of given grid_search parameters.
    Return the tuned Pipeline and the best hyper-parameters.
    """
    print("Running RandomizedSearchCV")
    rs = RandomizedSearchCV(
        pipeline, params, cv=cv, n_jobs=-1, refit=False
    ).fit(X, y)
    pipeline.set_params(**rs.best_params_)
    return pipeline, rs.best_params_


def build_pipeline(numeric, categorical, random_state=42):
    """
    Return a Pipeline skeleton and connected grid parameters.
    """
    print("Building pipeline")
    # grid search hyperparameters
    n_features = len(numeric) + len(categorical)
    grid_params = [
        # linear grid search
        {
            "transformer__num__imputer__strategy": ["mean", "median"],
            "reducer__n_components": range(2, int(n_features*.8)),
            "reducer__whiten": [True, False],
            "classifier": [LogisticRegression(random_state=random_state)],
            "classifier__C": np.logspace(-4, 1, 21),
        },
        # non-linear grid search
        {
            "transformer__num__imputer__strategy": ["mean", "median"],
            "reducer__n_components": range(2, int(n_features*.8)),
            "reducer__whiten": [True, False],
            "classifier": [SVC(probability=True, random_state=random_state)],
            "classifier__gamma": np.logspace(-4, 1, 20),
            "classifier__C": np.logspace(-3, 1, 21),
        },
        # ensample grid search
        {
            "transformer__num__imputer__strategy": ["mean", "median"],
            "reducer__n_components": range(2, int(n_features*.8)),
            "reducer__whiten": [True, False],
            "classifier": [RandomForestClassifier(random_state=random_state)],
            "classifier__n_estimators": [100, 250, 500],
            "classifier__max_depth": [3, 4, 5, 10, 15, None],
            "classifier__criterion": ["gini", "entropy"]
        },
        {
            "transformer__num__imputer__strategy": ["mean", "median"],
            "reducer__n_components": range(2, int(n_features*.8)),
            "reducer__whiten": [True, False],
            "classifier": [
                GradientBoostingClassifier(random_state=random_state)
            ],
            "classifier__n_estimators": [100, 250, 500],
            "classifier__learning_rate": np.logspace(-3, 1, 21),
            "classifier__subsample": np.logspace(-4, 0, 21)
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
            ("reducer", PCA()),
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
    num_cols, cat_cols = get_column_types(X)
    # build pipeline skeleton and hyperparams
    pipe, params = build_pipeline(num_cols, cat_cols, random_state=random_state)
    # tune pipeline steps
    pipe, best_params = tune(pipe, X, y, params, cv=cv)
    # evaluate pipeline perfs
    scores = get_cv_scores(pipe, X, y, cv=cv)
    return pipe, scores, best_params


def transform_dataset(pipeline, X, y):
    """
    Return transformed X as Pandas DataFrame, i.e. X goes through all transformer steps, `transformer` Pipeline step for X_test transformation, and names of new features.
    TN: Pipeline argument is un unfitted Pipeline created with init_pipeline().
    """
    # fit to data to enable transform() method
    transformer = pipeline["transformer"].fit(X, y)
    # get names of encoded features
    onehot_cols = get_transformed_features(transformer, X)
    # transform given set
    X_tr = pd.DataFrame(transformer.transform(X), columns=onehot_cols)
    return X_tr, transformer, onehot_cols
