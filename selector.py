import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# MODELS
from sklearn.model_selection import (
    cross_validate, RepeatedKFold, learning_curve
)
import rfpimp


def show_missing_values(train, test):
    """
    Return concatenated count of missing values for train and test sets.
    """
    na_dist = pd.concat([train.isna().sum(), test.isna().sum()], axis=1)
    na_dist.columns = ['train', 'test']
    return na_dist


def get_cv_scores(model, X, y, cv=None, scoring=["accuracy", "roc_auc"], train_score=True, fit=False):
    """
    Return CV Accuracy and ROC scores as Pandas DataFrame for a given model.
     - `fit` allows to return the fitted model for each CV split
     - `train_score` return the train score together with the CV score.
    """
    return pd.DataFrame(cross_validate(
        model, X, y, n_jobs=-1, scoring=scoring,
        cv=RepeatedKFold(n_splits=10, n_repeats=10) if cv is None else cv,
        return_train_score=train_score, return_estimator=fit
    )).drop(["fit_time", "score_time"], axis=1)


def features_correlation(X, method="pearson"):
    # features correlation (with normal-like dist)
    corr = X.corr(method=method)
    return corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))


def plot_features_correlation(X, method="pearson", figsize=(8,7), info=""):
    """
    Plot Parwise Correlation Heatmap.
    """
    plt.close("all")
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("Plot parwise correlation heatmap")
    sns.heatmap(
        features_correlation(X, method=method), annot=True, fmt='.2f',
        vmin=-1, vmax=1, cmap='RdBu_r', ax=ax
    )
    fig.savefig('img/correlation_'+info, bbox_inches='tight')
    return fig


def plot_features_collinearity(X, figsize=(8,7), info=""):
    """
    The scikit-learn Random Forest feature importances strategy is mean decrease in impurity (or gini importance) mechanism, which is unreliable. To get reliable results, use permutation importance, provided in the rfpimp package. Given training observation independent variables in a dataframe, compute the feature importance using each var as a dependent variable using a RandomForestRegressor or RandomForestClassifier. We retrain a random forest for each var as target using the others as independent vars. Only numeric columns are considered.
    # The dependence heatmap can be read as follows:
    # The features on the X axis predict the features on the Y axis, the higher the score, the higher the correlation.
    For more info, see https://github.com/parrt/random-forest-importances
    """
    plt.close("all")
    fig = plt.figure(figsize=figsize)
    plt.title("Plot features collinearity (rfpimp)")
    rfpimp.plot_dependence_heatmap(rfpimp.feature_dependence_matrix(X))
    fig.savefig('img/collinearity_'+info, bbox_inches='tight')
    return fig


def plot_learning_curves(model, X, y, cv=5, figsize=(10,5), info=""):
    """
    How the model learns on a growing number of training examples?
    Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """
    plt.close("all")
    intervals, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=np.linspace(.1, 1., 5), n_jobs=-1
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # setup plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("Learning Curves " + info)
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
    return fig


def plot_coefficient_importance(model, X, y, figsize=(8,7), info=""):
    """
    Plot the coefficient importance and its variability on CV repeated folds.
    X is the transformed dataset.
    """
    plt.close("all")
    cv_models = get_cv_scores(model, X, y, fit=True)
    # compute coefficients by variability as DataFrame
    coefs = pd.DataFrame(
        [logreg.coef_[0] * X.std(axis=0) for logreg in cv_models['estimator']],
        columns=X.columns
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.stripplot(data=coefs, orient='h', color='k', alpha=0.5, ax=ax)
    sns.boxplot(data=coefs, orient='h', color='cyan', saturation=0.5, ax=ax)
    ax.axvline(x=0, color='.5')
    ax.set_xlabel('Coefficient importance')
    ax.set_title('Coefficient importance and variability')
    fig.savefig('img/coef_importance_'+info, bbox_inches='tight')
    return fig


def plot_feature_target_dist(df, y, feature_names, info=""):
    """
    Features-target explaination (continuous).
    """
    n_features = len(feature_names)
    fig, axes = plt.subplots(n_features, 1, figsize=(6,n_features*3))
    died = df[y == 0]
    survived = df[y == 1]
    ax = axes.ravel()
    for i, f in enumerate(feature_names):
        _, bins = np.histogram(df.loc[:, f], bins=50)
        ax[i].hist(died.loc[:, f], bins=bins, color="r", alpha=.5)
        ax[i].hist(survived.loc[:, f], bins=bins, color="g", alpha=.5)
        ax[i].set_title(feature_names[i])
        ax[i].set_yticks(())
        ax[0].set_xlabel("Feature magnitude")
        ax[0].set_ylabel("Frequency")
        ax[0].legend(["died", "survived"], loc="best")
        fig.savefig("img/per_class_feature_dist_"+info, bbox_inches="tight")
