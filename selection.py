# COMMONS
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# default model for features importance computation
from sklearn.svm import LinearSVC
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


def plot_learning_curves(clf, X, y, cv=10, figsize=(10,5), info=""):
    """
    How the model learns on a growing number of training examples?
    Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """
    intervals, train_scores, test_scores = learning_curve(
        clf, X, y, cv=cv, train_sizes=np.linspace(.1, 1., 5), n_jobs=-1
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # setup plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f"Learning Curves {clf}")
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


def plot_features_correlation(X, figsize=(6,5), info=""):
    """
    Plot Parwise correlation heatmap.
    """
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
    fig = plt.figure(figsize=figsize)
    rfpimp.plot_dependence_heatmap(rfpimp.feature_dependence_matrix(X))
    fig.savefig('img/rfpimp_dependence_'+info, bbox_inches='tight')
    plt.close(fig)


def plot_coefficient_importance(X, y, random_state=42, figsize=(7,6), info=""):
    """
    Plot the coefficient importance and its variability on CV repeated folds.
    X is the transformed dataset.
    Tn: By default, the classifier to compute the coefficients is LinearSVC(),
    using the following hyper-parameters:
      - C=1.0
      - penalty="l1"
      - dual=False
    """
    clf = LinearSVC(C=1., penalty="l1", dual=False, random_state=random_state)
    cv_model = cross_validate(
        clf, X, y, cv=RepeatedKFold(n_splits=5, n_repeats=5),
        return_estimator=True, n_jobs=-1
    )
    # variability
    X_std = X.std(axis=0)
    # compute coefficients by variability as DataFrame
    coefs = pd.DataFrame(
        [clf.coef_[0] * X_std for clf in cv_model['estimator']],
        columns=X.columns
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.stripplot(data=coefs, orient='h', color='k', alpha=0.5, ax=ax)
    sns.boxplot(data=coefs, orient='h', color='cyan', saturation=0.5, ax=ax)
    ax.axvline(x=0, color='.5')
    ax.set_xlabel('Coefficient importance')
    ax.set_title('Coefficient importance and (std) variability')
    plt.tight_layout()
    fig.savefig('img/coef_importance_'+info, bbox_inches='tight')
    plt.close(fig)


def get_accuracy(clf, X_train, y_train, X_test):
    """
    Evaluate classifer accuracy against leaked y_test set.
    """
    y_true = pd.read_csv("predictions/y_leaked.csv", index_col="PassengerId")
    clf = clf.fit(X_train, y_train)
    print("Accuracy score on training:", clf.score(X_train, y_train))
    print("Accuracy score on test:", clf.score(X_test, y_true))
    return y_true
