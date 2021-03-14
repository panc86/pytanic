# COMMONS
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from collections import namedtuple
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, learning_curve
import rfpimp


def show_missing_values(train, test):
    """
    Return concatenated count of missing values for train and test sets.
    """
    na = pd.concat([train.isna().sum(), test.isna().sum()], axis=1)
    na.columns = ['train', 'test']
    print(na)


def column_indices(df, cols):
    # get column index
    return [df.columns.get_loc(c) for c in cols]


def get_encoded_features(transformer, categorical_features):
    encoder = transformer.named_transformers_["cat"].named_steps["onehot"]
    return encoder.get_feature_names(categorical_features)


def plot_coefficients(pipeline, X):
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(18, 7))
    columns = X.columns.values
    pd.DataFrame(
        pipeline["classifier"].coef_[0],
        columns=['Coefficients'],
        index=columns
    ).plot(kind='barh', ax=ax1)
    ax1.axvline(x=0, color='.5')
    # with standard deviation normalization
    pd.DataFrame(
        X.std(axis=0),
        columns=["Stddev"],
        index=columns
    ).plot(kind='barh', ax=ax2)
    # how is the normalized importance of features (by stddev)?
    coefs_norm = pd.DataFrame(
        pipeline["classifier"].coef_[0] * X.std(axis=0),
        columns=['Normalized, Coefficients'], index=columns
    )
    _ = coefs_norm.plot(kind='barh', ax=ax3)
    ax3.axvline(x=0, color='.5')
    fig.savefig('img/importance_vs_stdev_p2_model', bbox_inches='tight')
    return coefs_norm


def plot_learning_curves(model, X, y, cv=3, scoring=None, figsize=(10,5), info=""):
    """
    How the model learns on a growing number of training examples?
    Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """
    # compute scores/training_sizes
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes, n_jobs=-1
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # setup plot
    fig, ax = plt.subplots(figsize=figsize)
    final_gap = train_scores_mean[-1] - test_scores_mean[-1]
    ax.set_title(f"Learning Curves - score_gap={final_gap:.4f}")
    ax.set_xlabel('No. Training Examples')
    ax.set_ylabel('Scores')
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.2,
                        color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.2,
                        color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label='Training score')
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label='Cross-validation score')
    ax.legend(loc='best')
    ax.grid(True)
    fig.savefig('img/learning_curves_'+info, bbox_inches='tight')

