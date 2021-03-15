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
    ax.set_title("Learning Curves " + model["classifier"])
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
    ax.legend(loc='lower right')
    ax.grid(True)
    fig.savefig('img/learning_curves_'+info, bbox_inches='tight')
    plt.close('all')


def plot_features_correlation(X, figsize=(6,5), info=""):
    """
    Plot Parwise correlation heatmap.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        X.corr(), annot=True, fmt='.2f', ax=ax, vmin=-1, vmax=1, cmap='RdBu_r'
    )
    fig.savefig('img/correlation_'+info, bbox_inches='tight')
    plt.close('all')


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
    plt.close('all')


def plot_coefficients(pipeline, X, figsize=(18, 7), info=""):
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=figsize)
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
    fig.savefig('img/coef_by_stdev_'+info, bbox_inches='tight')
    plt.close('all')
    return coefs_norm


def plot_features_importance(pipeline, X, figsize=(7,6), info=""):
    fig, ax = plt.subplots(figsize=figsize)
    I = pd.DataFrame()
    I['Features'] = X.columns.values
    I['Importance'] = pipeline["classifier"].coef_[0]
    I = I.sort_values('Importance', ascending=False).set_index('Features')
    rfpimp.plot_importances(
        I, width=5, color='#FDDB7D', ax=ax,
        title="Feature importance via average gini/variance drop (sklearn)"
    )
    fig.savefig('img/rfpimp_importance_'+info, bbox_inches='tight')
    plt.close('all')
    return I


# def hypotesis_testing():
#     fig, ax = plt.subplots(figsize=(8, 4))
#     ids = hypotheses_df['id']
#     ax.barh(ids, hypotheses_df[scoring], xerr=hypotheses_df['error'], align='center', alpha=0.5, ecolor='black', capsize=10)
#     ax.set_xlabel(scoring.title())
#     ax.set_xlim(.7, .95)
#     ax.set_yticks(ids)
#     ax.set_yticklabels(ids)
#     ax.set_title('Cross Validation Report')
#     ax.xaxis.grid(True)
#     plt.tight_layout()