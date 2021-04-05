import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


# FEATURES ENGINEERING
# CABININD
def cabin_indicator(df):
    # whether a passenger has Cabin info or not
    return df["Cabin"].notna()


# NAMESIZE
def passenger_name_size(df):
    # passenger's title
    return df["Name"].str.len()


# TITLE
def passenger_title(df):
    # title mapping
    title_map = {'ms': 'mrs', 'mme': 'mrs', 'mlle': 'miss'}
    title_map.update({title: 'rare' for title in ['dr', 'rev', 'sir', 'don', 'jonkheer', 'lady', 'the countess', 'col', 'major', 'capt', 'dona']})
    # the Title of the passenger indicates gender, age, and social class
    def parse_title(name):
        # title is between last and first names
        title = name.split(',')[1].lstrip().split('.')[0].lower()
        return title_map.get(title, title)
    # passenger's title
    return df["Name"].apply(parse_title).astype("category")


# BINNING
# Linear models benefit from binning continuous features, i.e. Age and Fare
def to_bin(df, feature, bins):
    labels = [f"{feature}_b{q}" for q in range(bins)]
    binned = pd.qcut(df[feature], q=bins, labels=labels).pipe(pd.get_dummies, dtype=float)
    return pd.concat([df, binned], axis=1)


def with_interactions(X):
    # add features interactions, i.e. element-wise product
    interactions = PolynomialFeatures(interaction_only=True).fit(X)
    int_cols = interactions.get_feature_names(X.columns)
    X_int = pd.DataFrame(interactions.transform(X), columns=int_cols).drop("1", axis=1)
    return X_int, interactions
