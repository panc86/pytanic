# TITANIC PROJECT IN PYTHON: pytanic
import os
import shutil
import pandas as pd
import joblib
from sklearn.metrics import plot_confusion_matrix
# custom libs
import selector
import pipeline

# prepare plots dir
if os.path.exists("img"):
    shutil.rmtree("img")
    os.mkdir("img")
# DATA PREP
# load the training and test data
train = pd.read_csv('./data/train.csv', index_col='PassengerId')
test = pd.read_csv('./data/test.csv', index_col='PassengerId')
# leaked target
y_test = pd.read_csv("predictions/y_leaked.csv", index_col="PassengerId")
y_test = y_test.values.flatten()
# missing values count
print("Missing values count")
print(selector.show_missing_values(train, test))
# ensure categorical types
cat_cols = ["Name", "Sex", "Pclass", "Ticket", "Cabin", "Embarked"]
for temp in [train, test]:
    temp[cat_cols] = temp[cat_cols].astype("category")
# describe features
train.describe(include="all")
# multi-variate target analysis
g = train.hist(figsize=(10,10))
g.figure_.savefig("img/boxplots", bbox_inches='tight')
# marginal distributions and pairwise relationships between the features
# it is marginal because it describes the behavior of a specific variable without keeping the others fixed.
g = sns.pairplot(df, hue="Survived", kind='reg', diag_kind='kde')
g.savefig('img/marginal_distributions', bbox_inches='tight')
# PROTOTYPING
# training data
y = train["Survived"].astype(bool)
X = train.drop("Survived", axis=1)
# class distribution
y.value_counts(normalize=True)
# make pipeline
tpl, params = pipeline.titanic_pipeline()
# eval full pipeline (baseline)
selector.get_cv_scores(tpl, X, y).agg(["mean", "std"]).T
# tuning
pipeline.tuning(tpl, params, X, y, verbose=1)
# EVALUATION
# confusion matrix
_ = tpl.fit(X, y)
g = plot_confusion_matrix(tpl, test, y_test)
#g.figure_.savefig("img/confusion_matrix", bbox_inches='tight')
# check quality of predictions
y_hat = tpl.predict(test)
# get type I errors (False Positive)
type_I_mask = (y_test == 0) & (y_hat == 1)
type_I_err = test.assign(Survived=y_test)[type_I_mask]
print("\n5 type I errors")
print(type_I_err.head())
# get type II errors (False Negative)
type_II_mask = (y_test == 1) & (y_hat == 0)
type_II_err = test.assign(Survived=y_test)[type_II_mask]
print("\n5 type II errors")
print(type_II_err.head())
# Sex influence is very high and it fool the model on predictions

# save model
joblib.dump(pipeline, 'titanic_pipeline.joblib')
