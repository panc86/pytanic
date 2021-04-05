# TITANIC PROJECT IN PYTHON: pytanic
import os
import shutil
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
# custom libs
import selector
import pipeline

# clean output dir
if os.path.exists("artifacts"):
    shutil.rmtree("artifacts")
    os.mkdir("artifacts")
# DATA PREP
# load the training and test data
train = pd.read_csv("data/train.csv", index_col='PassengerId')
test = pd.read_csv("data/test.csv", index_col='PassengerId')
# leaked target
y_test = pd.read_csv("data/y_leaked.csv", index_col="PassengerId")
y_test = y_test.values.flatten()
# ensure categorical types
cat_cols = ["Name", "Sex", "Pclass", "Ticket", "Cabin", "Embarked"]
for temp in [train, test]:
    temp[cat_cols] = temp[cat_cols].astype("category")
# missing values count
print("Missing values count")
print(selector.show_missing_values(train, test))
# marginal distributions and pairwise relationships between the features
# it is marginal because it describes the behavior of a specific variable without keeping the others fixed.
print("Plot marginal distributions")
g = sns.pairplot(train, hue="Survived", kind='reg', diag_kind='kde')
g.savefig('artifacts/marginal_dists', bbox_inches='tight')
# decribe features
print("Describe training")
train.describe(include="all")
# multi-variate target analysis
print("Plot boxplots")
fig, ax = plt.subplots(figsize=(10,10))
train.hist(ax=ax)
fig.savefig("artifacts/boxplots.png", bbox_inches='tight')
# PROTOTYPING
print("Start prototyping")
# training data
y = train["Survived"].astype(bool)
X = train.drop("Survived", axis=1)
# class distribution
y.value_counts(normalize=True)
# make pipeline
tpl, params = pipeline.titanic_pipeline()
# eval full pipeline (baseline)
print("CV score")
selector.get_cv_scores(tpl, X, y).agg(["mean", "std"]).T
# tuning
pipeline.tuning(tpl, params, X, y, verbose=1)
# EVALUATION
# confusion matrix
_ = tpl.fit(X, y)
print("Test score:", tpl.score(test, y_test))
print("Plot confusion matrix")
g = plot_confusion_matrix(tpl, test, y_test)
g.figure_.savefig("artifacts/confusion_matrix", bbox_inches='tight')
# check quality of predictions
y_hat = tpl.predict(test)
# get type I errors (False Positive)
type_I_mask = (y_test == 0) & (y_hat == 1)
type_I_err = test.assign(Survived=y_test)[type_I_mask]
type_I_err.to_csv("artifacts/type_I_errors.csv")
# get type II errors (False Negative)
type_II_mask = (y_test == 1) & (y_hat == 0)
type_II_err = test.assign(Survived=y_test)[type_II_mask]
type_II_err.to_csv("artifacts/type_II_errors.csv")
# Sex influence is very high and it fool the model on predictions
# save model
joblib.dump(tpl, "artifacts/titanic_pipeline.joblib")
