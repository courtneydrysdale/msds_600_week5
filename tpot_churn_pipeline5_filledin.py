import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('prepped_churn_data.csv', index_col='customerID')
features = tpot_data.drop('Churn', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Churn'], random_state=42)

# Average CV score on the training set was: 0.7993924296518791
exported_pipeline = make_pipeline(
    MinMaxScaler(),
    StackingEstimator(estimator=MultinomialNB(alpha=1.0, fit_prior=True)),
    XGBClassifier(learning_rate=0.1, max_depth=1, min_child_weight=13, n_estimators=100, n_jobs=1, subsample=0.3, verbosity=0)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
print(results)
