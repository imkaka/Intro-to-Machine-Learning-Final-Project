#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.metrics import classification_report

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'total_stock_value', 'exercised_stock_options', 'bonus']   # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E", 0)

### convert NaN values for financial data to 0

payment_features = ['total_payments', 'salary', 'bonus', 'long_term_incentive', 'deferred_income',
                   'deferral_payments', 'loan_advances', 'other', 'expenses',
                   'director_fees']
stock_features = ['total_stock_value','exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred']

for name in data_dict:
    if data_dict[name]['total_payments'] >0 or data_dict[name]['total_stock_value'] >0:
        for feature in payment_features:
            if data_dict[name][feature] == 'NaN':
                data_dict[name][feature] = 0
        for feature in stock_features:
            if data_dict[name][feature] == 'NaN':
                data_dict[name][feature] = 0

### correct transposed values
data_dict['BELFER ROBERT']['deferred_income'] = -102500
data_dict['BELFER ROBERT']['deferral_payments'] = 0
data_dict['BELFER ROBERT']['expenses'] = 3285
data_dict['BELFER ROBERT']['director_fees'] = 102500
data_dict['BELFER ROBERT']['total_payments'] = 3285

data_dict['BELFER ROBERT']['exercised_stock_options'] = 0
data_dict['BELFER ROBERT']['restricted_stock'] = 44093
data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -44093
data_dict['BELFER ROBERT']['total_stock_value'] = 0



data_dict['BHATNAGAR SANJAY']['other'] = 0
data_dict['BHATNAGAR SANJAY']['expenses'] = 137864
data_dict['BHATNAGAR SANJAY']['director_fees'] = 0
data_dict['BHATNAGAR SANJAY']['total_payments'] = 137864

data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 15456290


### Task 3: Create new feature(s)
for name in data_dict:
    stock = float(data_dict[name]['total_stock_value'])
    total_value = float(data_dict[name]['total_payments'] + stock)
    if total_value == 0:
        data_dict[name]['new_stock_proportion_feature'] = 0
    else:
        data_dict[name]['new_stock_proportion_feature'] = stock/total_value

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

parameters = {'criterion':('gini', 'entropy')
                }
DT = tree.DecisionTreeClassifier(random_state = 10)
clf = GridSearchCV(DT, parameters, scoring = 'f1')
clf= clf.fit(features_train, labels_train)
clf = clf.best_estimator_

estimators = [('scaler', MinMaxScaler()),
            ('reduce_dim', PCA()),
            ('clf', clf)]



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

clf = Pipeline(estimators)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = clf.score(features_test, labels_test)
print "Accuracy: ", accuracy
target_names = ['non_poi', 'poi']
print classification_report(y_true = labels_test, y_pred =pred, target_names = target_names)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
