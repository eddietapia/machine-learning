# Import grid search and metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.svm import SVC


# Get the data
data = pd.read_csv('data2.csv')
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

# Create the classifier
clf = SVC()

# Select the parameters
parameters = {'kernel':['poly', 'rbf'],'C':[0.1, 1, 10]}

# Create a scorer
scorer = make_scorer(f1_score)

# Create the object.
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# Fit the data
grid_fit = grid_obj.fit(X, y)

# Get the best estimator
best_clf = grid_fit.best_estimator_
print(best_clf)
