from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y.
X = data[:,0:2]
y = data[:,2]

# Create the adaboost model
model = AdaBoostClassifier(base_estimator= DecisionTreeClassifier(max_depth=2), n_estimators= 4)

# Fit the model
model.fit(X, y)

# Make the predictions
y_pred = model.predict(X)

# Calculate the accuracy of your model with the training set
acc = accuracy_score(y, y_pred)
print(acc)
