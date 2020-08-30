import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from utils import randomize, draw_learning_curves


data = pd.read_csv('data2.csv')

X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

X2, y2 = randomize(X, y)

# Fix random seed
np.random.seed(55)

# Uncomment one of the three classifiers, and run learning_curves.py

### Logistic Regression
#estimator = LogisticRegression()

### Decision Tree
estimator = GradientBoostingClassifier()

### Support Vector Machine
#estimator = SVC(kernel='rbf', gamma=1000)

# Draw the learning curves
draw_learning_curves(X2, y2, estimator, num_trainings=400)

# Results after comparing the curves:
# LR underfits, DT is just right, and SVM overfits
