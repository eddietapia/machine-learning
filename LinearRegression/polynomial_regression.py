import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

'''
Note: Polynomial regression is used for relationships between variables 
that aren't linear
'''

# Assign the data to predictor and outcome variables
train_data = pd.read_csv('data.csv')
X = train_data['Var_X'].values.reshape(-1, 1)
y = train_data['Var_Y']

# Create polynomial features and predictor feature
poly_feat = PolynomialFeatures(degree=4)
X_poly = poly_feat.fit_transform(X)

# Make and fit the polynomial regression model
poly_model = LinearRegression().fit(X_poly, y)
