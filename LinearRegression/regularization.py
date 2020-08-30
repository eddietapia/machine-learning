import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

'''
Regularization is a technique to make sure that the models will not only fit 
to the data but also extend to new situations
'''

# Assign the data to predictor and outcome variables
train_data = pd.read_csv("data2.csv", header=None)
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

# Create the linear regression model with lasso regularization.
lasso_reg = Lasso()
model = LinearRegression()

# Fit the model.
lasso_reg.fit(X, y)
model.fit(X, y)

# Retrieve and print out the coefficients from the regression model.
print("Coefficients with regularization")
reg_coef = lasso_reg.coef_
print(reg_coef)

print("\n\nWithout regularization")
coef = model.coef_
print(coef)
