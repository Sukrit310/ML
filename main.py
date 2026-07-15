from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso



load_diabetes = load_diabetes()

x = load_diabetes.data
y = load_diabetes.target

print(load_diabetes.feature_names)
print(x)
print(y)

random_state = 0
ts = 0.99
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = ts )

model = Lasso(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(mse)

