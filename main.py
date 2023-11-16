import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

df_boston = fetch_openml(name="boston", as_frame=True)['frame']
print(df_boston.columns)

input_cols = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM',
              'AGE', 'DIS', 'TAX', 'PTRATIO', 'B',
              'LSTAT', 'constant']
target = 'MEDV'
df_boston['constant'] = 1.0

x_train, x_test, y_train, y_test = train_test_split(df_boston[input_cols],
                                                    df_boston[target],
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=10)
print(x_test.shape, x_train.shape)
print(y_test.shape, y_train.shape)

x_dot = np.dot(x_train.T, x_train)   ## multiplication of transposed x and x matrices
x_inv = np.linalg.inv(x_dot)        ### inverse matrix of multiplied matrix x_dot

y_dot = np.dot(x_train.T, y_train)         ### multiplication
weight_parameters = np.dot(x_inv, y_dot) ### regression coefficients

### get fitted value ####
y_pred = np.dot(x_train, weight_parameters)

# plt.plot(y_train.values, label='actual value', color='black')
# plt.plot(y_pred, label='fitted value', color='red')
# plt.legend()
# plt.show()

### get regression error ###
print('Train RMSE:', np.round(np.mean(np.sqrt((y_train.values-y_pred)**2)), 3))

#### prediction on test set ###
y_pred_test = np.dot(x_test, weight_parameters)

print('Test RMSE:', np.round(np.mean(np.sqrt((y_test.values-y_pred_test)**2)), 3))

plt.plot(y_test.values, label='actual value', color='black')
plt.plot(y_pred_test, label='fitted value', color='red')
plt.legend()
plt.show()