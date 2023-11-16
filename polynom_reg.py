import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

df_boston = fetch_openml(name="boston", as_frame=True)['frame']
print(df_boston.columns)

input_cols = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM',
              'AGE', 'DIS', 'TAX', 'PTRATIO', 'B',
              'LSTAT']
target = 'MEDV'
df_boston['constant'] = 1.0

x_train, x_test, y_train, y_test = train_test_split(df_boston[input_cols],
                                                    df_boston[target],
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=10)
print(x_test.shape, x_train.shape)
print(y_test.shape, y_train.shape)

def get_parameters(x, y):
    x_dot = np.dot(x.T, x)
    x_inv = np.linalg.inv(x_dot)

    y_dot = np.dot(x.T, y)
    weight_parameters = np.dot(x_inv, y_dot)

    return weight_parameters

degree = [1, 2, 3]

plt.plot(y_test.values, label='actual value', color='black')
for deg in degree:
    poly = PolynomialFeatures(degree=deg)
    phi = poly.fit(x_train)
    x = phi.transform(x_train)
    x_ = phi.transform(x_test)

    w = get_parameters(x, y_train)

    ### get fitted value ####
    y_pred = np.dot(x, w)

    ### get regression error ###
    print(f'Degree {deg} Train RMSE:', np.round(np.mean(np.sqrt((y_train.values-y_pred)**2)), 3))

    #### prediction on test set ###
    y_pred_test = np.dot(x_, w)

    print(f'Degree {deg} Test RMSE:', np.round(np.mean(np.sqrt((y_test.values-y_pred_test)**2)), 3))


    plt.plot(y_pred_test, label=f'fitted value_{deg}', color='red')
plt.legend()
plt.show()