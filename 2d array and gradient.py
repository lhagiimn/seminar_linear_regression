import numpy as np
import matplotlib.pyplot as plt

x_values_1 = np.arange(-50, 50.1, 0.1)

x_values_3 = np.arange(0, 50.1, 0.1)

# Define the functions
def function_1(x):
    return x**2
def function_2(x):
    return 2 * x**2 + 2**x
def function_3(x):
    return np.sin(np.sqrt(x))

gradient_1 = np.gradient(function_1(x_values_1), x_values_1)
gradient_2 = np.gradient(function_2(x_values_1), x_values_1)
gradient_3 = np.gradient(function_3(x_values_3), x_values_3)

plt.plot(x_values_1, gradient_1, label='Gradient of y = x^2')
plt.plot(x_values_1, gradient_2, label='Gradient of y = 2x^2 + 2^x')
plt.plot(x_values_3[:-1], gradient_3[:-1], label='Gradient of y = sin(sqrt(x))')

plt.xlabel('x')
plt.ylabel('Gradient')
plt.legend()
plt.grid(True)
plt.show()
