import gc
import joblib
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
warnings.filterwarnings('ignore')

iris = load_iris()

X = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length',
                                     'petal_width'])
y = pd.DataFrame(iris.target_names, columns=['Species'])

df = pd.concat([X, y], axis=1)

#[Assignment 1] Select features and categories for practice
# Scatter plots
sns.scatterplot(x='sepal_length', y='sepal_width', hue='Species', data=df)
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.show()

sns.scatterplot(x='pepal_length', y='pepal_width', hue='Species', data=df)
plt.title('Scatter Plot of Pepal Length vs Pepal Width')
plt.show()

#[Question 2] Data analysis


train_X, test_X, train_Y, test_y = train_test_split(X, y, test_size=0.3,
                                                     shuffle=True, random_state=42)


