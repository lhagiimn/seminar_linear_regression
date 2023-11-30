from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


#Problem 1: Data Acquisition
iris = load_iris()
X = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
y = pd.DataFrame(iris.target, columns=['Species'])

#Problem 2: Combining Data
df = pd.concat([X, y], axis=1)

#Problem 3: Checking the Data
# Display the 4th sample from the beginning
print(df.head(4))

print(df['Species'].value_counts())
print(df.isnull().sum())
print(df.describe())

#Problem 5: Extracting Required Data
sepal_width1 = df['sepal_width']
sepal_width2 = df.loc[:, 'sepal_width']

data_50_99 = df.iloc[50:100]
petal_length_50_99 = df.loc[50:99, 'petal_length']
petal_width_0_2 = df[df['petal_width'] == 0.2]

#Problem 6: Creating a Diagram
# Pie chart
plt.pie(df['Species'].value_counts(), labels=iris.target_names, autopct='%1.1f%%')
plt.title('Number of Samples per Label')
plt.show()

# Box plot
sns.boxplot(x='Species', y='sepal_length', data=df)
plt.title('Distribution of Sepal Length for Each Label')
plt.show()

# Violin plot
sns.violinplot(x='Species', y='sepal_length', data=df)
plt.title('Distribution of Sepal Length for Each Label')
plt.show()

# Explanation of Box plot vs Violin plot:
# Box plot shows summary statistics (median, quartiles), Violin plot shows the entire probability distribution of the data.
# Violin plot is more informative, but Box plot is more compact.

#Problem 7: Confirming the Relationship Between Features
# Scatter plots
sns.scatterplot(x='sepal_length', y='sepal_width', hue='Species', data=df)
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.show()
# Scatterplot matrix
sns.pairplot(df, hue='Species')
plt.suptitle('Scatterplot Matrix', y=1.02)
plt.show()

# Correlation coefficient matrix
correlation_matrix = df.corr()

# Heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Coefficient Matrix')
plt.show()

