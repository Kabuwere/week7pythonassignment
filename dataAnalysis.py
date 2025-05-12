import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the target column
df['species'] = iris.target

# Display the first few rows
print(df.head())

# Check data types and missing values
print(df.info())
print(df.isnull().sum())

# Clean the dataset (if needed)
df.dropna(inplace=True)

# Compute basic statistics
print(df.describe())

# Group by species and compute the mean of numerical columns
grouped_data = df.groupby('species').mean()
print(grouped_data)

# Identify patterns
print("Observation: Different species have distinct average petal and sepal measurements.")

import matplotlib.pyplot as plt
import seaborn as sns

# Line chart (example: cumulative sum of sepal length)
df['cumulative_sepal_length'] = df['sepal length (cm)'].cumsum()
plt.figure(figsize=(8, 5))
plt.plot(df.index, df['cumulative_sepal_length'], label='Cumulative Sepal Length')
plt.title("Cumulative Sepal Length Over Samples")
plt.xlabel("Sample Index")
plt.ylabel("Cumulative Sepal Length")
plt.legend()
plt.show()

# Bar chart (average petal length per species)
plt.figure(figsize=(8, 5))
sns.barplot(x=df['species'], y=df['petal length (cm)'])
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# Histogram (distribution of sepal width)
plt.figure(figsize=(8, 5))
sns.histplot(df['sepal width (cm)'], bins=20, kde=True)
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter plot (sepal length vs. petal length)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['sepal length (cm)'], y=df['petal length (cm)'], hue=df['species'])
plt.title("Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

try:
    df = pd.read_csv("your_dataset.csv")
except FileNotFoundError:
    print("Error: The file was not found. Please check the file path.")
except pd.errors.EmptyDataError:
    print("Error: The file is empty.")
except pd.errors.ParserError:
    print("Error: There was an issue parsing the file.")

