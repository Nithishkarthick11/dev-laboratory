# Part 1: NumPy Arrays
import numpy as np

print("---- NumPy Operations ----")
# 1D and 2D Arrays
array_1d = np.array([1, 2, 3, 4, 5])
array_2d = np.array([[1, 2], [3, 4], [5, 6]])

print("1D Array:\n", array_1d)
print("2D Array:\n", array_2d)

# Operations
add_array = array_1d + 5
mult_array = array_1d * 2
sliced_array = array_1d[1:4]
reshaped_array = array_2d.reshape(2, 3)

print("Added 5 to each element:\n", add_array)
print("Multiplied by 2:\n", mult_array)
print("Sliced (1:4):\n", sliced_array)
print("Reshaped 2D to 2x3:\n", reshaped_array)

# Part 2: Pandas DataFrame
import pandas as pd

print("\n---- Pandas DataFrame ----")
# Create DataFrame from dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'Score': [85.5, 90.0, 95.5, 88.0]
}

df = pd.DataFrame(data)

# Display DataFrame
print("DataFrame:\n", df)
print("\nInfo:")
print(df.info())
print("\nStatistics:")
print(df.describe())

# Indexing and slicing
print("\nFirst two rows:\n", df.head(2))
print("Access 'Score' column:\n", df['Score'])

# Basic operation: Increase all scores by 5
df['Score'] = df['Score'] + 5
print("Updated Scores:\n", df)

# Part 3: Matplotlib Basic Plots
import matplotlib.pyplot as plt

print("\n---- Matplotlib Plots ----")
# Sample data
students = ['Alice', 'Bob', 'Charlie', 'David']
scores = df['Score']

# Line Plot
plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.plot(students, scores, marker='o', color='blue')
plt.title('Line Plot')
plt.xlabel('Students')
plt.ylabel('Scores')

# Bar Plot
plt.subplot(1, 3, 2)
plt.bar(students, scores, color='green')
plt.title('Bar Plot')
plt.xlabel('Students')
plt.ylabel('Scores')

# Pie Chart
plt.subplot(1, 3, 3)
plt.pie(scores, labels=students, autopct='%1.1f%%')
plt.title('Pie Chart')

plt.tight_layout()
plt.show()
