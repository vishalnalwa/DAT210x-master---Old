# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 20:38:32 2017

@author: m037382
"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


matplotlib.style.use('ggplot') # Look Pretty
# If the above line throws an error, use plt.style.use('ggplot') instead

student_dataset = pd.read_csv("datasets/students.data", index_col=0)

my_series = student_dataset.G3
my_dataframe = student_dataset[['G3', 'G2', 'G1']] 

# Histogram
my_series.plot.hist(alpha=0.5)
my_dataframe.plot.hist(alpha=0.5)

#Scatter plot - 2D
student_dataset.plot.scatter(x='G1', y='G3')

#Scatter plot - 3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Final Grade')
ax.set_ylabel('First Grade')
ax.set_zlabel('Daily Alcohol')

ax.scatter(student_dataset.G1, student_dataset.G3, student_dataset['Dalc'], c='r', marker='.')
plt.show()