# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
df = pd.read_csv('Module2/Datasets/direct_marketing.csv')

print df.recency
print df['recency']
print df[['recency']]
print df.loc[:, 'recency']
print df.loc[:, ['recency']]
printdf.iloc[:, 0]
df.iloc[:, [0]]
df.ix[:, 0]