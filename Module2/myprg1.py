# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:14:36 2017

@author: m037382
"""


import pandas as pd
df = pd.read_html('http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2')
df1 = pd.concat(df)
df1.columns = ['RK', 'PLAYER', 'TEAM', 'GP','G','A','PTS','+/-',' PIM','PTS/G','SOG','PCT','GWG','PP-G','PP-A','SH-G','SH-A']
df1= df1[df1.RK != "RK"]
df1= df1.drop('RK',axis=1)
df1=df1.dropna()
df1=df1.reset_index(drop=True)
print df1
print len(df1.PCT.unique())
print add( df1.loc[15, 'GP'] , df1.loc[16, 'GP'])????