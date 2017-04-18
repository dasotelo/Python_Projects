#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 08:27:34 2017

@author: sotelo

Below script uses NESARC data to examine interaction between two variables:
    1) Current (last 12 mos) presence of Major Depression
    2) Marital Status
https://pubs.niaaa.nih.gov/publications/arh29-2/74-78.htm
"""

import pandas
import numpy
import scipy.stats
import matplotlib.pyplot as plt

#Read NESARC data file into DataFrame
proj=pandas.read_csv('nesarc_pds.csv',low_memory=False)

#Convert variables of interest to numeric
proj['MARITAL']=proj['MARITAL'].convert_objects(convert_numeric=True)
proj['S4AQ8BR']=proj['S4AQ8BR'].convert_objects(convert_numeric=True)
proj['S4AQ8BR']=proj['S4AQ8BR'].fillna(0)

#Map marital status values to discernable string categories
#Eliminate unknown values of major depression in last 12 mos variable
map1={
    1:'Married',
    2:'Co-habitate',
    3:'Widowed',
    4:'Divorced',
    5:'Separated',
    6:'Never Married'}
proj['MARITL_STS']=proj['MARITAL'].map(map1)
map2={
    0:'No',    
    1:'Yes',
    2:'No',
    9:'NA'}
proj['DEP_STS_12MO']=proj['S4AQ8BR'].map(map2)

#Subset data to variables of interest only, eliminate unknown and na values
proj_ed=proj.loc[(proj['DEP_STS_12MO']!='NA'),['MARITL_STS','DEP_STS_12MO']]

#Crosstab of counts
ct1=pandas.crosstab(proj_ed['DEP_STS_12MO'],proj_ed['MARITL_STS'])
print('Crosstab: Counts')
print(ct1)

#Crosstab of percentages
colsum=ct1.sum(axis=0)
pct1=ct1/colsum
print('Crosstab: Percentages by Column')
print(pct1)

#Plot rates of major depression using horizontal bar plot
plt.barh(range(len(pct1.columns)),numpy.array(pct1.iloc[1,:]),
    tick_label=pct1.columns.values,edgecolor='black',linewidth=0.6)
plt.title('Rates of Major Depression by Marital Status',x=0.35,size=18)
plt.xlabel('Major Depression Rate',size=14)
plt.ylabel('Marital Status',size=14)
plt.xticks(size=12)
plt.yticks(size=12)
plt.grid(False)

#Chi-Square statistic
proj_chisq=scipy.stats.chi2_contingency(ct1)
print('Chi Squared Statistic Summary for Depression Status on Marital Status')
print(proj_chisq)

for i in range(0,6):
    for j in range (0,6):
        if i<j:           
            #Crosstab of counts, subsetting data
            rct1=ct1.iloc[:,[i,j]]
            print('@@@@@@@@@@@@@@@@@@@@@@')
            print(list(rct1.columns.values))
            print('@@@@@@@@@@@@@@@@@@@@@@')
            print('Crosstab: Counts')
            print(rct1)
            
            #Crosstab of percentages
            rcolsum=rct1.sum(axis=0)
            rpct1=rct1/rcolsum
            print('Crosstab: Percentages by Column')
            print(rpct1)
            
            #Chi-Square statistic
            rproj_chisq=scipy.stats.chi2_contingency(rct1)
            print('Chi Squared Statistic Summary for Depression Status on',
                  ' Marital Status')
            print(rproj_chisq)
        
