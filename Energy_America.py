###################################
#   Exploring the relationship between energy consumption & 
#   energy price
#
#   Author: Dave Sotelo
#   Date: May 1, 2017
#
#   Data from various public sources, via Kaggle - see below:
#   https://www.kaggle.com/lislejoem/us_energy_census_gdp_10-14
###################################

import pandas as pd
import scipy.stats as stat
import matplotlib.pyplot as plt
import seaborn as sbn

#Read Energy Census & Economic Data file
#Source: Various {Bureau of Economic Analysis, EIA} via Kaggle.com
#https://www.kaggle.com/lislejoem/us_energy_census_gdp_10-14
nrg_econ=pd.read_csv('Energy Census and Economic Data US 2010-2014.csv',
    low_memory=False)

#Subset data: Energy price & production data, CY2014
nrg_cons=nrg_econ.loc[nrg_econ['State']!='United States',['StateCodes'
    ,'State','Region','Division','POPESTIMATE2014','TotalC2014'
    ,'TotalPrice2014']]

#Calculated variable: Energy consumption per capita, CY2014
nrg_cons['TOTC2014']=nrg_cons['TotalC2014']/ \
    nrg_cons['POPESTIMATE2014']
nrg_cons=nrg_cons.sort_values(['TOTC2014'],ascending=False)

#Barplot of Energy consumption per capita, by State
sbn.set_style('whitegrid')
plt.figure(figsize=(6,10))
plt.axes([0.27,0.1,0.68,0.83])
sbn.barplot(nrg_cons['TOTC2014'],nrg_cons['State'],orient='h',
    palette='RdBu')
plt.xlabel('Energy Consumption in Billion BTUs per Capita (2014)')
plt.title('Energy Consumption per Capita, by State',size=14,x=0.27)
plt.show()

nrg_cons=nrg_cons.sort_values(['TotalPrice2014'],ascending=False)

#Barplot of Energy Price per Million BTU, by State
sbn.set_style('whitegrid')
plt.figure(figsize=(6,10))
plt.axes([0.27,0.1,0.68,0.83])
sbn.barplot(nrg_cons['TotalPrice2014'],nrg_cons['State'],orient='h',
    palette='PRGn_r')
plt.xlabel('Energy Price USD per Million BTUs (2014)')
plt.title('Energy Price USD per Million BTUs, by State',size=14,x=0.27)
plt.show()

#Scatter plot of energy consumption on price per BTU(M)
sbn.set_style('whitegrid')
sbn.regplot(nrg_cons['TotalPrice2014'],nrg_cons['TOTC2014'],ci=67,
    color='green',scatter_kws={'alpha':0.7,'edgecolors':'k'})
plt.xlabel('Energy Price USD per Million BTUs')
plt.ylabel('Energy Consumption in Billion BTUs per Capita')
plt.title('Linear Regression of Energy Consumption on Energy Price')
plt.show()

#Describe linear fit using Pearson correlation coefficient
ols_coeff=stat.pearsonr(nrg_cons['TotalPrice2014'],
    nrg_cons['TOTC2014'])
print('Pearson Correlation Coefficient, 2-tailed p-value')
print(ols_coeff)
