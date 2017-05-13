###################################
### 
###  Examining speed dating results using Logistic regression
###  Author: Dave Sotelo
###  Date: 2017.05.13
###
###  Data freely available at Kaggle.com
###  
###################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import statsmodels.api as sma

# Read in speed dating CSV file
# "Gender Differences in Mate Selection: Evidence From a Speed Dating
#  Experiment."
# Authors: Fisman, Ray; Iyengar, Sheena
# Retrieved from:
# https://www.kaggle.com/annavictoria/speed-dating-experiment
spd_dat=pd.read_csv('Speed Dating Data.csv', low_memory=False,
    encoding='ISO-8859-1')

# Subset data.  Include one gender only. Each record is a 2-way
# interaction
spd_wk=spd_dat.loc[spd_dat['gender']==0,['iid','gender','match',
    'attr','attr_o','sinc','sinc_o','intel','intel_o','fun','fun_o',
    'amb','amb_o','shar','shar_o']]
spd_wk=spd_wk.dropna(axis=0)

# Create new variables: 2-way attribute averages
spd_wk['attr_2avg']=abs((spd_wk['attr']+spd_wk['attr_o'])/2-10)
spd_wk['sinc_2avg']=abs((spd_wk['sinc']+spd_wk['sinc_o'])/2-10)
spd_wk['intel_2avg']=abs((spd_wk['intel']+spd_wk['intel_o'])/2-10)
spd_wk['fun_2avg']=abs((spd_wk['fun']+spd_wk['fun_o'])/2-10)
spd_wk['amb_2avg']=abs((spd_wk['amb']+spd_wk['amb_o'])/2-10)
spd_wk['shar_2avg']=abs((spd_wk['shar']+spd_wk['shar_o'])/2-10)

spd_wk=spd_wk.drop(labels=['attr','attr_o','sinc','sinc_o','intel',
    'intel_o','fun','fun_o','amb','amb_o','shar','shar_o'],axis=1)

spd_wk['intercept']=1.0

# Interim dataframe diagnostics
print('Speed Dating dataframe shape')
print(spd_wk.shape)
print('Speed Dating dataframe column names')
print(spd_wk.columns)
print('Speed Dating dataframe summary')
print(spd_wk.describe())

# Fit the Logistic Regression, view summary statistics
indepv = spd_wk.columns[3:]
logreg = sma.Logit(spd_wk['match'],spd_wk[indepv])
logfit = logreg.fit()
print(logfit.summary2())

sbn.set_style('whitegrid')

# Holding all other variables @ 0 (optimal outcome), plot 'attr'
coeff1=logfit.params['attr_2avg']
inter=logfit.params['intercept']
x1=np.arange(0,10,0.1)
plt.plot(x1,np.exp(inter+coeff1*x1)/(1+np.exp(inter+coeff1*x1)),
    color='r')

# Holding all other variables @ 0 (optimal outcome), plot 'amb'
coeff2=logfit.params['amb_2avg']
x2=np.arange(0,10,0.1)
plt.plot(x2,np.exp(inter+coeff2*x2)/(1+np.exp(inter+coeff2*x2)),
    color='g')

# Holding all other variables @ 0 (optimal outcome), plot 'fun'
coeff3=logfit.params['fun_2avg']
x3=np.arange(0,10,0.1)
plt.plot(x3,np.exp(inter+coeff3*x3)/(1+np.exp(inter+coeff3*x3)),
    color='b')

# Plot fitted regression
plt.xlim(0,10)
plt.xlabel('subject/respondent average attribute score')
plt.ylim(0,1)
plt.ylabel('p(match=1)')
plt.legend(['Attractive','Ambition','Fun'],loc='lower left',
    title='2-way Attrib.',frameon=True)
plt.title('P(match=1) by attribute, holding other attributes=0',
    x=0.5,y=1.0,size=18)
plt.show()


