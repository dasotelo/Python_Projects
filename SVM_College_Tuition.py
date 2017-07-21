###################################
### Include this header at start of all IDLE .py scripts
import sys
sys.path.append('C:\\Anaconda3\Lib\site-packages')

import matplotlib
matplotlib.use('TkAgg')
###################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from datetime import datetime

# Read in College Scorecard data, keeping relevant fields
# Remove missing data, print descriptive statistics
cscr=pd.read_csv('Most-Recent-Cohorts-All-Data-Elements2.csv',
    usecols=['INSTNM','CONTROL','SAT_AVG','TUITIONFEE_IN','FAMINC'],
    encoding='ISO-8859-1',low_memory=False)
cscr=cscr.dropna(axis=0)
cscr.loc[cscr.FAMINC=='PrivacySuppressed',['FAMINC']]=np.nan
cscr.FAMINC=cscr.FAMINC.astype('float64')
unq,cnt=np.unique(cscr.CONTROL,return_counts=True)
print('@@@@@@@@@@@@@@@@@@@@@@@@')
print('Count of college by institution type')
print('[Public, Private Non-Profit, Private For-Profit]')
print(cnt)
print('@@@@@@@@@@@@@@@@@@@@@@@@')
print('Descriptive statistics for Private institutions only')
print(cscr.loc[(cscr.CONTROL.isin([2,3])),:].describe())

# Remove public institutions
cscr=cscr.loc[(cscr.CONTROL.isin([2,3])),:]

# Create tuition class categorical variable.  This will be our target class
# variable for prediction
cscr['TUIT30']=0.0
cscr.loc[(cscr.TUITIONFEE_IN>=30000.0),['TUIT30']]=1.0

# Plot datapoints on x-y scatter
x1=cscr.loc[cscr['TUIT30']==0.0,['SAT_AVG']]
y1=cscr.loc[cscr['TUIT30']==0.0,['FAMINC']]
x2=cscr.loc[cscr['TUIT30']==1.0,['SAT_AVG']]
y2=cscr.loc[cscr['TUIT30']==1.0,['FAMINC']]
plt.scatter(x1,y1,c='green',marker='o',edgecolors='k',alpha=0.5,
    label='Under $30k')
plt.scatter(x2,y2,c='red',marker='o',edgecolors='k',alpha=0.5,
    label='$30k & Over')
plt.legend()
plt.xlabel('Average SAT Score')
plt.ylabel('Family Income')
plt.title('Private College Tuition Over/Under $30k')
plt.show()

# Split into 60/40 train/test datasets
cscr_train,cscr_test=train_test_split(cscr,test_size=0.4)

# Train Support Vector Machine (linear) on non-normalized training data
# Create feature and classification vectors & dataframes to hold results
feat_train=cscr_train.iloc[:,[2,4]]
target_train=cscr_train.iloc[:,5]
feat_test=cscr_test.iloc[:,[2,4]]
target_test=cscr_test.iloc[:,5]
acc_train=pd.DataFrame([],index=None,columns=['C','measure','coef1','coef2',
    'inter'])
acc_test=pd.DataFrame([],index=None,columns=['C','measure','coef1','coef2',
    'inter'])

# Test using the following values for coefficient 'c'
c_coeff=np.array([0.5,1,2,5,10,25,75,200])

# Iterate across all 'c' coefficients and record accuracy & linear params
for i in c_coeff:
    print(i)
    svf=SVC(C=i,kernel='linear')
    svf=svf.fit(feat_train,target_train)
    train_acc=sum(svf.predict(feat_train)==target_train)/len(target_train)
    test_acc=sum(svf.predict(feat_test)==target_test)/len(target_test)
    acc_train=acc_train.append(pd.DataFrame(np.reshape([i,train_acc,
        svf.coef_[(0,0)],svf.coef_[(0,1)],svf.intercept_[0]],(1,5)),
        columns=['C','measure','coef1','coef2','inter'],index=None))
    acc_test=acc_test.append(pd.DataFrame(np.reshape([i,test_acc,
        svf.coef_[(0,0)],svf.coef_[(0,1)],svf.intercept_[0]],(1,5)),
        columns=['C','measure','coef1','coef2','inter'],index=None))
