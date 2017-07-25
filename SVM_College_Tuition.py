###############################
#
# Exploring Effect of Standardization on
# Linear Support Vector Machines
#
# Author: David Sotelo
# sotelo.d.a@gmail.com
# Date: 7.22.2017
#
# Based on college scorecard data available from US
# government and downloaded from Kaggle
# https://www.kaggle.com/kaggle/college-scorecard
# on 7.19.2017
#
###############################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from time import time

# Read in College Scorecard data, keeping relevant fields
# Remove missing data, print descriptive statistics
cscr=pd.read_csv('Scorecard.csv',
    usecols=['INSTNM','CONTROL','SAT_AVG','TUITIONFEE_IN','faminc'],
    encoding='ISO-8859-1')
cscr.loc[cscr.faminc=='PrivacySuppressed',['faminc']]=np.nan
cscr.faminc=cscr.faminc.astype('float64')
cscr=cscr.dropna(axis=0)
unq,cnt=np.unique(cscr.CONTROL,return_counts=True)
print('@@@@@@@@@@@@@@@@@@@@@@@@')
print('Count of college by institution type')
print('[Private Non-Profit, Private For-Profit, Public]')
print(cnt)
print('@@@@@@@@@@@@@@@@@@@@@@@@')
print('Descriptive statistics for private institutions only')
print(cscr.loc[(cscr.CONTROL.isin(['Private nonprofit','Private for-profit'])),
    :].describe())

# Remove public institutions
cscr=cscr.loc[(cscr.CONTROL.isin(['Private nonprofit','Private for-profit'])),
    :]

# Create tuition class categorical variable.  This will be our target class
# variable for prediction
cscr['TUIT20']=0.0
cscr.loc[(cscr.TUITIONFEE_IN>=20000.0),['TUIT20']]=1.0

# Plot datapoints on x-y scatter
x1=cscr.loc[cscr['TUIT20']==0.0,['SAT_AVG']]
y1=cscr.loc[cscr['TUIT20']==0.0,['faminc']]
x2=cscr.loc[cscr['TUIT20']==1.0,['SAT_AVG']]
y2=cscr.loc[cscr['TUIT20']==1.0,['faminc']]
plt.figure(figsize=(8,6))
plt.scatter(x1,y1,c='green',marker='o',edgecolors='k',alpha=0.4,
    label='Under $20k')
plt.scatter(x2,y2,c='red',marker='o',edgecolors='k',alpha=0.4,
    label='$20k & Over')
plt.legend()
plt.xlabel('Average SAT Score')
plt.ylabel('Family Income')
plt.title('Private College Tuition Over/Under $20k')
plt.ylim(0,150000)
plt.xlim(500,1600)
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
c_coeff=np.array([0.005,0.02,0.05,0.2,0.5,1,2,5,10,25,100])

# Iterate across all 'c' coefficients and record accuracy & linear params
t1=time()
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
t2=time()

print('@@@@@@@@@@@@@@@@@@@@@@@@')
print('Run-time for 11 SVM runs on non-standardized data')
print(t2-t1,' seconds')

# Standardize training features and apply standardization to test features
# Record mean and standard deviations for later use
scaler=StandardScaler()
feat_train_sc=scaler.fit_transform(feat_train)
feat_test_sc=scaler.transform(feat_test)
mean=np.mean(feat_train)
std=np.std(feat_train)
acc_train_sc=pd.DataFrame([],index=None,columns=['C','measure','coef1','coef2',
    'inter'])
acc_test_sc=pd.DataFrame([],index=None,columns=['C','measure','coef1','coef2',
    'inter'])
    
# Iterate across all 'c' coefficients and record accuracy & linear params
t1=time()
for i in c_coeff:
    print(i)
    svf=SVC(C=i,kernel='linear')
    svf=svf.fit(feat_train_sc,target_train)
    train_acc_sc=sum(svf.predict(feat_train_sc)==target_train)/len(target_train)
    test_acc_sc=sum(svf.predict(feat_test_sc)==target_test)/len(target_test)
    acc_train_sc=acc_train_sc.append(pd.DataFrame(np.reshape([i,train_acc_sc,
        svf.coef_[(0,0)],svf.coef_[(0,1)],svf.intercept_[0]],(1,5)),
        columns=['C','measure','coef1','coef2','inter'],index=None))
    acc_test_sc=acc_test_sc.append(pd.DataFrame(np.reshape([i,test_acc_sc,
        svf.coef_[(0,0)],svf.coef_[(0,1)],svf.intercept_[0]],(1,5)),
        columns=['C','measure','coef1','coef2','inter'],index=None))
t2=time()

print('@@@@@@@@@@@@@@@@@@@@@@@@')
print('Run-time for 11 SVM runs on standardized data')
print(t2-t1,' seconds')

# Print accuracy measures of non-standardized & standardized fitted SVMs
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print('SVM training & test accuracy-')
print('Non-standardized data:')
print(acc_train[['C','measure']])
print(acc_test[['C','measure']])

print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print('SVM training & test accuracy-')
print('Standardized data:')
print(acc_train_sc[['C','measure']])
print(acc_test_sc[['C','measure']])
  
# Set up facet grid showing plots of results
f,ax=plt.subplots(2,3)
f.set_size_inches(15,8)
f.text(x=0,y=0.5,s='Family Income',rotation='vertical',size=12)
f.text(x=0.02,y=0.8,s='Non-Standardized',rotation='vertical',size=10)
f.text(x=0.02,y=0.28,s='Standardized',rotation='vertical',size=10)
f.text(x=0.5,y=0,s='Average SAT Score',rotation='horizontal',size=12)
f.text(x=0.15,y=0.92,s='C=0.05',size=12)
f.text(x=0.5,y=0.92,s='C=1.00',size=12)
f.text(x=0.85,y=0.92,s='C=25.0',size=12)
f.suptitle('Private College Tuition Over/Under $20k, with Linear SVM Hyperplane',
    size=16)
f.tight_layout(rect=[0.05,0,1,0.93]) 
    
# Plot linear decision planes having coef C=0.05 for non-standardized data
xr=np.arange(1,1600)
slope=acc_train.loc[acc_train.C==0.05,['coef1']].values/(-1*
    acc_train.loc[acc_train.C==0.05,['coef2']].values)              
intercept=acc_train.loc[acc_train.C==0.05,['inter']].values/(-1*
    acc_train.loc[acc_train.C==0.05,['coef2']].values)
intercept_p1=(acc_train.loc[acc_train.C==0.05,['inter']]-1).values/(-1*
    acc_train.loc[acc_train.C==0.05,['coef2']].values)
intercept_m1=(acc_train.loc[acc_train.C==0.05,['inter']]+1).values/(-1*
    acc_train.loc[acc_train.C==0.05,['coef2']].values)
yr=np.ravel(xr*slope+intercept)
yrp1=np.ravel(xr*slope+intercept_p1)
yrm1=np.ravel(xr*slope+intercept_m1)
ax[0,0].plot(xr,yr,linestyle="-",c='blue')
ax[0,0].plot(xr,yrp1,linestyle="--",c='blue')
ax[0,0].plot(xr,yrm1,linestyle="--",c='blue')

# Plot linear decision planes having coef C=1 for non-standardized data
xr=np.arange(1,1600)
slope=acc_train.loc[acc_train.C==1.0,['coef1']].values/(-1*
    acc_train.loc[acc_train.C==1.0,['coef2']].values)              
intercept=acc_train.loc[acc_train.C==1.0,['inter']].values/(-1*
    acc_train.loc[acc_train.C==1.0,['coef2']].values)
intercept_p1=(acc_train.loc[acc_train.C==1.0,['inter']]-1).values/(-1*
    acc_train.loc[acc_train.C==1.0,['coef2']].values)
intercept_m1=(acc_train.loc[acc_train.C==1.0,['inter']]+1).values/(-1*
    acc_train.loc[acc_train.C==1.0,['coef2']].values)
yr=np.ravel(xr*slope+intercept)
yrp1=np.ravel(xr*slope+intercept_p1)
yrm1=np.ravel(xr*slope+intercept_m1)
ax[0,1].plot(xr,yr,linestyle="-",c='blue')
ax[0,1].plot(xr,yrp1,linestyle="--",c='blue')
ax[0,1].plot(xr,yrm1,linestyle="--",c='blue')

# Plot linear decision planes having coef C=25.0 for non-standardized data
xr=np.arange(1,1600)
slope=acc_train.loc[acc_train.C==25.0,['coef1']].values/(-1*
    acc_train.loc[acc_train.C==25.0,['coef2']].values)              
intercept=acc_train.loc[acc_train.C==25.0,['inter']].values/(-1*
    acc_train.loc[acc_train.C==25.0,['coef2']].values)
intercept_p1=(acc_train.loc[acc_train.C==25.0,['inter']]-1).values/(-1*
    acc_train.loc[acc_train.C==25.0,['coef2']].values)
intercept_m1=(acc_train.loc[acc_train.C==25.0,['inter']]+1).values/(-1*
    acc_train.loc[acc_train.C==25.0,['coef2']].values)
yr=np.ravel(xr*slope+intercept)
yrp1=np.ravel(xr*slope+intercept_p1)
yrm1=np.ravel(xr*slope+intercept_m1)
ax[0,2].plot(xr,yr,linestyle="-",c='blue')
ax[0,2].plot(xr,yrp1,linestyle="--",c='blue')
ax[0,2].plot(xr,yrm1,linestyle="--",c='blue')

# Plot linear decision planes having coef C=0.05 for standardized data
xr=np.arange(-4,4,0.01)
slope=acc_train_sc.loc[acc_train_sc.C==0.05,['coef1']].values/(-1*
    acc_train_sc.loc[acc_train_sc.C==0.05,['coef2']].values)              
intercept=acc_train_sc.loc[acc_train_sc.C==0.05,['inter']].values/(-1*
    acc_train_sc.loc[acc_train_sc.C==0.05,['coef2']].values)
intercept_p1=(acc_train_sc.loc[acc_train_sc.C==0.05,['inter']]-1).values/(-1*
    acc_train_sc.loc[acc_train_sc.C==0.05,['coef2']].values)
intercept_m1=(acc_train_sc.loc[acc_train_sc.C==0.05,['inter']]+1).values/(-1*
    acc_train_sc.loc[acc_train_sc.C==0.05,['coef2']].values)
yr=np.ravel(xr*slope+intercept)
yrp1=np.ravel(xr*slope+intercept_p1)
yrm1=np.ravel(xr*slope+intercept_m1)
ax[1,0].plot(xr,yr,linestyle="-",c='purple')
ax[1,0].plot(xr,yrp1,linestyle="--",c='purple')
ax[1,0].plot(xr,yrm1,linestyle="--",c='purple')

# Plot linear decision planes having coef C=1 for standardized data
xr=np.arange(-4,4,0.01)
slope=acc_train_sc.loc[acc_train_sc.C==1.0,['coef1']].values/(-1*
    acc_train_sc.loc[acc_train_sc.C==1.0,['coef2']].values)              
intercept=acc_train_sc.loc[acc_train_sc.C==1.0,['inter']].values/(-1*
    acc_train_sc.loc[acc_train_sc.C==1.0,['coef2']].values)
intercept_p1=(acc_train_sc.loc[acc_train_sc.C==1.0,['inter']]-1).values/(-1*
    acc_train_sc.loc[acc_train_sc.C==1.0,['coef2']].values)
intercept_m1=(acc_train_sc.loc[acc_train_sc.C==1.0,['inter']]+1).values/(-1*
    acc_train_sc.loc[acc_train_sc.C==1.0,['coef2']].values)
yr=np.ravel(xr*slope+intercept)
yrp1=np.ravel(xr*slope+intercept_p1)
yrm1=np.ravel(xr*slope+intercept_m1)
yr=np.ravel(xr*slope+intercept)
ax[1,1].plot(xr,yr,linestyle="-",c='purple')
ax[1,1].plot(xr,yrp1,linestyle="--",c='purple')
ax[1,1].plot(xr,yrm1,linestyle="--",c='purple')

# Plot linear decision planes having coef C=25.0 for standardized data
xr=np.arange(-4,4,0.01)
slope=acc_train_sc.loc[acc_train_sc.C==25.0,['coef1']].values/(-1*
    acc_train_sc.loc[acc_train_sc.C==25.0,['coef2']].values)              
intercept=acc_train_sc.loc[acc_train_sc.C==25.0,['inter']].values/(-1*
    acc_train_sc.loc[acc_train_sc.C==25.0,['coef2']].values)
intercept_p1=(acc_train_sc.loc[acc_train_sc.C==25.0,['inter']]-1).values/(-1*
    acc_train_sc.loc[acc_train_sc.C==25.0,['coef2']].values)
intercept_m1=(acc_train_sc.loc[acc_train_sc.C==25.0,['inter']]+1).values/(-1*
    acc_train_sc.loc[acc_train_sc.C==25.0,['coef2']].values)
yr=np.ravel(xr*slope+intercept)
yrp1=np.ravel(xr*slope+intercept_p1)
yrm1=np.ravel(xr*slope+intercept_m1)
ax[1,2].plot(xr,yr,linestyle="-",c='purple')
ax[1,2].plot(xr,yrp1,linestyle="--",c='purple')
ax[1,2].plot(xr,yrm1,linestyle="--",c='purple')

# Re-plot datapoints on x-y scatter for (2,3) subplots
# non-standardized data in [0,:], standardized data in [1,:]
ai=np.array(target_test==0.0)
bi=np.array(target_test==1.0)
i=0
for j in range(len(ax[i])):
    x1=feat_test.loc[target_test==0.0,['SAT_AVG']]
    y1=feat_test.loc[target_test==0.0,['faminc']]
    x2=feat_test.loc[target_test==1.0,['SAT_AVG']]
    y2=feat_test.loc[target_test==1.0,['faminc']]
    ax[i,j].scatter(x1,y1,c='green',marker='o',edgecolors='k',alpha=0.4,
        label='Under $20k')
    ax[i,j].scatter(x2,y2,c='red',marker='o',edgecolors='k',alpha=0.4,
        label='$20k & Over')
    ax[i,j].legend(loc=2)
    ax[i,j].set_yticks([])
i=1
for j in range(len(ax[i])):
    x1=feat_test_sc[ai,0]
    y1=feat_test_sc[ai,1]
    x2=feat_test_sc[bi,0]
    y2=feat_test_sc[bi,1]
    ax[i,j].scatter(x1,y1,c='green',marker='o',edgecolors='k',alpha=0.4,
        label='Under $20k')
    ax[i,j].scatter(x2,y2,c='red',marker='o',edgecolors='k',alpha=0.4,
        label='$20k & Over')
    ax[i,j].legend(loc=2)
    ax[i,j].set_yticks([])
    
#Miscellaneous final formatting
ax[0,0].set_yticks(np.arange(0,150000,25000))
ax[1,0].set_yticks(np.arange((0-mean[1])/std[1],(150000-mean[1])/std[1],1))
for i in range(0,3):
    ax[0,i].set_ylim([0,150000])
    ax[0,i].set_xlim([500,1600])
    ax[1,i].set_ylim([(0-mean[1])/std[1],(150000-mean[1])/std[1]])
    ax[1,i].set_xlim([(500-mean[0])/std[0],(1600-mean[0])/std[0]])

plt.show()
