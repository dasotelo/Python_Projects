##############################################
#
#  Exploring Pulsating Variable Stars with Gaussian Processes
#  Author: Dave Sotelo
#  sotelo.d.a@gmail.com
#  March, 2018
#
#  Visual luminosity observations from the AAVSO International Database,
#  contributed and made freely available by observers worldwide.
#  https://www.aavso.org/
#
##############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as GPR               
from sklearn.gaussian_process.kernels import RBF,RationalQuadratic
from sklearn.gaussian_process.kernels import ExpSineSquared,Matern
from sklearn.gaussian_process.kernels import WhiteKernel

# Control size of sample taken from data & used to train GPR
sample_size=1000

# Read in data.  Keep visual band observations only & filter to 
# dates 01/01/1985 & later
omicet=pd.read_csv('aavsodata_5ab14bf940378.csv',low_memory=False)
omicet.loc[omicet['Magnitude'].str.contains('<'),'Magnitude']=np.nan
omicet=omicet.loc[((np.isreal(omicet.Magnitude))&(omicet.Band=='Vis.')), \
  ('JD','Magnitude')]
omicet.Magnitude=omicet['Magnitude'].astype('float64')
filter=np.array(np.isnan(omicet.Magnitude))
inv_filter=filter==False
omicet=omicet.loc[inv_filter,:]
omicet=omicet.loc[(omicet.JD>=2446066),:]

# Plot visual observations in scatter plot
f,ax=plt.subplots(2)
f.set_size_inches(9.5,4.5)
f.suptitle('Omicron Ceti (Mira) Stellar Magnitude - Visual Band', \
  weight='bold',x=0.45,y=0.97,size=18)
for i in range(0,2):
	ax[i].xaxis.set_tick_params(labelsize=8)
	ax[i].yaxis.set_tick_params(labelsize=8)
ax[0].scatter(omicet.loc[((omicet.JD >= 2446066) & \
  (omicet.JD < 2452131)),'JD'], omicet.loc[((omicet.JD >= 2446066) & \
  (omicet.JD < 2452131)),'Magnitude'],s=4,c='orange',edgecolors='k', \
  linewidths=0.2)
ax[1].scatter(omicet.loc[((omicet.JD >= 2452131) & \
  (omicet.JD < 2458196)),'JD'], omicet.loc[((omicet.JD >= 2452131) & \
  (omicet.JD < 2458196)),'Magnitude'],s=4,c='orange',edgecolors='k', \
  linewidths=0.2)
plt.tight_layout(rect=[0,0,1,0.93])
plt.show()

# Separate training data from test data
# GPR will be trained using data spanning the period 01/01/1985 - 01/01/2008
omicet_train=omicet.loc[omicet.JD < 2454466.5,:]
train_idx=np.random.choice(omicet_train.index,sample_size)
x_train=np.array(omicet_train.JD[train_idx])[:,np.newaxis]
y_train=np.array(omicet_train.Magnitude[train_idx])[:,np.newaxis]
X_=np.linspace(2446066,2454466,8400)
PrX_=np.linspace(2454467,2458196,3729)

# Train a GPR with a single ExpSineSquared kernel
# Predict observations over the test data domain, calculate residuals 
# & RMSE
kern=1*ExpSineSquared(length_scale=10,periodicity=330, \
  length_scale_bounds=(1,1000),periodicity_bounds=(300,350)) \
  + WhiteKernel(noise_level=1,noise_level_bounds=(0.01,0.5))
gp=GPR(kernel=kern)
gp.fit(x_train,y_train) 
y_mean=gp.predict(X_[:,np.newaxis])
pr_mean=gp.predict(PrX_[:,np.newaxis])
yy_mean=gp.predict(np.array(omicet.loc[omicet.JD>=2454466.5,'JD'] \
  )[:,np.newaxis])
yy_resid=np.array(omicet.loc[omicet.JD>=2454466.5,'Magnitude'] \
  )-np.squeeze(yy_mean)
RMSE_1=np.average(yy_resid**2)**0.5
print(gp.kernel_)

# Plot predicted observations from fitted GPR over train & test data
# domain on X,Y scatter plot.  
f,ax=plt.subplots(2)
f.set_size_inches(9.5,4.5)
f.suptitle('Omicron Ceti (Mira) Stellar Magnitude - Visual Band', \
  weight='bold',x=0.45,y=0.97,size=18)
for i in range(0,2):
	ax[i].xaxis.set_tick_params(labelsize=8)
	ax[i].yaxis.set_tick_params(labelsize=8)
f.text(x=0.05,y=0.90,s=gp.kernel_,fontsize=8)
ax[0].scatter(omicet.loc[((omicet.JD >= 2446066) & \
  (omicet.JD < 2452131)),'JD'], omicet.loc[((omicet.JD >= 2446066) & \
  (omicet.JD < 2452131)),'Magnitude'],s=4,c='orange',edgecolors='k', \
  linewidths=0.2)
ax[0].plot(X_[X_<2452131],y_mean[:X_[X_<2452131].size],c='b',lw=1.5,ls='--')
ax[1].scatter(omicet.loc[((omicet.JD >= 2452131) & \
  (omicet.JD < 2458196)),'JD'], omicet.loc[((omicet.JD >= 2452131) & \
  (omicet.JD < 2458196)),'Magnitude'],s=4,c='orange',edgecolors='k', \
  linewidths=0.2)
ax[1].plot(X_[X_>=2452131],y_mean[X_[X_<2452131].size:],c='b',lw=1.5,ls='--')
ax[1].plot(PrX_,pr_mean,c='r',lw=1.5)
plt.tight_layout(rect=[0,0,1,0.91])
plt.show()

# Plot residuals
plt.figure(figsize=(7,2))
plt.scatter(omicet.loc[omicet.JD>=2454466.5,'JD'],np.squeeze(yy_resid), \
  s=4,c='k',edgecolors='k',alpha=0.5)
plt.tick_params(axis='both',labelsize=8)
plt.title('Residual Plot: ExpSineSquared Kernel')
plt.show()

##############################################
# All of the following 2-kernel GPRs were tested as part of this
# project.
##############################################

### PRODUCT of ExpSineSquared kernel & RBF + WhiteNoise ###
#kern=1*ExpSineSquared(length_scale=10,periodicity=330, \
#  length_scale_bounds=(1,1000),periodicity_bounds=(300,350)) \
#  * 1*RBF(length_scale=1,length_scale_bounds=(1,1000)) \
#  + WhiteKernel(noise_level=1,noise_level_bounds=(0.01,0.5))

### SUM of ExpSineSquared kernel & Matern + WhiteNoise ###
#kern=1*ExpSineSquared(length_scale=10,periodicity=330, \
#  length_scale_bounds=(1,1000),periodicity_bounds=(300,350)) \
#  + 1*Matern(length_scale=30, length_scale_bounds=(1, 1000),nu=1.5) \
#  + WhiteKernel(noise_level=1,noise_level_bounds=(0.01,0.5))

### PRODUCT of 2 ExpSineSquared kernels + WhiteNoise ###
#kern=1*ExpSineSquared(length_scale=10,periodicity=330, \
#  length_scale_bounds=(1,1000),periodicity_bounds=(300,350)) \
#  * 1*ExpSineSquared(length_scale=10,periodicity=1000, \
#  length_scale_bounds=(1,1000),periodicity_bounds=(500,3000)) \
#  + WhiteKernel(noise_level=1,noise_level_bounds=(0.01,0.5))

### SUM of 2 ExpSineSquared kernels + WhiteNoise ###
kern=1*ExpSineSquared(length_scale=10,periodicity=330, \
  length_scale_bounds=(1,1000),periodicity_bounds=(300,350)) \
  + 1*ExpSineSquared(length_scale=10,periodicity=1000, \
  length_scale_bounds=(1,1000),periodicity_bounds=(500,3000)) \
  + WhiteKernel(noise_level=1,noise_level_bounds=(0.01,0.5))
gp=GPR(kernel=kern)
gp.fit(x_train,y_train) 
y_mean=gp.predict(X_[:,np.newaxis])
pr_mean=gp.predict(PrX_[:,np.newaxis])
yy_mean=gp.predict(np.array(omicet.loc[omicet.JD>=2454466.5,'JD'] \
  )[:,np.newaxis])
yy_resid=np.array(omicet.loc[omicet.JD>=2454466.5,'Magnitude'] \
  )-np.squeeze(yy_mean)
RMSE_2=np.average(yy_resid**2)**0.5
print(gp.kernel_)

# Plot predicted observations from fitted GPR over train & test data
# domain on X,Y scatter plot.
f,ax=plt.subplots(2)
f.set_size_inches(9.5,4.5)
f.suptitle('Omicron Ceti (Mira) Stellar Magnitude - Visual Band', \
  weight='bold',x=0.45,y=0.97,size=18)
for i in range(0,2):
	ax[i].xaxis.set_tick_params(labelsize=8)
	ax[i].yaxis.set_tick_params(labelsize=8)
f.text(x=0.05,y=0.90,s=gp.kernel_,fontsize=8)
ax[0].scatter(omicet.loc[((omicet.JD >= 2446066) & \
  (omicet.JD < 2452131)),'JD'], omicet.loc[((omicet.JD >= 2446066) & \
  (omicet.JD < 2452131)),'Magnitude'],s=4,c='orange',edgecolors='k', \
  linewidths=0.2)
ax[0].plot(X_[X_<2452131],y_mean[:X_[X_<2452131].size],c='b',lw=1.5,ls='--')
ax[1].scatter(omicet.loc[((omicet.JD >= 2452131) & \
  (omicet.JD < 2458196)),'JD'], omicet.loc[((omicet.JD >= 2452131) & \
  (omicet.JD < 2458196)),'Magnitude'],s=4,c='orange',edgecolors='k', \
  linewidths=0.2)
ax[1].plot(X_[X_>=2452131],y_mean[X_[X_<2452131].size:],c='b',lw=1.5,ls='--')
ax[1].plot(PrX_,pr_mean,c='r',lw=1.5)
plt.tight_layout(rect=[0,0,1,0.93])
plt.show()

# Plot residuals
plt.figure(figsize=(7,2))
plt.scatter(omicet.loc[omicet.JD>=2454466.5,'JD'],np.squeeze(yy_resid), \
  s=4,c='k',edgecolors='k',alpha=0.5)
plt.tick_params(axis='both',labelsize=8)
plt.title('Residual Plot: Sum of 2 ExpSineSquared Kernels')
plt.show()

print("RMSE of single kernel GPR:",RMSE_1)
print("RMSE of double kernel GPR:",RMSE_2)