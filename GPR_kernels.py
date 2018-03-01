###################################
#
#  Comparing GPR kernels 
#  Using temperature observations in Belgium
#
#  Author: Dave Sotelo
#  2/28/2018
#  sotelo.d.a@gmail.com	
#
###################################

import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor                 
from sklearn.gaussian_process.kernels import RBF,RationalQuadratic
from sklearn.gaussian_process.kernels import ExpSineSquared,Matern
from sklearn.gaussian_process.kernels import WhiteKernel
import matplotlib.pyplot as plt
import datetime as dtm
from time import time

# Read in Belgium Temperature History and keep relevant fields
belg_data=pd.read_csv('Avg_Temp_Belgium.csv')
belg_data=belg_data.loc[:,('dt','AverageTemperature')]
belg_data['Date']=pd.to_datetime(belg_data.dt)
belg_data['Day_Num']=belg_data.Date-dtm.date(1959,12,31)
belg_data['Day_Num']=belg_data['Day_Num'].astype('timedelta64[D]')
belg_data=belg_data.loc[:,('Day_Num','AverageTemperature')]

# Define training and target vectors as well as sampling inputs
x_train=np.array(belg_data.Day_Num)[:,np.newaxis]
y_train=np.array(belg_data.AverageTemperature)[:,np.newaxis]
X_=np.linspace(1,19500,5000)

# Set up facet grid showing plots of fitted GPRs
f,ax=plt.subplots(2,2)
f.set_size_inches(11,7)

# Input prior kernels and parameters
# Several of these parameters need to be tuned by the user
# Based on the nature of the data being modeled
kern = [1*RBF(length_scale=1,length_scale_bounds=(0.1,3650)) \
  + WhiteKernel(noise_level=1,noise_level_bounds=(0.1,50)), \
  1*RationalQuadratic(length_scale=30,alpha=0.1, \
    length_scale_bounds=(0.1,3650),alpha_bounds=(1e-2,10)) \
  + WhiteKernel(noise_level=1,noise_level_bounds=(0.1,50)), \
  1*ExpSineSquared(length_scale=30,periodicity=365, \
    length_scale_bounds=(0.1,3650),periodicity_bounds=(30,3650)) \
  + WhiteKernel(noise_level=1,noise_level_bounds=(0.1,50)), \
  1*Matern(length_scale=30, length_scale_bounds=(0.1, 3650), \
    nu=1.5) \
  + WhiteKernel(noise_level=1,noise_level_bounds=(0.1,50))] 

# Plot data and fitted GPR predictions
# Display posterior kernel details and Log Marginal-Likelihood 
# Display training time of each GPR kernel
for i in range(0,2):
  for j in range(0,2):
    t1=time()
    gp=GaussianProcessRegressor(kernel=kern[i+1+j*2-1],normalize_y=True)
    gp.fit(x_train,y_train)
    y_mean,y_std=gp.predict(X_[:,np.newaxis],return_std=True)
    t2=time()
    print('@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('Kernel(prior): %s\nFit+Predict runtime: %.2f' % \
      (kern[i+1+j*2-1],t2-t1))
    print('@@@@@@@@@@@@@@@@@@@@@@@@@')
    ax[i,j].scatter(x_train[:,0],y_train[:,0],s=5,c='green',marker='o', \
      edgecolors='k',alpha=0.6)
    ax[i,j].plot(X_,y_mean,c='b',lw=1.5)
    ax[i,j].fill_between(X_,y_mean[:,0]-y_std,y_mean[:,0]+y_std,alpha=0.4, \
      color='b')
    ax[i,j].set_title('%s\nLog-Marginal-Likelihood: %.2f' % (gp.kernel_, \
      gp.log_marginal_likelihood(gp.kernel_.theta)),size=8)
    ax[i,j].xaxis.set_tick_params(labelsize=8)
    ax[i,j].xaxis.set_label_text("Days from 1/1/1960")
    ax[i,j].yaxis.set_tick_params(labelsize=8)
    ax[i,j].yaxis.set_label_text("Daily High Temp C")

plt.tight_layout()
plt.show()