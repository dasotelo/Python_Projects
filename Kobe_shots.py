###################################
### Include this header at start of all IDLE .py scripts
import sys
sys.path.append('C:\\Anaconda3\Lib\site-packages')

import matplotlib
matplotlib.use('TkAgg')
###################################

###################################
#
# Analysis of NBA shooting percentages using
# Bayesian statistics & stochastic simulation
#
# Author: Dave Sotelo
# Date: 6/21/2017
#
# Data freely available at kaggle.com
# https://www.kaggle.com/c/kobe-bryant-shot-selection/data
#
###################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta

# Read data.  Keep records with non-missing shot_made information
kobe=pd.read_csv('data.csv')
kobe=kobe.loc[kobe.shot_made_flag.notnull(),:]

# Slice dataframe = games in October and November 2005 only
kobe_05=kobe.loc[((kobe['game_date'].str[:4]=='2005')& \
    (kobe['game_date'].str[5:7]=='11')),:]

# Calculate jump shot statistics for '05 Oct/Nov & print results
kobe_05js2=kobe_05.loc[((kobe_05.combined_shot_type=='Jump Shot')& \
    (kobe_05.shot_type=='2PT Field Goal')),:]
kobe_05itp=kobe_05js2.loc[kobe_05js2['shot_zone_basic']=='In The Paint (Non-RA)',:]
kobe_05mr=kobe_05js2.loc[kobe_05js2['shot_zone_basic']=='Mid-Range',:]
unq_05itp,cnt_05itp=np.unique(kobe_05itp.shot_made_flag,return_counts=True)
unq_05mr,cnt_05mr=np.unique(kobe_05mr.shot_made_flag,return_counts=True)
print('2005 Nov in-the-paint made jump shots: ',cnt_05itp[unq_05itp==1.])
print('     in-the-paint missed jump shots: ',cnt_05itp[unq_05itp==0.])
print('     in-the-paint shot pct: ',cnt_05itp[unq_05itp==1.]/sum(cnt_05itp))
print('2005 Nov mid-range made jump shots: ',cnt_05mr[unq_05mr==1.])
print('     mid-range missed jump shots: ',cnt_05mr[unq_05mr==0.])
print('     mid-range shot pct: ',cnt_05mr[unq_05mr==1.]/sum(cnt_05mr))

# Set up parameters for beta distributions of '05 Oct/Nov jump shots & plot
ipta=cnt_05itp[unq_05itp==1.]
itpb=cnt_05itp[unq_05itp==0.]
mra=cnt_05mr[unq_05mr==1.]
mrb=cnt_05mr[unq_05mr==0.]
x=np.arange(0.001,1,0.001)

plt.plot(x,beta.pdf(x,ipta,itpb),c='blue',label='In-the-paint pdf')
plt.plot(x,beta.pdf(x,mra,mrb),c='orange',label='Mid-Range pdf')
plt.fill_between(x,0,beta.pdf(x,ipta,itpb),facecolor='blue',alpha=0.3)
plt.fill_between(x,0,beta.pdf(x,mra,mrb),facecolor='orange',alpha=0.3)
plt.title('Kobe 2005 Nov: Mid-range vs. In-the-paint Jump Shots',size=14)
plt.legend(loc='upper right')
plt.xlim(0.10,0.75)
plt.xlabel('True shot percentage')
plt.show()

# Simulate random draws from beta distributions.  Plot differences
# in a histogram
n_sims=100000
itp_randn=np.random.beta(ipta,itpb,size=n_sims)
mr_randn=np.random.beta(mra,mrb,size=n_sims)
itp_minus_mr=itp_randn-mr_randn
cats=7
n,bins,patches=plt.hist(itp_minus_mr,cats,facecolor='b',alpha=0.5,edgecolor='k')
plt.title('Kobe 2005 Nov In-the-paint Minus Mid-range, nsims=100,000',
    size=13)
plt.xlabel('In-the-paint minus Mid-range')
for i in range(cats):
    plt.text(bins[i]+0.002,n[i]+200,'{:,.0f}'.format(n[i]))
plt.xticks(bins[1:-1])
plt.show()

# Beta coefficients for prior distribution
pri_a=72
pri_b=82

# Calculate posterior distributions and plot pdf's
plt.plot(x,beta.pdf(x,ipta+pri_a,itpb+pri_b),c='green',label='In-the-paint pdf')
plt.plot(x,beta.pdf(x,mra+pri_a,mrb+pri_b),c='purple',label='Mid-Range pdf')
plt.fill_between(x,0,beta.pdf(x,ipta+pri_a,itpb+pri_b),facecolor='green',
    alpha=0.3)
plt.fill_between(x,0,beta.pdf(x,mra+pri_a,mrb+pri_b),facecolor='purple',
    alpha=0.3)
plt.title('Posterior Distributions of Mid-range vs. In-the-paint Jump Shots',
    size=14)
plt.legend(loc='upper right')
plt.xlim(0.10,0.75)
plt.xlabel('True shot percentage')
plt.show()

# Simulate random draws from beta distributions.  Plot differences
# in a histogram
n_sims=100000
itp_randn=np.random.beta(ipta+pri_a,itpb+pri_b,size=n_sims)
mr_randn=np.random.beta(mra+pri_a,mrb+pri_b,size=n_sims)
itp_minus_mr=itp_randn-mr_randn
cats=7
n,bins,patches=plt.hist(itp_minus_mr,cats,facecolor='g',alpha=0.5,edgecolor='k')
plt.title('Posterior Nov In-the-paint Minus Mid-range, nsims=100,000',
    size=13)
plt.xlabel('In-the-paint minus Mid-range')
for i in range(cats):
    plt.text(bins[i]+0.002,n[i]+200,'{:,.0f}'.format(n[i]))
plt.xticks(bins[1:-1])
plt.show()

# Compare to Kobe's full-season 05-06 shooting performance.  re-slice
# dataframe and calculate comparison statistics
kobe_0506full=kobe.loc[((kobe['game_date'].str[:4].isin(['2005','2006']))& \
    (kobe['game_date'].str[5:7].isin(['11','12','01','02','03','04','05','06']))& \
    (kobe.shot_type=='2PT Field Goal')),:]
kobe_0506itp=kobe_0506full.loc[kobe_0506full['shot_zone_basic']=='In The Paint (Non-RA)',:]
kobe_0506mr=kobe_0506full.loc[kobe_0506full['shot_zone_basic']=='Mid-Range',:]
unq_0506itp,cnt_0506itp=np.unique(kobe_0506itp.shot_made_flag,return_counts=True)
unq_0506mr,cnt_0506mr=np.unique(kobe_0506mr.shot_made_flag,return_counts=True)
print('2005/06 FY in-the-paint made jump shots: ',cnt_0506itp[unq_0506itp==1.])
print('     in-the-paint missed jump shots: ',cnt_0506itp[unq_0506itp==0.])
print('     in-the-paint shot pct: ',cnt_0506itp[unq_0506itp==1.]/sum(cnt_0506itp))
print('2005/06 FY mid-range made jump shots: ',cnt_0506mr[unq_0506mr==1.])
print('     mid-range missed jump shots: ',cnt_0506mr[unq_0506mr==0.])
print('     mid-range shot pct: ',cnt_0506mr[unq_0506mr==1.]/sum(cnt_0506mr))
