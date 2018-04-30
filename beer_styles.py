##############################
#
# Exploring Linear Discriminant Analysis
# Author: Dave Sotelo
# April 30, 2018
#
# Data available from Kaggle.com, & scraped from the Brewer's Friend
# https://www.kaggle.com/jtrofe/beer-recipes
# https://www.brewersfriend.com/search/
# 
##############################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Read in beer recipe data.  Keep beer IBU & Color data for
# four beer styles:
#   Blonde Ale
#   Robust Porter
#   English IPA
#   Imperial Stout
bsty=pd.read_csv('recipeData.csv',low_memory=False,encoding='ISO-8859-1')
# Remove unreasonably high IBU entry for Imperial Stout
bsty=bsty.loc[bsty.BeerID!=48527,:]
# Remove entries w/ zero Color & IBU values
bsty=bsty.loc[(bsty.IBU!=0)&(bsty.Color!=0),:]
styles=['Blonde Ale','Robust Porter','English IPA','Imperial Stout']
features=['Style','Color','IBU']
bsty=bsty.loc[bsty.Style.isin(styles),features]

# Split into 70/30 training & testing datasets
bsty_train,bsty_test=train_test_split(bsty,test_size=0.3)

# Plot training dataset on pyplot scatterplot
plt.scatter(bsty_train.loc[bsty_train.Style=='Blonde Ale','Color'], \
  bsty_train.loc[bsty_train.Style=='Blonde Ale','IBU'],s=10,c='orange', \
  edgecolors='k',linewidth=0.7,alpha=0.5,label='Blonde Ale')
plt.scatter(bsty_train.loc[bsty_train.Style=='Robust Porter','Color'], \
  bsty_train.loc[bsty_train.Style=='Robust Porter','IBU'],s=10,c='red', \
  edgecolors='k',linewidth=0.7,alpha=0.5,label='Robust Porter')
plt.scatter(bsty_train.loc[bsty_train.Style=='English IPA','Color'], \
  bsty_train.loc[bsty_train.Style=='English IPA','IBU'],s=10,c='blue', \
  edgecolors='k',linewidth=0.7,alpha=0.5,label='English IPA')
plt.scatter(bsty_train.loc[bsty_train.Style=='Imperial Stout','Color'], \
  bsty_train.loc[bsty_train.Style=='Imperial Stout','IBU'],s=10,c='green', \
  edgecolors='k',linewidth=0.7,alpha=0.5,label='Imperial Stout')  
plt.legend(loc=2,fontsize='8')
plt.xlim(-2,52)
plt.ylim(-5,150)
plt.xlabel('Color')
plt.ylabel('IBUs')
plt.suptitle('Color & IBU for Four Major Beer Styles',weight='bold', \
  size='15',x=0.45,y=0.94)
plt.show()

# Standardize IBU & Color data for linear discriminant analysis
sscl=StandardScaler()
x_train=sscl.fit_transform(bsty_train.iloc[:,1:])
x_test=sscl.transform(bsty_test.iloc[:,1:])

# Train basic linear discriminant model.  Predict classes values
# for all observations in both training and test datasets
lda=LinearDiscriminantAnalysis()
lda.fit(x_train,bsty_train.Style)
bsty_train['predict']=bsty_train['Style']==lda.predict(x_train)
bsty_test['predict']=bsty_test['Style']==lda.predict(x_test)

# Print basic accuracy measures for test dataset class predictions
print('##  TEST DATA  ##########')
print('##  Overall Accuracy:',bsty_test.loc[bsty_test.predict==True, \
  'predict'].count(),'out of',bsty_test.loc[:,'predict'].count())
acc_rate=bsty_test.loc[bsty_test.predict==True, \
  'predict'].count()/bsty_test.loc[:,'predict'].count()
print('##  Overall Accuracy Rate:','{0:.0f}%'.format(acc_rate*100))
print('##  Blonde Ale Accuracy:',bsty_test.loc[(bsty_test.predict==True)& \
  (bsty_test.Style=='Blonde Ale'),'predict'].count(),'out of', \
  bsty_test.loc[bsty_test.Style=='Blonde Ale','predict'].count())
acc_rate=bsty_test.loc[(bsty_test.predict==True)& \
  (bsty_test.Style=='Blonde Ale'),'predict'].count()/ \
  bsty_test.loc[bsty_test.Style=='Blonde Ale','predict'].count()
print('##  Blonde Ale Accuracy Rate:','{0:.0f}%'.format(acc_rate*100))
print('##  English IPA Accuracy:',bsty_test.loc[(bsty_test.predict==True)& \
  (bsty_test.Style=='English IPA'),'predict'].count(),'out of', \
  bsty_test.loc[bsty_test.Style=='English IPA','predict'].count())
acc_rate=bsty_test.loc[(bsty_test.predict==True)& \
  (bsty_test.Style=='English IPA'),'predict'].count()/ \
  bsty_test.loc[bsty_test.Style=='English IPA','predict'].count()
print('##  English IPA Accuracy Rate:','{0:.0f}%'.format(acc_rate*100))
print('##  Imperial Stout Accuracy:',bsty_test.loc[(bsty_test.predict==True)& \
  (bsty_test.Style=='Imperial Stout'),'predict'].count(),'out of', \
  bsty_test.loc[bsty_test.Style=='Imperial Stout','predict'].count())
acc_rate=bsty_test.loc[(bsty_test.predict==True)& \
  (bsty_test.Style=='Imperial Stout'),'predict'].count()/ \
  bsty_test.loc[bsty_test.Style=='Imperial Stout','predict'].count()
print('##  Imperial Stout Accuracy Rate:','{0:.0f}%'.format(acc_rate*100))
print('##  Robust Porter Accuracy:',bsty_test.loc[(bsty_test.predict==True)& \
  (bsty_test.Style=='Robust Porter'),'predict'].count(),'out of', \
  bsty_test.loc[bsty_test.Style=='Robust Porter','predict'].count())
acc_rate=bsty_test.loc[(bsty_test.predict==True)& \
  (bsty_test.Style=='Robust Porter'),'predict'].count()/ \
  bsty_test.loc[bsty_test.Style=='Robust Porter','predict'].count()
print('##  Robust Porter Accuracy Rate:','{0:.0f}%'.format(acc_rate*100))

# Set up meshgrid for countour plot
# Contour plot will be used to show boundaries between class predictions
# by Color & IBU values on a scatterplot 
x1,y1=np.meshgrid(np.linspace(0,50,150),np.linspace(0,150,450))
p1=lda.predict_proba(sscl.transform(np.c_[x1.ravel(),y1.ravel()]))
Z=np.array([])

for i in np.arange(0,p1.shape[0]):
  if p1[i][0]==np.max(p1[i]):
    Z=np.append(Z,[0])
  elif p1[i][1]==np.max(p1[i]):
    Z=np.append(Z,[1])
  elif p1[i][2]==np.max(p1[i]):
    Z=np.append(Z,[2])
  elif p1[i][3]==np.max(p1[i]):
    Z=np.append(Z,[3])
  else:
    Z=np.append(Z,[99])
Zr=Z.reshape(x1.shape)

# Plot countour and scatterplots.  Misclassified test observations
# color coded in red.  
# Label & format
plt.contour(x1,y1,Zr,levels=[0.5,1.5,2.5,3.5],linewidths=1.5,colors='k', \
  linestyles='solid')
plt.scatter(bsty_test.loc[(bsty_test.predict==False),'Color'], \
  bsty_test.loc[(bsty_test.predict==False),'IBU'],s=10,c='red', \
  edgecolors='k',linewidth=0.7,alpha=0.5,label='Falsely Classified')
plt.scatter(bsty_test.loc[(bsty_test.predict==True),'Color'], \
  bsty_test.loc[(bsty_test.predict==True),'IBU'],s=10,c='w', \
  edgecolors='k',linewidth=0.7,alpha=0.5,label='Correctly Classified')
plt.legend(loc=2,fontsize='8')
plt.xlim(-2,52)
plt.ylim(-5,150)
plt.xlabel('Color')
plt.ylabel('IBUs')
plt.suptitle('LDA Accuracy for Four Major Beer Styles',weight='bold', \
  size='15',x=0.45,y=0.96)
plt.text(x=10,y=7,s='Blonde Ale')
plt.text(x=15,y=110,s='English IPA')
plt.text(x=35,y=135,s='Imperial Stout')
plt.text(x=23,y=10,s='Robust Porter')
plt.show()