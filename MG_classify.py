#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 07:18:25 2017

@author: sotelo
"""

import requests as req
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

# Retrieve female character page information, label 'female'
url_female='http://metalgear.wikia.com/api/v1/Articles/List/?limit=2000&category=Female'
fem_results=req.get(url_female)
fem_json=fem_results.json()
fem_df=pd.DataFrame(fem_json['items'])
fem_df['true_sex']='female'

# Retrieve male character page information, label 'male'
url_male='http://metalgear.wikia.com/api/v1/Articles/List/?limit=2000&category=Male'
male_results=req.get(url_male)
male_json=male_results.json()
male_df=pd.DataFrame(male_json['items'])
male_df['true_sex']='male'

# Combine two datasets, print summary counts
chars_df=fem_df
chars_df=chars_df.append(male_df)
print('Total number of character articles: ',len(chars_df.id))
print('   Number of female character articles: ',sum(chars_df.true_sex=='female'))
print('   Number of male character articles: ',sum(chars_df.true_sex=='male'))

# Split into 50/50 train/test datasets
chars_train,chars_test=train_test_split(chars_df,train_size=0.5)

# Retrieve article text
def retrieve(src,idn):
    cont_json=[]
    text_json=[]
    art_req=req.get(src+str(idn))
    art_json=art_req.json()
    sec_json=art_json['sections']
    for i in range(len(sec_json)):
        if sec_json[i]['content']:
            cont_json.append(sec_json[i]['content'])      
    for g in range(len(cont_json)):
        for h in range(len(cont_json[g])):
            if cont_json[g][h]['type']=='paragraph':
                text_json.append(cont_json[g][h]['text'])
    return ' '.join(text_json)

# Convert all words to lower case and eliminate punctuation
def fmt(txt):
    txt_file=txt.lower()
    txt_file=txt_file.replace('.','')
    txt_file=txt_file.replace(',','')
    txt_file=txt_file.replace('!','')
    txt_file=txt_file.replace('?','')
    txt_file=txt_file.replace(':','')
    txt_file=txt_file.replace("'",'')
    txt_file=txt_file.replace('"','')
    return re.split('\W+',txt_file)

# Pronoun list - this is the list of pronouns that will be scanned for
pronouns=['she','her','hers','herself','he','him','his','himself']
fpron_cnt=pd.DataFrame(data={'counts':0*8},index=pronouns)
mpron_cnt=pd.DataFrame(data={'counts':0*8},index=pronouns)

# Retrieve character article content and count pronouns for males/females
cnt=1
req_src='http://metalgear.wikia.com/api/v1/Articles/AsSimpleJson/?id='
for iden in chars_train['id']: 
    print(cnt)
    txt_concat=[]
    txt_concat=retrieve(req_src,iden)
    txt_concat=fmt(txt_concat)
    for word in txt_concat:
        if word in pronouns:
            true_sex=chars_train.loc[chars_train.id==iden,['true_sex']].values[0]
            if true_sex=='male':
                mpron_cnt.loc[word]=mpron_cnt.loc[word]+1  
            else:
                fpron_cnt.loc[word]=fpron_cnt.loc[word]+1
    cnt=cnt+1
  
# Calculate bayesian probabilities: conditional probabilities then priors.
# Add 0.0001 to force all conditional probabilities > 0.0.  This should create
# de minimus difference in results, while ensuring calculation integrity
tpron_cnt=fpron_cnt+mpron_cnt
mpron_pct=mpron_cnt/sum(mpron_cnt['counts'])+0.0001
fpron_pct=fpron_cnt/sum(fpron_cnt['counts'])+0.0001
tpron_pct=tpron_cnt/sum(tpron_cnt['counts'])+0.0001
pct_male=chars_train.loc[chars_train.true_sex=='male',['id']].count()/ \
    chars_train['id'].count()
pct_fem=chars_train.loc[chars_train.true_sex=='female',['id']].count()/ \
    chars_train['id'].count()
    
def predict(txt,mprior,fprior):
    mscore=0.0
    fscore=0.0
    for word in txt:
        if word in pronouns:
            fscore=fscore+np.log(fpron_pct.loc[word].values*fprior/ \
                tpron_pct.loc[word].values)
            mscore=mscore+np.log(mpron_pct.loc[word].values*mprior/ \
                tpron_pct.loc[word].values)
    return fscore,mscore

# Apply Bayesian probabilities to test dataset, generating male & female scores
# Use four different prior probabilities: (1) actual prior male/female
# (2) female+0.15/male-0.15 (3) 50/50 (4) female+0.45/male-0.45
cnt=1
fem_pred1=np.array([])
fem_pred2=np.array([])
fem_pred3=np.array([])
fem_pred4=np.array([])
for iden in chars_test['id']:
    print(cnt)
    txt_concat=[]
    txt_concat=retrieve(req_src,iden)
    txt_concat=fmt(txt_concat)
    fscore1,mscore1=predict(txt_concat,pct_male.values,pct_fem.values)
    fscore2,mscore2=predict(txt_concat,pct_male.values-0.15,pct_fem.values+0.15)
    fscore3,mscore3=predict(txt_concat,0.5,0.5)
    fscore4,mscore4=predict(txt_concat,pct_male.values-0.45,pct_fem.values+0.45)
    if mscore1+fscore1==0.0 or (np.exp(fscore1)+np.exp(mscore1))==0.:
        fem_pred1=np.concatenate((fem_pred1,[999.]))
    else:
        fem_pred1=np.concatenate((fem_pred1,np.round(np.exp(fscore1)/
            (np.exp(fscore1)+np.exp(mscore1)),2)))
    if mscore2+fscore2==0.0 or (np.exp(fscore2)+np.exp(mscore2))==0.:
        fem_pred2=np.concatenate((fem_pred2,[999.]))
    else:
        fem_pred2=np.concatenate((fem_pred2,np.round(np.exp(fscore2)/
            (np.exp(fscore2)+np.exp(mscore2)),2)))
    if mscore3+fscore3==0.0 or (np.exp(fscore3)+np.exp(mscore3))==0.:
        fem_pred3=np.concatenate((fem_pred3,[999.]))
    else:
        fem_pred3=np.concatenate((fem_pred3,np.round(np.exp(fscore3)/
            (np.exp(fscore3)+np.exp(mscore3)),2)))
    if mscore4+fscore4==0.0 or (np.exp(fscore4)+np.exp(mscore4))==0.:
        fem_pred4=np.concatenate((fem_pred4,[999.]))
    else:
        fem_pred4=np.concatenate((fem_pred4,np.round(np.exp(fscore4)/
            (np.exp(fscore4)+np.exp(mscore4)),2)))
    cnt=cnt+1
    
# Crosstab of true_sex on predicted_sex for all priors
ct1=pd.crosstab(chars_test['true_sex'],np.where(fem_pred1==999.,'none',
    np.where(fem_pred1>=0.5,'female','male')))
print('@@@@@@ Actual Priors: 80/20 male/female')
print(ct1)
ct2=pd.crosstab(chars_test['true_sex'],np.where(fem_pred2==999.,'none',
    np.where(fem_pred2>=0.5,'female','male')))
print('@@@@@@ Modified Priors: 65/35 male/female')
print(ct2)
ct3=pd.crosstab(chars_test['true_sex'],np.where(fem_pred3==999.,'none',
    np.where(fem_pred3>=0.5,'female','male')))
print('@@@@@@ Modified Priors: 50/50 male/female')
print(ct3)
ct4=pd.crosstab(chars_test['true_sex'],np.where(fem_pred4==999.,'none',
    np.where(fem_pred4>=0.5,'female','male')))
print('@@@@@@ Modified Priors: 35/65 male/female')
print(ct4)

# Generate and plot ROC curves
fpr1,tpr1,thr1=roc_curve(np.where(chars_test['true_sex']=='female',1.,0.),
    np.where(fem_pred1==999.,0,fem_pred1))
fpr2,tpr2,thr2=roc_curve(np.where(chars_test['true_sex']=='female',1.,0.),
    np.where(fem_pred2==999.,0,fem_pred2))
fpr3,tpr3,thr3=roc_curve(np.where(chars_test['true_sex']=='female',1.,0.),
    np.where(fem_pred3==999.,0,fem_pred3))
fpr4,tpr4,thr4=roc_curve(np.where(chars_test['true_sex']=='female',1.,0.),
    np.where(fem_pred4==999.,0,fem_pred4))
plt.figure(figsize=(8,6))
plt.plot(fpr1,tpr1,c='orange',linewidth=4,label='80/20 Priors')
plt.plot(fpr2,tpr2,c='green',linewidth=2,alpha=0.5,label='65/35 Priors')
plt.plot(fpr3,tpr3,c='blue',linewidth=2,alpha=0.5,label='50/50 Priors')
plt.plot(fpr4,tpr4,c='purple',linewidth=2,alpha=0.5,label='35/65 Priors')
plt.plot([0,1],[0,1],'k--',alpha=0.5)
plt.legend(loc=4,fontsize=12)
plt.title('Statistical Measures for Female Prediction: ROC Curve',size=14)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()