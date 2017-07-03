########################
#
# CART with Wine Review Data
# Author: D Sotelo
# Date: July 2, 2017
#
# Data available via Kaggle.com
# https://www.kaggle.com/zynicide/wine-reviews
#
########################

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Read in & munge wine information dataset.  US wines only
wine_df=pd.read_csv('winemag-data_first150k.csv')
wine_df=wine_df.loc[wine_df.country=='US',['points','price','region_1', \
    'variety','winery']]
wine_df=wine_df.dropna(axis=0,how='any')

# Hexbin of points on price
plt.figure(figsize=(7,4))
plt.hexbin(wine_df.price,wine_df.points,cmap='bone_r',gridsize=35,
    extent=(0,100,73,115))
plt.xlim(0,100)
plt.ylim(80,98)
plt.title('Point Score by Wine Price',size=14)
plt.ylabel('Points')
plt.xlabel('Wine Price')
plt.show()

# Violin plot of points on region - use regions with 50 or more wines only
vio_dfm=wine_df.loc[:,['region_1','points']].groupby('region_1').median()
vio_dfm.reset_index(inplace=True)
vio_df=wine_df.loc[:,['region_1','points']].groupby('region_1').count()
vio_df.reset_index(inplace=True)
vio_df=vio_df.merge(vio_dfm,'inner',left_on='region_1',right_on='region_1')
vio_df=vio_df.sort_values('points_y',ascending=False)
vio_df=vio_df.loc[vio_df.points_x>=50,:]
vio_df=vio_df.iloc[0:30,0]

vio_df2=[wine_df.loc[wine_df.region_1==j,['points']] for j in vio_df.values]

plt.figure(figsize=(6,11))
parts=plt.violinplot(vio_df2,vert=False,showextrema=False,showmedians=True)
for pc in parts['bodies']:
    pc.set_edgecolor('black')
    pc.set_alpha(0.3)
    pc.set_facecolor('green')
plt.yticks(np.arange(30,0,-1),vio_df.values)
plt.title('Distribution of Point Scores by Top Regions',size=14)
plt.ylabel('Top 30 Regions by Median Point Score, >50 Wines per Region')
plt.xlabel('Points')
plt.show()

# Map point values to categories
bin_map={
    100:'90+',
    99:'90+',
    98:'90+',
    97:'90+',
    96:'90+',
    95:'90+',
    94:'90+',
    93:'90+',
    92:'90+',
    91:'90+',
    90:'90+',
    89:'<90',
    88:'<90',
    87:'<90',
    86:'<90',
    85:'<90',
    84:'<90',
    83:'<90',
    82:'<90',
    81:'<90',
    80:'<90',
    79:'<90',
    78:'<90',
    77:'<90',
    76:'<90'}
wine_df['point_bins']=wine_df.points.map(bin_map)
wine_df.point_bins.unique() # Ensure no records are un-binned
wine_df=wine_df.drop('points',axis=1)

# Split into 50/50 train/test datasets
wine_train,wine_test=train_test_split(wine_df,test_size=0.15)

# Prepare train data for classification tree
regn_lab=LabelEncoder().fit(np.unique(wine_df.region_1.values))
var_lab=LabelEncoder().fit(np.unique(wine_df.variety.values))
wnry_lab=LabelEncoder().fit(np.unique(wine_df.winery.values))
wine_train['regn_enc']=regn_lab.transform(wine_train.region_1)
wine_train['var_enc']=var_lab.transform(wine_train.variety)
wine_train['wnry_enc']=wnry_lab.transform(wine_train.winery)
wine_train=wine_train.drop(['region_1','variety','winery'],axis=1)

# Prepare test data for classification tree
wine_test['regn_enc']=regn_lab.transform(wine_test.region_1)
wine_test['var_enc']=var_lab.transform(wine_test.variety)
wine_test['wnry_enc']=wnry_lab.transform(wine_test.winery)
wine_test=wine_test.drop(['region_1','variety','winery'],axis=1)

# Train classification tree
x=wine_train.loc[:,['price','regn_enc','var_enc','wnry_enc']]
y=wine_train['point_bins']
min_samp_split=2                           
clf=DecisionTreeClassifier(min_samples_split=min_samp_split,max_features=None)
clf=clf.fit(x,y)

# Report classification results.  training dataset first, then test.  
# BASELINE - all features, no tree termination criteria
train_error=y==clf.predict(x)
test_error=wine_test['point_bins']==clf.predict(wine_test.loc[:,['price', \
    'regn_enc','var_enc','wnry_enc']])
print('@@@@@@@@@@@@@@@@@@@@@@@@@')
print('CART w/ #leaf nodes = ',clf.tree_.node_count) 
print('   ',clf.n_features_,' features out of: 4 features')                          
print('   training accuracy: ','{:.1%}'.format(sum(train_error)/len(
    train_error)))
print('   test accuracy: ','{:.1%}'.format(sum(test_error)/len(test_error)))

# Report feature importance
print('@@@@@@@@@@@@@@@@@@@@@@@@@')
print('Feature importance for max_leaves model')
print(pd.DataFrame([clf.feature_importances_],columns=x.columns.values))

# Train classification tree - remove varietal information
x=wine_train.loc[:,['price','regn_enc','wnry_enc']]
y=wine_train['point_bins']
min_samp_split=2                           
clf=DecisionTreeClassifier(min_samples_split=min_samp_split,max_features=None)
clf=clf.fit(x,y)

# Report classification results.  train first, then test.  
# Varietal information removed - no tree termination criteria
train_error=y==clf.predict(x)
test_error=wine_test['point_bins']==clf.predict(wine_test.loc[:,['price', \
    'regn_enc','wnry_enc']])
print('@@@@@@@@@@@@@@@@@@@@@@@@@')
print('CART w/ #leaf nodes = ',clf.tree_.node_count) 
print('   Varietal information removed')                          
print('   training accuracy: ','{:.1%}'.format(sum(train_error)/len(
    train_error)))
print('   test accuracy: ','{:.1%}'.format(sum(test_error)/len(test_error)))

# Train classification tree - remove region information
x=wine_train.loc[:,['price','var_enc','wnry_enc']]
y=wine_train['point_bins']
min_samp_split=2                           
clf=DecisionTreeClassifier(min_samples_split=min_samp_split,max_features=None)
clf=clf.fit(x,y)

# Report classification results.  train first, then test.  
# Region information removed - no tree termination criteria
train_error=y==clf.predict(x)
test_error=wine_test['point_bins']==clf.predict(wine_test.loc[:,['price', \
    'var_enc','wnry_enc']])
print('@@@@@@@@@@@@@@@@@@@@@@@@@')
print('CART w/ #leaf nodes = ',clf.tree_.node_count) 
print('   Region information removed')                          
print('   training accuracy: ','{:.1%}'.format(sum(train_error)/len(
    train_error)))
print('   test accuracy: ','{:.1%}'.format(sum(test_error)/len(test_error)))

# Train classification tree - remove varietal & region information
x=wine_train.loc[:,['price','wnry_enc']]
y=wine_train['point_bins']
min_samp_split=2                           
clf=DecisionTreeClassifier(min_samples_split=min_samp_split,max_features=None)
clf=clf.fit(x,y)

# Report classification results.  train first, then test.  
# Varietal & region information removed - no tree termination criteria
train_error=y==clf.predict(x)
test_error=wine_test['point_bins']==clf.predict(wine_test.loc[:,['price', \
    'wnry_enc']])
print('@@@@@@@@@@@@@@@@@@@@@@@@@')
print('CART w/ #leaf nodes = ',clf.tree_.node_count) 
print('   Varietal & Region information removed')                          
print('   training accuracy: ','{:.1%}'.format(sum(train_error)/len(
    train_error)))
print('   test accuracy: ','{:.1%}'.format(sum(test_error)/len(test_error)))

# Re-train with 4 features
x=wine_train.loc[:,['price','regn_enc','var_enc','wnry_enc']]
y=wine_train['point_bins']
min_samp_split=2                           
clf=DecisionTreeClassifier(min_samples_split=min_samp_split,max_features=None)
clf=clf.fit(x,y)

# Create dataframe object to record results of tree termination tests
# using model trained with all 4 features
results4=pd.DataFrame([],columns=list(['n_leaves','train_acc','test_acc']))
bench_nodes=clf.tree_.node_count
for i in np.arange(bench_nodes,500,-100):
    print(i)
    min_samp_split=2                           
    clf=DecisionTreeClassifier(min_samples_split=min_samp_split,
        max_features=None,max_leaf_nodes=i)
    clf=clf.fit(x,y)
    train_error=y==clf.predict(x)
    test_error=wine_test['point_bins']==clf.predict(wine_test.loc[:,['price', \
    'regn_enc','var_enc','wnry_enc']])
    run_rslt=pd.DataFrame([[i,sum(train_error)/len(train_error),
        sum(test_error)/len(test_error)]],
        columns=list(['n_leaves','train_acc','test_acc']))
    results4=results4.append(run_rslt)
    
for i in np.arange(500,0,-10):
    print(i)
    min_samp_split=2                           
    clf=DecisionTreeClassifier(min_samples_split=min_samp_split,
        max_features=None,max_leaf_nodes=i)
    clf=clf.fit(x,y)
    train_error=y==clf.predict(x)
    test_error=wine_test['point_bins']==clf.predict(wine_test.loc[:,['price', \
    'regn_enc','var_enc','wnry_enc']])
    run_rslt=pd.DataFrame([[i,sum(train_error)/len(train_error),
        sum(test_error)/len(test_error)]],
        columns=list(['n_leaves','train_acc','test_acc']))
    results4=results4.append(run_rslt)
    
# Re-train with 2 features
x=wine_train.loc[:,['price','wnry_enc']]
y=wine_train['point_bins']
min_samp_split=2                           
clf=DecisionTreeClassifier(min_samples_split=min_samp_split,max_features=None)
clf=clf.fit(x,y)

# Create dataframe object to record results of tree termination tests
# using model trained with price and winery only
results2=pd.DataFrame([],columns=list(['n_leaves','train_acc','test_acc']))
bench_nodes=clf.tree_.node_count
for i in np.arange(bench_nodes,500,-100):
    print(i)
    min_samp_split=2                           
    clf=DecisionTreeClassifier(min_samples_split=min_samp_split,
        max_features=None,max_leaf_nodes=i)
    clf=clf.fit(x,y)
    train_error=y==clf.predict(x)
    test_error=wine_test['point_bins']==clf.predict(wine_test.loc[:,['price', \
    'wnry_enc']])
    run_rslt=pd.DataFrame([[i,sum(train_error)/len(train_error),
        sum(test_error)/len(test_error)]],
        columns=list(['n_leaves','train_acc','test_acc']))
    results2=results2.append(run_rslt)
    
for i in np.arange(500,0,-10):
    print(i)
    min_samp_split=2                           
    clf=DecisionTreeClassifier(min_samples_split=min_samp_split,
        max_features=None,max_leaf_nodes=i)
    clf=clf.fit(x,y)
    train_error=y==clf.predict(x)
    test_error=wine_test['point_bins']==clf.predict(wine_test.loc[:,['price', \
    'wnry_enc']])
    run_rslt=pd.DataFrame([[i,sum(train_error)/len(train_error),
        sum(test_error)/len(test_error)]],
        columns=list(['n_leaves','train_acc','test_acc']))
    results2=results2.append(run_rslt)
    
# Plot resulting train & test accuracy rates for max_leaves runs
plt.figure(figsize=(8,5))
plt.plot(results4.n_leaves,results4.train_acc,linewidth=2
    ,label='train data, 4 features')
plt.plot(results4.n_leaves,results4.test_acc,linewidth=2
    ,label='test data, 4 features')
plt.plot(results2.n_leaves,results2.train_acc,linewidth=2,linestyle='--'
    ,label='train data, 2 features')
plt.plot(results2.n_leaves,results2.test_acc,linewidth=2,linestyle='--'
    ,label='test data, 2 features')
plt.title('Classification Accuracy for n Leaf Models',size=14)
plt.ylabel('Classification Accuracy')
plt.xlabel('n_max_leaves')
plt.legend()
