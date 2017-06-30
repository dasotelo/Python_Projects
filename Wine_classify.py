###################################
### Include this header at start of all IDLE .py scripts
import sys
sys.path.append('C:\\Anaconda3\Lib\site-packages')

import matplotlib
matplotlib.use('TkAgg')
###################################

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Read in & munge wine information dataset.  US wines only
wine_df=pd.read_csv('winemag-data_first150k.csv')
wine_df=wine_df.loc[wine_df.country=='US',['points','price','region_1', \
    'variety','winery']]
wine_df=wine_df.dropna(axis=0,how='any')

# Map point values to categories
bin_map={
    100:'96-100',
    99:'96-100',
    98:'96-100',
    97:'96-100',
    96:'96-100',
    95:'91-95',
    94:'91-95',
    93:'91-95',
    92:'91-95',
    91:'91-95',
    90:'86-90',
    89:'86-90',
    88:'86-90',
    87:'86-90',
    86:'86-90',
    85:'81-85',
    84:'81-85',
    83:'81-85',
    82:'81-85',
    81:'81-85',
    80:'80 or less',
    79:'80 or less',
    78:'80 or less',
    77:'80 or less',
    76:'80 or less'}
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

# Train classification tree
x=wine_train.iloc[:,[0,2,3,4]]
y=wine_train['point_bins']
min_samp_split=2                           
clf=DecisionTreeClassifier(min_samples_split=min_samp_split,max_features=None)
clf=clf.fit(x,y)

# Prepare test data for classification tree
wine_test['regn_enc']=regn_lab.transform(wine_test.region_1)
wine_test['var_enc']=var_lab.transform(wine_test.variety)
wine_test['wnry_enc']=wnry_lab.transform(wine_test.winery)
wine_test=wine_test.drop(['region_1','variety','winery'],axis=1)

train_error=y==clf.predict(x)
test_error=wine_test['point_bins']==clf.predict(wine_test.iloc[:,[0,2,3,4]])
print('@@@@@@@@@@@@@@@@@@@@@@@@@')
print('CART w/ #leaf nodes = ',clf.tree_.node_count)                           
print('   training accuracy: ','{:.1%}'.format(sum(train_error)/len(
    train_error)))
print('   test accuracy: ','{:.1%}'.format(sum(test_error)/len(test_error)))






















