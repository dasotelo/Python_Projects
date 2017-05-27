###################################
#
# Demonstration of K-Means clustering
# Author: Dave Sotelo (sotelo.d.a@gmail.com)
# Date: 20170528
#
###################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import sklearn.cluster as clu

# Read McDonald's menu
# Freely available and downloaded from kaggle.com
# https://www.kaggle.com/mcdonalds/nutrition-facts
ff_menu=pd.read_csv('menu.csv',low_memory=False)

# Select menu items along w/ sodium and calorie content
menu_edt=ff_menu.loc[:,['Item','Calories','Sodium']]

# Visualize the data - simple scatter plot
sbn.set_style('whitegrid')
plt.figure(figsize=(10,6))
plt.scatter(menu_edt['Sodium'],menu_edt['Calories'],linewidths=1.2,
    edgecolors='k')
plt.xlabel('Sodium(g)')
plt.ylabel('Calories')
plt.title('Fast Food Menu - Items by Sodium & Caloric Content',size=14)
plt.show()

# K-means w/ n=4 clusters
cluster=clu.KMeans(n_clusters=4)
menu_edt['cluster']=cluster.fit_predict(menu_edt.iloc[:,1:])
clu_col_map={0:'red',1:'blue',2:'green',3:'purple',4:'orange',
    5:'yellow',6:'gray',7:'pink'}
menu_edt['clust_col']=menu_edt['cluster'].map(clu_col_map)

# Visualize clustered results
plt.figure(figsize=(10,6))
plt.scatter(menu_edt['Sodium'],menu_edt['Calories'],linewidths=1.2,
    edgecolors='k',c=menu_edt['clust_col'])
plt.xlabel('Sodium(g)')
plt.ylabel('Calories')
plt.title('Fast Food Menu - K-Means Clustering Results',size=14)
plt.show()

# K-Means with 8 clusters, 6 runs of standard algorithm
cluster=clu.KMeans(n_clusters=8)
menu_edt['cluster_n1']=cluster.fit_predict(menu_edt.iloc[:,1:3])
menu_edt['clust_col_n1']=menu_edt['cluster_n1'].map(clu_col_map)
menu_edt['cluster_n2']=cluster.fit_predict(menu_edt.iloc[:,1:3])
menu_edt['clust_col_n2']=menu_edt['cluster_n2'].map(clu_col_map)
menu_edt['cluster_n3']=cluster.fit_predict(menu_edt.iloc[:,1:3])
menu_edt['clust_col_n3']=menu_edt['cluster_n3'].map(clu_col_map)
menu_edt['cluster_n4']=cluster.fit_predict(menu_edt.iloc[:,1:3])
menu_edt['clust_col_n4']=menu_edt['cluster_n4'].map(clu_col_map)
menu_edt['cluster_n5']=cluster.fit_predict(menu_edt.iloc[:,1:3])
menu_edt['clust_col_n5']=menu_edt['cluster_n5'].map(clu_col_map)
menu_edt['cluster_n6']=cluster.fit_predict(menu_edt.iloc[:,1:3])
menu_edt['clust_col_n6']=menu_edt['cluster_n6'].map(clu_col_map)

# 2x3 subplot showing cluster results
f,ax=plt.subplots(2,3,sharex=True,sharey=True)
f.set_size_inches(10,6)
f.suptitle('Fast Food Menu - Compare Standard K-Means')
ax[0,0].scatter(menu_edt['Sodium'],menu_edt['Calories'],
    linewidths=0.6,edgecolors='k',c=menu_edt['clust_col_n1'])
ax[0,1].scatter(menu_edt['Sodium'],menu_edt['Calories'],
    linewidths=0.6,edgecolors='k',c=menu_edt['clust_col_n2'])
ax[0,2].scatter(menu_edt['Sodium'],menu_edt['Calories'],
    linewidths=0.6,edgecolors='k',c=menu_edt['clust_col_n3'])
ax[1,0].scatter(menu_edt['Sodium'],menu_edt['Calories'],
    linewidths=0.6,edgecolors='k',c=menu_edt['clust_col_n4'])
ax[1,1].scatter(menu_edt['Sodium'],menu_edt['Calories'],
    linewidths=0.6,edgecolors='k',c=menu_edt['clust_col_n5'])
ax[1,2].scatter(menu_edt['Sodium'],menu_edt['Calories'],
    linewidths=0.6,edgecolors='k',c=menu_edt['clust_col_n6'])
f.text(x=0,y=0.5,s='Calories',rotation='vertical')
f.text(x=0.5,y=0,s='Sodium(g)',rotation='horizontal')
f.tight_layout(rect=[0,0,1,0.96])
plt.show()

# delete and re-initialize KMeans 
menu_edt=menu_edt.iloc[:,:5]
cluster=clu.KMeans(n_clusters=8,init='k-means++')

# K-Means with 8 clusters, 6 runs of with distant initial centroids
menu_edt['cluster_n1']=cluster.fit_predict(menu_edt.iloc[:,1:3])
menu_edt['clust_col_n1']=menu_edt['cluster_n1'].map(clu_col_map)
menu_edt['cluster_n2']=cluster.fit_predict(menu_edt.iloc[:,1:3])
menu_edt['clust_col_n2']=menu_edt['cluster_n2'].map(clu_col_map)
menu_edt['cluster_n3']=cluster.fit_predict(menu_edt.iloc[:,1:3])
menu_edt['clust_col_n3']=menu_edt['cluster_n3'].map(clu_col_map)
menu_edt['cluster_n4']=cluster.fit_predict(menu_edt.iloc[:,1:3])
menu_edt['clust_col_n4']=menu_edt['cluster_n4'].map(clu_col_map)
menu_edt['cluster_n5']=cluster.fit_predict(menu_edt.iloc[:,1:3])
menu_edt['clust_col_n5']=menu_edt['cluster_n5'].map(clu_col_map)
menu_edt['cluster_n6']=cluster.fit_predict(menu_edt.iloc[:,1:3])
menu_edt['clust_col_n6']=menu_edt['cluster_n6'].map(clu_col_map)

# 2x3 subplot showing cluster results
f,ax=plt.subplots(2,3,sharex=True,sharey=True)
f.set_size_inches(10,6)
f.suptitle('Fast Food Menu - Compare K-Means with Distant Initial Centroids')
ax[0,0].scatter(menu_edt['Sodium'],menu_edt['Calories'],
    linewidths=0.6,edgecolors='k',c=menu_edt['clust_col_n1'])
ax[0,1].scatter(menu_edt['Sodium'],menu_edt['Calories'],
    linewidths=0.6,edgecolors='k',c=menu_edt['clust_col_n2'])
ax[0,2].scatter(menu_edt['Sodium'],menu_edt['Calories'],
    linewidths=0.6,edgecolors='k',c=menu_edt['clust_col_n3'])
ax[1,0].scatter(menu_edt['Sodium'],menu_edt['Calories'],
    linewidths=0.6,edgecolors='k',c=menu_edt['clust_col_n4'])
ax[1,1].scatter(menu_edt['Sodium'],menu_edt['Calories'],
    linewidths=0.6,edgecolors='k',c=menu_edt['clust_col_n5'])
ax[1,2].scatter(menu_edt['Sodium'],menu_edt['Calories'],
    linewidths=0.6,edgecolors='k',c=menu_edt['clust_col_n6'])
f.text(x=0,y=0.5,s='Calories',rotation='vertical')
f.text(x=0.5,y=0,s='Sodium(g)',rotation='horizontal')
f.tight_layout(rect=[0,0,1,0.96])
plt.show()

# delete and re-initialize KMeans 
menu_edt=menu_edt.iloc[:,:5]
cluster=clu.KMeans(n_clusters=8,n_init=1)

# K-Means with 8 clusters, 6 runs, single centroid seed only
menu_edt['cluster_n1']=cluster.fit_predict(menu_edt.iloc[:,1:3])
menu_edt['clust_col_n1']=menu_edt['cluster_n1'].map(clu_col_map)
menu_edt['cluster_n2']=cluster.fit_predict(menu_edt.iloc[:,1:3])
menu_edt['clust_col_n2']=menu_edt['cluster_n2'].map(clu_col_map)
menu_edt['cluster_n3']=cluster.fit_predict(menu_edt.iloc[:,1:3])
menu_edt['clust_col_n3']=menu_edt['cluster_n3'].map(clu_col_map)
menu_edt['cluster_n4']=cluster.fit_predict(menu_edt.iloc[:,1:3])
menu_edt['clust_col_n4']=menu_edt['cluster_n4'].map(clu_col_map)
menu_edt['cluster_n5']=cluster.fit_predict(menu_edt.iloc[:,1:3])
menu_edt['clust_col_n5']=menu_edt['cluster_n5'].map(clu_col_map)
menu_edt['cluster_n6']=cluster.fit_predict(menu_edt.iloc[:,1:3])
menu_edt['clust_col_n6']=menu_edt['cluster_n6'].map(clu_col_map)

# 2x3 subplot showing cluster results
f,ax=plt.subplots(2,3,sharex=True,sharey=True)
f.set_size_inches(10,6)
f.suptitle('Fast Food Menu - Compare K-Means without Multiple Centroid Seeding')
ax[0,0].scatter(menu_edt['Sodium'],menu_edt['Calories'],
    linewidths=0.6,edgecolors='k',c=menu_edt['clust_col_n1'])
ax[0,1].scatter(menu_edt['Sodium'],menu_edt['Calories'],
    linewidths=0.6,edgecolors='k',c=menu_edt['clust_col_n2'])
ax[0,2].scatter(menu_edt['Sodium'],menu_edt['Calories'],
    linewidths=0.6,edgecolors='k',c=menu_edt['clust_col_n3'])
ax[1,0].scatter(menu_edt['Sodium'],menu_edt['Calories'],
    linewidths=0.6,edgecolors='k',c=menu_edt['clust_col_n4'])
ax[1,1].scatter(menu_edt['Sodium'],menu_edt['Calories'],
    linewidths=0.6,edgecolors='k',c=menu_edt['clust_col_n5'])
ax[1,2].scatter(menu_edt['Sodium'],menu_edt['Calories'],
    linewidths=0.6,edgecolors='k',c=menu_edt['clust_col_n6'])
f.text(x=0,y=0.5,s='Calories',rotation='vertical')
f.text(x=0.5,y=0,s='Sodium(g)',rotation='horizontal')
f.tight_layout(rect=[0,0,1,0.96])
plt.show()