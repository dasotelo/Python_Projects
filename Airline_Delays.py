#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 09:52:24 2017

@author: sotelo
######
2015 Flight Delays and Cancellations
######
Data set available through Kaggle.com, as well as directly from US
Department of Transportation 
######
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import calendar as cal

sns.set(style='white',palette='BuGn_d')

# Read flight delay information files from .csv
airpt = pd.read_csv('airports.csv',low_memory=False)
airln = pd.read_csv('airlines.csv',low_memory=False)
flt = pd.read_csv('flights.csv',low_memory=False)
airpt_5d = pd.read_csv('64516557_T_MASTER_CORD.csv',low_memory=False)

# Munge & merge data to support data analysis
flt_edt = flt.loc[:,['YEAR','MONTH','AIRLINE','ORIGIN_AIRPORT',
    'DESTINATION_AIRPORT','SCHEDULED_DEPARTURE','DEPARTURE_DELAY',
    'SCHEDULED_TIME','ELAPSED_TIME','AIR_TIME','DISTANCE','SCHEDULED_ARRIVAL',
    'ARRIVAL_DELAY','DIVERTED','CANCELLED']]
del flt

airln=airln.rename(columns={'AIRLINE':'AIRLINE_NM','IATA_CODE':'AIRLINE'})
flt_mrgpre = pd.merge(flt_edt,airln,on='AIRLINE',how='inner')
del flt_edt

flt_mrgpre_oct=flt_mrgpre.loc[flt_mrgpre['MONTH']==10,:]
flt_mrgpre=flt_mrgpre.loc[flt_mrgpre['MONTH']!=10,:]
airpt_5d=airpt_5d.iloc[:,0:2]
airpt_5d['AIRPORT_ID']=airpt_5d['AIRPORT_ID'].astype('str')
airpt_5d=airpt_5d.drop_duplicates(keep='first')

airpt_5d=airpt_5d.rename(columns={'AIRPORT':'AIRPORT_3DIG_O',
    'AIRPORT_ID':'ORIGIN_AIRPORT'})
flt_mrgpre_oct=pd.merge(flt_mrgpre_oct,airpt_5d,how='left',on='ORIGIN_AIRPORT')
flt_mrgpre_oct=flt_mrgpre_oct.drop('ORIGIN_AIRPORT',1)
flt_mrgpre_oct=flt_mrgpre_oct.rename(
    columns={'AIRPORT_3DIG_O':'ORIGIN_AIRPORT'})

airpt_5d=airpt_5d.rename(columns={'AIRPORT_3DIG_O':'AIRPORT_3DIG_D',
    'ORIGIN_AIRPORT':'DESTINATION_AIRPORT'})
flt_mrgpre_oct=pd.merge(flt_mrgpre_oct,airpt_5d,how='left',
    on='DESTINATION_AIRPORT')
flt_mrgpre_oct=flt_mrgpre_oct.drop('DESTINATION_AIRPORT',1)
flt_mrgpre_oct=flt_mrgpre_oct.rename(
    columns={'AIRPORT_3DIG_D':'DESTINATION_AIRPORT'})

flt_mrgpre=flt_mrgpre.append(flt_mrgpre_oct)

airpt=airpt.rename(columns={'IATA_CODE':'ORIGIN_AIRPORT',
    'AIRPORT':'OR_AIRPORT_NM','CITY':'OR_AIRPORT_CITY','STATE':'OR_AIRPORT_ST',
    'COUNTRY':'OR_AIRPORT_CNTRY','LATITUDE':'OR_AIRPORT_LAT',
    'LONGITUDE':'OR_AIRPORT_LONG'})
flt_mrgpre1 = pd.merge(flt_mrgpre,airpt,on='ORIGIN_AIRPORT',how='inner')
del flt_mrgpre

airpt=airpt.rename(columns={'ORIGIN_AIRPORT':'DESTINATION_AIRPORT',
    'OR_AIRPORT_NM':'DEST_AIRPORT_NM','OR_AIRPORT_CITY':'DEST_AIRPORT_CITY',
    'OR_AIRPORT_ST':'DEST_AIRPORT_ST','OR_AIRPORT_CNTRY':'DEST_AIRPORT_CNTRY',
    'OR_AIRPORT_LAT':'DEST_AIRPORT_LAT','OR_AIRPORT_LONG':'DEST_AIRPORT_LONG'})
flt_mrg = pd.merge(flt_mrgpre1,airpt,on='DESTINATION_AIRPORT',how='inner')
del flt_mrgpre1

# Hexbin plot, Average Speed ~ Departure Delay, by Month
airspd_mos = flt_mrg.loc[(flt_mrg['DIVERTED']==0)&(flt_mrg['CANCELLED']==0)
    &(flt_mrg['DEPARTURE_DELAY']<150)&(flt_mrg['DISTANCE']>=300),
    ['MONTH','DEPARTURE_DELAY','AIR_TIME','DISTANCE']]
airspd_mos['AVG_SPD']=airspd_mos['DISTANCE']/(airspd_mos['AIR_TIME']/60)
plt.close('all')
f,ax=plt.subplots(3,4,sharex=True,sharey=True)
f.set_size_inches(14,10.5)
f.suptitle('Airspeed on Departure Delay',x=0.18,y=0.98,weight='bold',
    size=18)
f.text(x=0,y=0.5,s='Airspeed, MPH',rotation='vertical')
f.text(x=0.5,y=0,s='Departure Delay, Minutes',rotation='horizontal')
for c in range(len(ax)):
    for d in range(len(ax[0])):
        ax[c,d].hexbin(x=airspd_mos.loc[airspd_mos['MONTH']==((c*4)+d+1),
            ['DEPARTURE_DELAY']],
            y=airspd_mos.loc[airspd_mos['MONTH']==((c*4)+d+1),
            ['AVG_SPD']],gridsize=(30,15),extent=(-20,50,300,600),edgecolors='w',
            cmap=cm.ocean_r)
        ax[c,d].set_title(cal.month_name[((c*4)+d+1)])
f.tight_layout(rect=[0,0,1,0.96])
plt.show()

# Hexbin plot, Average Speed ~ Departure Delay, by Month, Delay > 30
airspd_mos2 = flt_mrg.loc[(flt_mrg['DIVERTED']==0)&(flt_mrg['CANCELLED']==0)
    &(flt_mrg['DEPARTURE_DELAY']<150)&(flt_mrg['DISTANCE']>=300)
    &(flt_mrg['DEPARTURE_DELAY']>30),
    ['MONTH','DEPARTURE_DELAY','AIR_TIME','DISTANCE']]
airspd_mos2['AVG_SPD']=airspd_mos2['DISTANCE']/(airspd_mos2['AIR_TIME']/60)
plt.close('all')
f,ax=plt.subplots(3,4,sharex=True,sharey=True)
f.set_size_inches(14,10.5)
f.suptitle('Airspeed on Departure Delay, Delay > 30',x=0.18,y=0.98,
    weight='bold',size=18)
f.text(x=0,y=0.5,s='Airspeed, MPH',rotation='vertical')
f.text(x=0.5,y=0,s='Departure Delay, Minutes',rotation='horizontal')
for c in range(len(ax)):
    for d in range(len(ax[0])):
        ax[c,d].hexbin(x=airspd_mos2.loc[airspd_mos2['MONTH']==((c*4)+d+1),
            ['DEPARTURE_DELAY']],
            y=airspd_mos2.loc[airspd_mos2['MONTH']==((c*4)+d+1),
            ['AVG_SPD']],gridsize=(30,15),extent=(30,100,300,600),edgecolors='w',
            cmap=cm.cubehelix_r)
        ax[c,d].set_title(cal.month_name[((c*4)+d+1)])
f.tight_layout(rect=[0,0,1,0.96])  
plt.show()  

# Hexbin plot, Average Speed ~ Departure Delay, by Month, Delay > 30 w fit
airspd_mos2 = flt_mrg.loc[(flt_mrg['DIVERTED']==0)&(flt_mrg['CANCELLED']==0)
    &(flt_mrg['DEPARTURE_DELAY']<150)&(flt_mrg['DISTANCE']>=300)
    &(flt_mrg['DEPARTURE_DELAY']>30),
    ['MONTH','DEPARTURE_DELAY','AIR_TIME','DISTANCE']]
airspd_mos2['AVG_SPD']=airspd_mos2['DISTANCE']/(airspd_mos2['AIR_TIME']/60)
plt.close('all')
f,ax=plt.subplots(3,4,sharex=True,sharey=True)
f.set_size_inches(14,10.5)
f.suptitle('Airspeed on Departure Delay, Delay > 30, Linear Best Fit',x=0.18,y=0.98,
    weight='bold',size=18)
f.text(x=0,y=0.5,s='Airspeed, MPH',rotation='vertical')
f.text(x=0.5,y=0,s='Departure Delay, Minutes',rotation='horizontal')
for c in range(len(ax)):
    for d in range(len(ax[0])):
        x=np.squeeze(np.array(airspd_mos2.loc[airspd_mos2['MONTH']==((c*4)+d+1),
            ['DEPARTURE_DELAY']]))
        y=np.squeeze(np.array(airspd_mos2.loc[airspd_mos2['MONTH']==((c*4)+d+1),
            ['AVG_SPD']]))
        fit=np.polyfit(x,y,deg=1)
        ax[c,d].plot(x,fit[0]*x+fit[1],'k--')
        ax[c,d].hexbin(x=airspd_mos2.loc[airspd_mos2['MONTH']==((c*4)+d+1),
            ['DEPARTURE_DELAY']],
            y=airspd_mos2.loc[airspd_mos2['MONTH']==((c*4)+d+1),
            ['AVG_SPD']],gridsize=(30,15),extent=(30,100,300,600),edgecolors='w',
            cmap=cm.cubehelix_r)
        ax[c,d].set_xlim([30,100])
        ax[c,d].set_ylim([300,600])
        ax[c,d].set_title(cal.month_name[((c*4)+d+1)])
f.tight_layout(rect=[0,0,1,0.96])
plt.show()    
    
