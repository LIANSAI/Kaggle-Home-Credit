# -*- coding: utf-8 -*-
"""
Created on Thu May 24 23:41:35 2018

@author: Allen
"""


import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
import gc
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

application_test = pd.read_csv('file:///D:/数据/Kaggle/HomeCredit/application_test.csv')
application_train = pd.read_csv('file:///D:/数据/Kaggle/HomeCredit/application_train.csv')
bureau = pd.read_csv('file:///D:/数据/Kaggle/HomeCredit/bureau.csv')
#bureau_balance = pd.read_csv('file:///D:/数据/Kaggle/HomeCredit/bureau_balance.csv')



######### Missing Data ##############
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


######### Feature Engineering for Application_train
application_test['TARGET']=-1
data = pd.concat([application_train,application_test])
data_bureau = pd.merge(data,bureau,on='SK_ID_CURR',how='left')


################## Bureau #################

#credit active
flist = list(data_bureau.CREDIT_ACTIVE.dropna().unique())
m = data_bureau[['SK_ID_CURR','CREDIT_ACTIVE']]
m['total_credit_count'] =1
m0 = m[['SK_ID_CURR','total_credit_count']]
m0 = m0.groupby('SK_ID_CURR').agg('sum').reset_index()

for f in flist:
    m1 = m[m.CREDIT_ACTIVE==f][['SK_ID_CURR']]
    m1['credit_rate_'+f]=1
    m1 = m1.groupby('SK_ID_CURR').agg('sum').reset_index()
    m0 = pd.merge(m0,m1,on='SK_ID_CURR',how='left')
    m0['credit_rate_'+f] = m0['credit_rate_'+f]/m0['total_credit_count']

credit = m0.copy()
credit = credit.fillna(0)  

#currency one hot
currency = data_bureau[['SK_ID_CURR','CREDIT_CURRENCY']].drop_duplicates()

#DAYS_CREDIT
days_credit = data_bureau[['SK_ID_CURR']].drop_duplicates()
m = data_bureau[['SK_ID_CURR','DAYS_CREDIT']]
m1 = m.groupby('SK_ID_CURR').agg('mean').reset_index()
#m2 = m.groupby('SK_ID_CURR').agg('std').reset_index()
days_credit = pd.merge(days_credit,m1,on='SK_ID_CURR',how='left')
#days_credit = pd.merge(days_credit,m2,on='SK_ID_CURR',how='left')
days_credit.fillna(0)

#CREDIT_DAY_OVERDUE
credit_day_overdue = data_bureau[['SK_ID_CURR']].drop_duplicates()
m = data_bureau[['SK_ID_CURR','CREDIT_DAY_OVERDUE']]
m1 = m.groupby('SK_ID_CURR').agg('mean').reset_index()
#m2 = m.groupby('SK_ID_CURR').agg('std').reset_index()
credit_day_overdue = pd.merge(credit_day_overdue,m1,on='SK_ID_CURR',how='left')
#credit_day_overdue = pd.merge(credit_day_overdue,m2,on='SK_ID_CURR',how='left')

#DAYS_CREDIT_ENDDATE
days_credit_enddate =  data_bureau[['SK_ID_CURR']].drop_duplicates()
m = data_bureau[['SK_ID_CURR','DAYS_CREDIT_ENDDATE']]
m1 = m.groupby('SK_ID_CURR').agg('mean').reset_index()
m2 = m.groupby('SK_ID_CURR').agg('std').reset_index()
days_credit_enddate = pd.merge(days_credit_enddate,m1,on='SK_ID_CURR',how='left')
days_credit_enddate = pd.merge(days_credit_enddate,m2,on='SK_ID_CURR',how='left')

#DAYS_ENDDATE_FACT
days_enddat_fact =  data_bureau[['SK_ID_CURR']].drop_duplicates()
m = data_bureau[['SK_ID_CURR','DAYS_ENDDATE_FACT']]
m1 = m.groupby('SK_ID_CURR').agg('mean').reset_index()
days_enddat_fact = pd.merge(days_enddat_fact,m1,on='SK_ID_CURR',how='left')

#AMT_CREDIT_MAX_OVERDUE
amt_credit_max_overdue =  data_bureau[['SK_ID_CURR']].drop_duplicates()
m = data_bureau[['SK_ID_CURR','AMT_CREDIT_MAX_OVERDUE']]
m1 = m.groupby('SK_ID_CURR').agg('mean').reset_index()
m2 = m.groupby('SK_ID_CURR').agg('std').reset_index()
amt_credit_max_overdue = pd.merge(amt_credit_max_overdue,m1,on='SK_ID_CURR',how='left')
amt_credit_max_overdue = pd.merge(amt_credit_max_overdue,m2,on='SK_ID_CURR',how='left')

#AMT_CREDIT_SUM 不显著
amt_credit_sum =  data_bureau[['SK_ID_CURR']].drop_duplicates()
m = data_bureau[['SK_ID_CURR','AMT_CREDIT_SUM']]
m1 = m.groupby('SK_ID_CURR').agg('mean').reset_index()
m2 = m.groupby('SK_ID_CURR').agg('std').reset_index()
amt_credit_sum = pd.merge(amt_credit_sum,m1,on='SK_ID_CURR',how='left')
amt_credit_sum = pd.merge(amt_credit_sum,m2,on='SK_ID_CURR',how='left')

#AMT_CREDIT_SUM_DEBT
amt_credit_sum_debt =  data_bureau[['SK_ID_CURR']].drop_duplicates()
m = data_bureau[['SK_ID_CURR','AMT_CREDIT_SUM_DEBT']]
m1 = m.groupby('SK_ID_CURR').agg('mean').reset_index()
m2 = m.groupby('SK_ID_CURR').agg('std').reset_index()
amt_credit_sum_debt = pd.merge(amt_credit_sum_debt,m1,on='SK_ID_CURR',how='left')
amt_credit_sum_debt = pd.merge(amt_credit_sum_debt,m2,on='SK_ID_CURR',how='left')

#AMT_CREDIT_SUM_LIMIT  
amt_credit_sum_limit =  data_bureau[['SK_ID_CURR']].drop_duplicates()
m = data_bureau[['SK_ID_CURR','AMT_CREDIT_SUM_LIMIT']]
m1 = m.groupby('SK_ID_CURR').agg('mean').reset_index()
m2 = m.groupby('SK_ID_CURR').agg('std').reset_index()
amt_credit_sum_limit = pd.merge(amt_credit_sum_limit,m1,on='SK_ID_CURR',how='left')
amt_credit_sum_limit = pd.merge(amt_credit_sum_limit,m2,on='SK_ID_CURR',how='left')

#AMT_CREDIT_SUM_OVERDUE  很显著
amt_credit_sum_overdue =  data_bureau[['SK_ID_CURR']].drop_duplicates()
m = data_bureau[['SK_ID_CURR','AMT_CREDIT_SUM_OVERDUE']]
m1 = m.groupby('SK_ID_CURR').agg('mean').reset_index()
m2 = m.groupby('SK_ID_CURR').agg('std').reset_index()
amt_credit_sum_overdue = pd.merge(amt_credit_sum_overdue,m1,on='SK_ID_CURR',how='left')
amt_credit_sum_overdue = pd.merge(amt_credit_sum_overdue,m2,on='SK_ID_CURR',how='left')

#CREDIT_TYPE
flist = list(data_bureau.CREDIT_TYPE.dropna().unique())
m =  data_bureau[['SK_ID_CURR','CREDIT_TYPE']]
m['total_credit_type_count'] =1
m0 = m[['SK_ID_CURR','total_credit_type_count']]
m0 = m0.groupby('SK_ID_CURR').agg('sum').reset_index()

for f in flist:
    m1 = m[m.CREDIT_TYPE==f][['SK_ID_CURR']]
    m1['credit_type_rate_'+f]=1
    m1 = m1.groupby('SK_ID_CURR').agg('sum').reset_index()
    m0 = pd.merge(m0,m1,on='SK_ID_CURR',how='left')
    m0['credit_type_rate_'+f] = m0['credit_type_rate_'+f]/m0['total_credit_type_count']

credit_type = m0.copy()
credit_type = credit.fillna(0)  


#DAYS_CREDIT_UPDATE
days_credit_update =  data_bureau[['SK_ID_CURR']].drop_duplicates()
m = data_bureau[['SK_ID_CURR','DAYS_CREDIT_UPDATE']]
m1 = m.groupby('SK_ID_CURR').agg('mean').reset_index()
days_credit_update = pd.merge(days_credit_update,m1,on='SK_ID_CURR',how='left')

csvlist = [credit,currency,days_credit,credit_day_overdue,days_credit_enddate,
           days_enddat_fact,amt_credit_max_overdue,amt_credit_sum,amt_credit_sum_debt,
           amt_credit_sum_limit,amt_credit_sum_overdue,credit_type,days_credit_update]

bureau_feature = data_bureau[['SK_ID_CURR']].drop_duplicates()
for csv in csvlist:
    bureau_feature = pd.merge(bureau_feature,csv,on='SK_ID_CURR',how='left')
    
bureau_feature.to_csv('bureau_feature.csv',index=False)

head = bureau_feature.head(50)