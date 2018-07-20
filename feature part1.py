# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 13:53:04 2018

@author: Allen
"""

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold

a=data.head(50000)

########################################### data ###########################################
def one_hot_encoder(df):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= True)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

application_test = pd.read_csv('file:///D:/数据/Kaggle/HomeCredit/application_test.csv')
application_train = pd.read_csv('file:///D:/数据/Kaggle/HomeCredit/application_train.csv')

data = application_train.append(application_test)

for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        data[bin_feature], uniques = pd.factorize(data[bin_feature])

data, cat_cols = one_hot_encoder(data) 
data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
data['DAYS_EMPLOYED_PERC'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
data['INCOME_CREDIT_PERC'] = data['AMT_INCOME_TOTAL'] / data['AMT_CREDIT']
data['INCOME_PER_PERSON'] = data['AMT_INCOME_TOTAL'] / data['CNT_FAM_MEMBERS']
data['ANNUITY_INCOME_PERC'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']

del application_train,application_test
gc.collect()

########################################### bureau ###########################################
bureau = pd.read_csv('file:///D:/数据/Kaggle/HomeCredit/bureau.csv')
bureau_balance = pd.read_csv('file:///D:/数据/Kaggle/HomeCredit/bureau_balance.csv')
bureau, bureau_cat = one_hot_encoder(bureau)
bureau_balance, bb_cat = one_hot_encoder(bureau_balance)

a=bureau.head(10000)

# Bureau balance: Perform aggregations and merge with bureau.csv
bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
for col in bb_cat:
    bb_aggregations[col] = ['mean']
    
bb_agg = bureau_balance.groupby('SK_ID_BUREAU').agg(bb_aggregations).reset_index()   
bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
bureau.drop('SK_ID_BUREAU', axis=1,inplace= True)

# Bureau and bureau_balance numeric features
num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
        'AMT_ANNUITY': ['max', 'mean'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }

cat_aggregations = {}
for cat in bureau_cat: cat_aggregations[cat] = ['mean']
for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations}).reset_index()
bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
bureau_agg.rename(index=str,columns = {'BURO_SK_ID_CURR_': 'SK_ID_CURR'},inplace=True)

# Bureau: ACTIVE credits - using only numerical aggregations
active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations).reset_index() 
active_agg.columns = pd.Index(['ACT_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
active_agg.rename(index=str,columns = {'ACT_SK_ID_CURR_': 'SK_ID_CURR'},inplace=True)
bureau_agg = pd.merge(bureau_agg,active_agg, how='left', on='SK_ID_CURR')
del active, active_agg
gc.collect

# Bureau: Closed credits - using only numerical aggregations
closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations).reset_index() 
closed_agg.columns = pd.Index(['CLS_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
closed_agg.rename(index=str,columns = {'CLS_SK_ID_CURR_': 'SK_ID_CURR'},inplace=True)
bureau_agg = pd.merge(bureau_agg,closed_agg, how='left', on='SK_ID_CURR')

bureau_agg.to_csv('bureau_agg.csv',index=False)

################ # Preprocess previous_applications #######################
prev = pd.read_csv('file:///D:/数据/Kaggle/HomeCredit/previous_application.csv')
prev, cat_cols = one_hot_encoder(prev)

a = prev.head(100000)

prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
# Add feature: value ask / value received percentage   
prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
cat_aggregations = {}
for cat in cat_cols:
    cat_aggregations[cat] = ['mean']

prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations}).reset_index()
prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
prev_agg.rename(index=str,columns = {'PREV_SK_ID_CURR_': 'SK_ID_CURR'},inplace=True)

# Previous Applications: Approved Applications - only numerical features
approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations).reset_index()
approved_agg.columns = pd.Index(['APR_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
approved_agg.rename(index=str,columns = {'APR_SK_ID_CURR_': 'SK_ID_CURR'},inplace=True)

prev_agg = pd.merge(prev_agg,approved_agg, how='left', on='SK_ID_CURR')

 # Previous Applications: Refused Applications - only numerical features
refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations).reset_index()
refused_agg.columns = pd.Index(['REF_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
refused_agg.rename(index=str,columns = {'REF_SK_ID_CURR_': 'SK_ID_CURR'},inplace=True)

prev_agg = pd.merge(prev_agg,refused_agg, how='left', on='SK_ID_CURR')
del refused, refused_agg, approved, approved_agg, prev
gc.collect()

prev_agg.to_csv('prev_agg.csv',index=False)



#################### Preprocess POS_CASH_balance #######################
pos_cash_balance = pd.read_csv('file:///D:/数据/Kaggle/HomeCredit/POS_CASH_balance.csv')

a= pos_cash_balance.head(10000)

pos_cash_balance['MONTHS_BALANCE'] = pos_cash_balance['MONTHS_BALANCE'].astype('int16')
pos_cash_balance['CNT_INSTALMENT'] = pos_cash_balance['CNT_INSTALMENT'].astype('float16')
pos_cash_balance['CNT_INSTALMENT_FUTURE'] = pos_cash_balance['CNT_INSTALMENT_FUTURE'].astype('float16')
pos_cash_balance['SK_DPD'] = pos_cash_balance['SK_DPD'].astype('int32')
pos_cash_balance['SK_DPD_DEF'] = pos_cash_balance['SK_DPD_DEF'].astype('int32')

pos, cat_cols = one_hot_encoder(pos_cash_balance)

aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
for cat in cat_cols:
        aggregations[cat] = ['mean']
    
pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations).reset_index()
pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
pos_agg.rename(index=str,columns = {'POS_SK_ID_CURR_': 'SK_ID_CURR'},inplace=True)
pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    
pos_agg.to_csv('prev_agg.csv',index=False)

#################### Preprocess Installment #######################
ins = pd.read_csv('file:///D:/数据/Kaggle/HomeCredit/installments_payments.csv')
ins, cat_cols = one_hot_encoder(ins)


ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']

 # Days past due and days before due (no negative values)
ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
for cat in cat_cols:
    aggregations[cat] = ['mean']
ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations).reset_index()

ins_agg.columns = pd.Index(['INS_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
ins_agg.rename(index=str,columns = {'INS_SK_ID_CURR_': 'SK_ID_CURR'},inplace=True)

ins_agg['INS_COUNT'] = ins.groupby('SK_ID_CURR').size()

ins_agg.to_csv('ins_agg.csv',index=False)


#################### Preprocess Credit card Balance #######################
cc = pd.read_csv('file:///D:/数据/Kaggle/HomeCredit/credit_card_balance.csv')
cc, cat_cols = one_hot_encoder(cc)
cc.drop('SK_ID_PREV',axis=1, inplace = True)
cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var']).reset_index()
cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
cc_agg.rename(index=str,columns = {'CC_SK_ID_CURR_': 'SK_ID_CURR'},inplace=True)
cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()

cc_agg.to_csv('cc_agg.csv',index=False)


