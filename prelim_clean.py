# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 04:09:43 2023

@author: vshas
"""
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

# from sklearn.metrics import explained_variance_score
# from sklearn.metrics import mean_absolute_error
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import RidgeCV
from sklearn.impute import KNNImputer
# from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from collections import Counter
import pickle as pkl
from funcs import *

dvbic = pd.read_excel("./DVBIC/PAI San Antonio dataset for phenotype analyses.xlsx")
hamilton = pd.read_excel("./Hamilton/HHS PAI Database Dec 2008.xlsx")

#%%
df1 = dvbic.copy()
df2 = hamilton.copy()


df1.rename(columns={'Age':'age',
                    'Gender':'sex',
                    'time_wk_DOE_DOI':'weeks_post_injury'
                    },inplace=True)
df1.replace({999: np.nan},inplace=True)
df1['weeks_post_injury'] = df1['weeks_post_injury'].fillna(df1['weeks_post_injury'].mean())
# df1['weeks_post_injury'] = df1['weeks_post_injury'].replace({999: pd.NA})
# Race and eth

df1['eth'] = np.where(df1['race'] == 1,1,0)
race_conv ={0 : 'White',
            1 : 'Multi',
            2:  'Black',
            3: 'Asian',
            4: 'Multi',
            5: 'Native American'
            }
df1['race_strings'] = df1['race'].replace(race_conv)
df1 = one_hot_encode(df1,'race_strings',True)

edu_outlier_ind = get_outliers(df1['edu']).index
df1.loc[edu_outlier_ind,'edu'] = pd.NA
prelim_edu = get_prelim_data(df1,'edu')
edu_nums = np.array(list(prelim_edu['value_counts'].keys()),dtype="int")
edu_weights = np.array(list(prelim_edu['value_counts'].values()),dtype="int")
edu_fill_list = np.random.choice(a = edu_nums,
                                 size = prelim_edu['null_count'],
                                 p = edu_weights/edu_weights.sum())
df1.loc[edu_outlier_ind,'edu'] = edu_fill_list
df1['edu_strings'] = df1['edu'].replace({8: "Early childhood education" ,
                                         10: "Primary education" ,
                                         11: "Lower secondary education",
                                         12: "Upper secondary education",
                                         13: "Post-secondary non-tertiary education",
                                         14: "Short-cycle tertiary education",
                                         15: "Bachelor's or equivalent",
                                         16: "Bachelor's or equivalent",
                                         17: "Master's or equivalent",
                                         18: "Doctorate or equivalent"})
df1 = eduINT_to_US(df1,'edu_strings')
df1['tbi'] = 1

# Getting values like 99,98. Setting them to NaN. Imputing all NaNs based on column distribution
marital_outliers = get_outliers(df1['marital']).index
df1.loc[marital_outliers,'marital'] = pd.NA
prelim_marital = get_prelim_data(df1,'marital')
marital_nums = np.array(list(prelim_marital['value_counts'].keys()),dtype="int")
marital_weights = np.array(list(prelim_marital['value_counts'].values()),dtype="int")
marital_fill_list = np.random.choice(a = marital_nums,
                                 size = prelim_marital['null_count'],
                                 p = marital_weights/marital_weights.sum())
df1.loc[marital_outliers,'marital'] = marital_fill_list
df1['married'] = np.where(df1['marital'] == 1,1,0)
#%%
df2.rename(columns = {'IDNumber':'ID',
                      'Sex':'sex',
                      'Age_at_Testing':'age',
                      'Highest_Education_Level':'edu',
                      'Relationship_status': 'marital'
                      },inplace=True)
df2['weeks_post_injury'] = df2['Years_Post_Injury']*52.142
# Sex
df2['sex'] -= 1
df2['tbi'] = 1

def GCS_severity(num):
    if num < 8: 
        return 3
    elif num >=8 and num <=12:
        return 2
    elif num >=13 and num <=15:
        return 1
    elif num > 15:
        return 1
df2['severity'] = df2['GCS'].map(GCS_severity)
df2['eth'] = 0
df2['race'] = np.random.choice(a = ["White","Black","Asian"] ,
                               size = len(df2),
                               p = [0.88,0.038,0.082])
df2 = one_hot_encode(df2, 'race', False)

sex_outlier_ind = get_outliers(df2['sex']).index
df2.loc[sex_outlier_ind,'sex'] = pd.NA
prelim_sex = get_prelim_data(df2,'sex')
null_indices = sex_outlier_ind.append(prelim_sex['null_indices'])
df2.loc[null_indices,'sex'] = 0

edu_null_ind = get_prelim_data(df2,'edu')['null_indices']
df2.loc[edu_null_ind,'edu'] = 12.0

# severity: GCS 13-15 and the rest 374 : Mild, 8-12 : Moderate, < 8: severe
def edu_str_conv(num):
    if num <= 5: 
        return "Early childhood education"
    elif num > 5 and num <=10:
        return "Primary education"
    elif num > 10 and num <=11:
        return "Lower secondary education"
    elif num > 11 and num <=12:
        return "Upper secondary education"
    elif num > 12 and num <=14:
        return "Post-secondary non-tertiary education"
    elif num > 14 and num <=15:
        return "Short-cycle tertiary education"
    elif num > 15 and num <=18:
        return "Bachelor's or equivalent"
    elif num > 18 and num <=20:
        return "Master's or equivalent"
    elif num > 20:
        return "Doctorate or equivalent"

df2['edu_strings'] = df2['edu'].map(edu_str_conv)
df2 = eduINT_to_US(df2,'edu_strings')
marital_null_indices = get_prelim_data(df2, 'marital')['null_indices']

prelim_marital = get_prelim_data(df2,'marital')
marital_nums = np.array(list(prelim_marital['value_counts'].keys()),dtype="int")
marital_weights = np.array(list(prelim_marital['value_counts'].values()),dtype="int")
marital_fill_list = np.random.choice(a = marital_nums,
                                 size = prelim_marital['null_count'],
                                 p = marital_weights/marital_weights.sum())
df2.loc[marital_null_indices,'marital'] = marital_fill_list
df2['married'] = np.where(df2['marital'] == 2,1,0)
#%%
# Hamilton has no race, eth column

df1 = df1[['ID', 'weeks_post_injury', 'age', 'sex','race','eth','edu','edu_strings', 
           'severity','tbi','married','Asian', 'Black', 'Multi',
           'Native American', 'White', 'Associates Degree',
           'Bachelors Degree', 'High School Diploma', 'No Diploma',
           'Post-graduate Degree', 'Some college', 
           'ICN', 'INF','NIM', 'PIM', 'SOM', 'ANX', 'ARD', 'DEP', 'MAN', 'PAR', 'SCZ',
           'BOR', 'ANT', 'ALC', 'DRG', 'AGG', 'SUI', 'STR', 'NON', 'RXR',
           'DOM', 'WRM', ]]
df2 = df2[['ID','weeks_post_injury', 'age','sex','race','eth', 'edu','edu_strings',
           'severity','tbi','married', "White","Black","Asian",
           'Associates Degree', 'Bachelors Degree',
            'High School Diploma', 'No Diploma', 'Post-graduate Degree',
            'Some college',
           'PAIICN1', 'PAIINF1', 'PAINIM1', 'PAIPIM1', 'PAISOM1',
           'PAIANX1', 'PAIARD1', 'PAIDEP1', 'PAIMAN1', 'PAIPAR1', 'PAISCZ1',
           'PAIBOR1', 'PAIANT1', 'PAIALC1', 'PAIDRG1',
           ]]

df2.rename(columns= {'PAIICN1': 'ICN', 
                     'PAIINF1': 'INF', 
                     'PAINIM1': 'NIM',
                     'PAIPIM1': 'PIM',
                     'PAISOM1': 'SOM',
                     'PAIANX1': 'ANX',
                     'PAIARD1': 'ARD',
                     'PAIDEP1': 'DEP',
                     'PAIMAN1': 'MAN',
                     'PAIPAR1': 'PAR',
                     'PAISCZ1': 'SCZ',
                     'PAIBOR1': 'BOR',
                     'PAIANT1': 'ANT', 
                     'PAIALC1': 'ALC', 
                     'PAIDRG1': 'DRG'},inplace=True)
df2['PAR'] = df2['PAR'].fillna(df2['PAR'].mean())
#%%
knn = KNNImputer(n_neighbors=3)
impute_cols = ['weeks_post_injury', 'age','sex', 'edu', 'eth',
               'Associates Degree', 'Bachelors Degree',
                'High School Diploma', 'No Diploma', 'Post-graduate Degree',
                'Some college','married', 'White','Black','Asian','severity',
               'ICN', 'INF','NIM', 'PIM', 'SOM', 'ANX', 'ARD', 'DEP', 'MAN', 'PAR', 'SCZ',
               'BOR', 'ANT', 'ALC', 'DRG', ]
                                   
df2.loc[:,impute_cols] = knn.fit_transform(np.array(df2[impute_cols]))

df1['site'] = "DVBIC"
df2['site'] = "Hamilton"

df1['military'] = 1
df2['military'] = 0

df_combined = pd.concat([df1,df2])

df_combined[['Multi','Native American']] = df_combined[['Multi','Native American']].fillna(0)

df_combined.to_csv("combined_dataset.csv",index=False)

#%%

############################ Tables code  ###################################
#%%
## Table 1
df = pd.read_csv("combined_dataset.csv")
df.drop(columns="Unnamed: 0",inplace=True)

#%%
indices = ['Sample size (n)','Mean years (std)','Male','Female','High School or less','Some College','BA/BSc or Grad Degree',
           'Black','White','Asian','Other','Hispanic/Latino','TBI','Mean weeks post injury (std)','TBI severity']
cols = ['DVBIC','Hamilton']
'Some College'

desc_df = pd.DataFrame(index = indices, columns= cols)

for site in cols:
    desc_df.loc['Sample size (n)',site] = str(len(df[df['site'] == site]))
    
    desc_df.loc['Mean years (std)',site] = str(round(df[df['site']==site]['age'].mean(),2)) + "(" + \
                                            str(round(df[df['site']==site]['age'].std(),1)) + ")"
    
    desc_df.loc['Male',site] = str(round((len(df[(df['site'] == site) & (df['sex'] == 0)])/len(df[df['site'] == site])) * 100,2)) + "%"
    desc_df.loc['Female',site] = str(round((len(df[(df['site'] == site) & (df['sex'] == 1)])/len(df[df['site'] == site])) * 100,2)) + "%"
    
    desc_df.loc['High School or less',site] = round((df[df['site']==site]['No Diploma'].mean() + \
                                               df[df['site']==site]['High School Diploma'].mean()) * 100,2)
                                               
    desc_df.loc['Some College',site] = round((df[df['site']==site]['Associates Degree'].mean() + \
                                               df[df['site']==site]['Some college'].mean()) * 100,2)
    desc_df.loc['BA/BSc or Grad Degree',site] = round((df[df['site']==site]['Bachelors Degree'].mean() + \
                                               df[df['site']==site]['Post-graduate Degree'].mean()) * 100,2)
    
    desc_df.loc['Black',site] = str(round(df[df['site']==site]['Black'].mean() * 100,2)) + "%"
    desc_df.loc['White',site] = str(round(df[df['site']==site]['White'].mean() * 100,2)) + "%"
    desc_df.loc['Asian',site] = str(round(df[df['site']==site]['Asian'].mean() * 100,2)) + "%"
    desc_df.loc['Other',site] = str(round((df[df['site']==site]['Multi'].mean() + \
                                 df[df['site']==site]['Native American'].mean()) * 100,2)) + "%"
    
    desc_df.loc['Hispanic/Latino',site] = str(round(df[df['site']==site]['eth'].mean() * 100,2)) + "%"
    
    desc_df.loc['TBI',site] =  str(round(df[df['site']==site]['tbi'].mean() * 100,2)) + "%"
    desc_df.loc['Mean weeks post injury (std)',site] =  str(round(df[df['site']==site]['weeks_post_injury'].mean(),2)) + "(" + \
                                                        str(round(df[df['site']==site]['weeks_post_injury'].std(),2)) + ")"
    desc_df.loc['TBI severity',site] = round(df[df['site']==site]['severity'].mean(),2)
    
    
    

#%%
## Table 2
cols = [ 'ICN', 'INF', 'NIM', 'PIM','SOM', 'ANX', 'ARD', 'DEP',
        'MAN', 'PAR', 'SCZ', 'BOR', 'ANT', 'ALC',
        'DRG', 'AGG', 'SUI', 'STR', 'NON', 'RXR', 'DOM', 'WRM']
mean_cols_df = pd.DataFrame(index=cols,columns=['total','DVBIC','Hamilton'])
for col in cols:
    mean_cols_df.loc[col,'total'] = round(df[col].mean(),1)
    mean_cols_df.loc[col,'DVBIC'] = round(df[df['site']=="DVBIC"][col].mean(),1)
    mean_cols_df.loc[col,'Hamilton'] = round(df[df['site']=="Hamilton"][col].mean(),1)
    
mean_cols_df.to_csv("mean_score_PAI.csv")

#%%

## OLS regression ##

# Age divisions
df['agelessthan35'] = 1*(df['age'] < 35)
df['age35and65'] = 1*((df['age'] >=35) & (df['age']<65))
df['agegreaterthan65'] = 1*(df['age'] > 65)

df['weeks_post_injury_last_yr'] = 1 * (df['weeks_post_injury'] < 52)
# Ceiling effect on weeks post injury
x_cols = ['weeks_post_injury_last_yr',
           'age35and65','agegreaterthan65',
          'sex', 'eth','severity',
          'Asian', 'Black', 'Multi','Native American',
          'Associates Degree', 'Bachelors Degree',
          'No Diploma', 'Post-graduate Degree','Some college','military']
cols = [ 'ICN', 'INF', 'NIM', 'PIM','SOM', 'ANX', 'ARD', 'DEP',
        'MAN', 'PAR', 'SCZ', 'BOR', 'ANT', 'ALC',
        'DRG',]
        # 'AGG', 'SUI', 'STR', 'NON', 'RXR', 'DOM', 'WRM']

X = df[x_cols]
X['const'] = 1


for col in cols: 
    y = df[col]
    out_df = results_summary_to_dataframe(sm.OLS(y.values,X).fit())
    print(y)
    print(out_df)
    




          
