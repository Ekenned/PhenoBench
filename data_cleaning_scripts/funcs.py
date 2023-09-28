# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 06:33:10 2023

@author: vshas
"""

import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationLearn
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib as mpl
import os 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def unique_with_str(df,num=None):
    '''
    
    Returns dictionary of unique values for all columns. If num is given, also
    returns filtered dictionary whose values contain num
    
    '''
    cols_with_num = []
    dict_out = {}
    for col in df.columns.values:
        dict_out[col] = df[col].unique()
        if num in df[col].unique():
            cols_with_num.append(col)
    return dict_out,{key:dict_out[key] for key in cols_with_num} if num else None
    
def get_prelim_data(df,col):
    
    """
    
    Return dictionary of value counts, unique values and null count given column in dataframe
    
    """
    dict1 = {}
    dict1['value_counts'] = df[col].value_counts().to_dict()
    dict1['unique_values'] = df[col].unique()
    dict1['null_count'] = df[col].isnull().sum()
    dict1['null_indices'] = df[df[col].isnull()].index
    return dict1

def get_outliers(data, m=2):
    return data[abs(data - np.mean(data)) > m * np.std(data)]

def median_absolute_deviation(x):
    """
    Returns the median absolute deviation from the window's median
    :param x: Values in the window
    :return: MAD
    """
    return np.median(np.abs(x - np.median(x)))


def hampel(ts, window_size=5, n=3, imputation=False):

    """
    Median absolute deviation (MAD) outlier in Time Series
    :param ts: a pandas Series object representing the timeseries
    :param window_size: total window size will be computed as 2*window_size + 1
    :param n: threshold, default is 3 (Pearson's rule)
    :param imputation: If set to False, then the algorithm will be used for outlier detection.
        If set to True, then the algorithm will also imput the outliers with the rolling median.
    :return: Returns the outlier indices if imputation=False and the corrected timeseries if imputation=True
    """

    if type(ts) != pd.Series:
        raise ValueError("Timeserie object must be of tyme pandas.Series.")

    if type(window_size) != int:
        raise ValueError("Window size must be of type integer.")
    else:
        if window_size <= 0:
            raise ValueError("Window size must be more than 0.")

    if type(n) != int:
        raise ValueError("Window size must be of type integer.")
    else:
        if n < 0:
            raise ValueError("Window size must be equal or more than 0.")

    # Copy the Series object. This will be the cleaned timeserie
    ts_cleaned = ts.copy()

    # Constant scale factor, which depends on the distribution
    # In this case, we assume normal distribution
    k = 1.4826

    rolling_ts = ts_cleaned.rolling(window_size*2, center=True)
    rolling_median = rolling_ts.median().fillna(method='bfill').fillna(method='ffill')
    rolling_sigma = k*(rolling_ts.apply(median_absolute_deviation).fillna(method='bfill').fillna(method='ffill'))

    outlier_indices = list(
        np.array(np.where(np.abs(ts_cleaned - rolling_median) >= (n * rolling_sigma))).flatten())

    if imputation:
        ts_cleaned[outlier_indices] = rolling_median[outlier_indices]
        return ts_cleaned

    return outlier_indices

# Functions to help with analysis

# List of columns
def cols(df):
    """
    Prints list of columns in df

    Parameters
    ----------
    df : DataFrame

    """
    print(list(df.columns.values))

# Get dictionary of all null values in columns 
def check_null_cols(df):
    """
    Prints dictionary with key as column and value as number of null values
    
    Parameters
    ----------
    df : Dataframe

    """
    print({col:df[col].isnull().sum() for col in df.columns.values 
           if df[col].isnull().sum() > 0})
    
def print_null_cols(df, print_type = 0):
    """
    Prints dictionary with key as column and value as number of null values
    
    Parameters
    ----------
    df : Dataframe

    """
    
    dict_of_missing = {col:df[col].isnull().sum() for col in df.columns.values}
    
    if print_type == 'total_df':
        for i in zip(dict_of_missing.keys(),dict_of_missing.values()):
            print(i[0], ', # Null:', i[1],)
            
        N = len(df)
        for i in zip(dict_of_missing.keys(),dict_of_missing.values()):
            print(i[0], ', # filled:', N - i[1],'/',N)
    else:
        
        dict_cvlt = {col:df[df['instrument']=='cvlt'][col].isnull().sum() for col in df.columns.values}
        
        dict_hvlt = {col:df[df['instrument']=='hvlt'][col].isnull().sum() for col in df.columns.values}
        
        dict_ravlt = {col:df[df['instrument']=='ravlt'][col].isnull().sum() for col in df.columns.values}
        
        N = len(df)
        N_cvlt = len(df[df['instrument']=='cvlt'])
        N_ravlt = len(df[df['instrument']=='ravlt'])
        N_hvlt = len(df[df['instrument']=='hvlt'])
        
        for j,i in enumerate(zip(dict_of_missing.keys(),dict_of_missing.values())):
            if i[0][0:4] == 'hvlt':
                hvlt_missing = list(dict_hvlt.values())[j]
                print(i[0], ', # filled:', N_hvlt - hvlt_missing,'/',N_hvlt,',', str((N_hvlt - hvlt_missing)/N_hvlt)[0:4])
            elif i[0][0:4] == 'cvlt':
                cvlt_missing = list(dict_cvlt.values())[j]
                print(i[0], ', # filled:', N_cvlt - cvlt_missing,'/',N_cvlt,',', str((N_cvlt - cvlt_missing)/N_cvlt)[0:4])
            elif i[0][0:4] == 'ravl':
                ravlt_missing = list(dict_ravlt.values())[j]
                print(i[0], ', # filled:', N_ravlt - ravlt_missing,'/',N_ravlt,',', str((N_ravlt - ravlt_missing)/N_ravlt)[0:4])
            else:
                print(i[0], ', # filled:', N - i[1],'/',N,',', str((N - i[1])/N)[0:4])
        
    
    
    
    
# Get dictionary of unique values in columns
def unique_val_cols(df,cols=None):
    """
    Prints dictionary with key as column and value as unique values in column
    
    Parameters
    ----------
    df : Dataframe
    cols : list, optional
        The default is None.

    """
    column = cols or df.columns.values
    print({col:df[col].unique() for col in column})

# Assign all 0s to race and edu columns not present in df
def add_race_edu(df,cols=None):
    """
    Assign 0 to race and edu columns not present in df.

    Parameters
    ----------
    df : Dataframe
        DESCRIPTION.
    cols : list, optional
        The default is None.

    """
    col_list = ['No Diploma','High School Diploma','GED','Some college',
                     'Associates Degree', 'Bachelors Degree', 'Post-graduate Degree',
                          'White','Asian', 'Black','Multi','Alaskan','Hawaiian']
    cols = col_list or cols
    for col in cols:
        if col not in df.columns.values:
            df[col]=0

def add_country(df,country):
    """
    Add country column to df

    Parameters
    ----------
    df : Dataframe
        DESCRIPTION.
    country : string

    """
    
    df['country'] = country

def add_lang(df,lang):
    """
    Add language column to df

    Parameters
    ----------
    df : Dataframe
    
    lang : string

    """
    
    df['lang'] = lang
    
def add_visit(df,subject_id):
    """
    Add visit column to df using subject_id values
    
    Parameters
    ----------
    df : Dataframe
    
    subject_id : string

    """
    df['visit'] = [None]*len(df)
    d={}
    for i in df.index:
        guid = df.loc[i,subject_id]
        d[guid]= 1 if guid not in d else d[guid]+1
        df.loc[i,'visit'] = d[guid]

def col_limits(df,cols=None):
    """
    Prints a dictionary with key as column and value as [min,max] value in column
    
    Parameters
    ----------
    df : Dataframe

    cols : list

    """
    cols = cols or df.columns.values
    print({col:[min(df[col]),max(df[col])] for col in cols})

def one_hot_encode(df,col,drop):
    """
    One hot encode the column col and append it to dataframe. Also option to drop the column
    Parameters
    ----------
    df : DataFrame
    
    col : column string

    """
    df = pd.concat([df,pd.get_dummies(df[col])],axis=1)
    if drop==True:
        df.drop(columns=col,inplace=True)
    return df
    
def append_educ_years(df):
    """
    Add years of education column to df

    Parameters
    ----------
    df : DataFrame

    """
    d = {'No Diploma': 8,
         'High School Diploma':12,
         'GED': 11,
         'Some college': 13,
         'Associates Degree':14,
         'Bachelors Degree': 16,
         'Post-graduate Degree': 19}
    df['edu_yr']= [0]*len(df)
    for i in df.index:
        for edu in d.keys():
            if edu in df.columns.values and df.loc[i,edu]==1: # Extra check since GED doesn't exist in edu
                df.loc[i,'edu_yr']+=1*d[edu]

def raceINT_to_str(df,race):
    """
    Convert international "race" column values to strings
    
    Parameters
    ----------
    df : DataFrame
    
    race : string

    """
    raceINT_conv = {0:"Black African American",
                    1: "Black African",
                    2: "Black Afro Caribbean",
                    3: "South/Central American Indian",
                    4: "North American Indian",
                    5: "Alaskan Native",
                    6: "Inuit",
                    7: "Hispanic or Latino",
                    8: "South Asian",
                    9: "Far Eastern Asian",
                    10: "Western Asian",
                    11: "White North American",
                    12: "White South American",
                    13:  "White European",
                    14: "White Middle Eastern",
                    15: "White African",
                    16: "White Australian",
                    17: "Hawaiian",
                    18: "Pacific Islander",
                    19: "Aboriginal/Torres Strait Islander",
                    20: "Not reported",
                    21: "Other",
                    22: "Unknown"
                    }
    df[race] = df[race].replace(raceINT_conv)

# Converting raceUS values to strings
def raceUS_to_str(df,race):
    """
    Convert US "race" column values to strings
    
    Parameters
    ----------
    df : DataFrame
    
    race : string

    """
    raceUS_conv = {0:"White",1:"Black",2:"Asian",3:"Alaskan",4:"Hawaiian",5:"Multi"}
    df[race] = df[race].replace(raceUS_conv)

def total_sum_check(test,df):
    """
    Individual trial scores are added up to a total score column for tests.

    Parameters
    ----------
    test : string
        
    df : dataframe


    """
    if test=="cvlt":
        df['cvlt_imfr_t15_total'] = np.sum(df[['cvlt_imfr_t1_c','cvlt_imfr_t2_c',
                                               'cvlt_imfr_t3_c','cvlt_imfr_t4_c',
                                               'cvlt_imfr_t5_c']],axis=1)
    elif test=="ravlt":
        df['ravlt_imfr_t15_total'] = np.sum(df[['ravlt_imfr_t1_c','ravlt_imfr_t2_c',
                                               'ravlt_imfr_t3_c','ravlt_imfr_t4_c',
                                               'ravlt_imfr_t5_c']],axis=1)
    elif test=="hvlt":
        df['hvlt_imfr_t13_total'] = np.sum(df[['hvlt_imfr_t1_c','hvlt_imfr_t2_c',
                                               'hvlt_imfr_t3_c']],axis=1)

def replace_with_nan(df,col,values):
    """
    Replace the list of values with NaN in a column

    Parameters
    ----------
    df : dataframe
    
    values : list
        
    """
    df[col] = df[col].replace(values,np.nan)

def col_prob_fill(df,col,cumul_list,val_list):
    """
    Fill empty values in column using probabilities.

    Parameters
    ----------
    df : DataFrame
        
    col : string
    
    cumul_list : list
        List of probability distributions for race
    val_list : list
        List of values to be assigned for random values in specific ranges

    Returns
    ----------
    
    df : DataFrame
    
    """
    if col not in df.columns.values:
        df[col]=np.nan
    missing_index = df[df[col].isnull()].index
    cumul_prob = np.cumsum(np.array(cumul_list))
    for ind in missing_index:
        rand_val = np.random.uniform(0,sum(cumul_list))
        val = None
        for i in range(len(cumul_list)):
            if rand_val < cumul_prob[i]:
                val = val_list[i]
                break  
        # Get the highest value in cumul_list and assign val to element with same index in val_list
        # Since rand_val doesn't exceed bounds, below is not needed.
        # val = max(zip(cumul_list,val_list),key=lambda x:x[0])[1] if val==None else val
        # print(rand_val,val)
        df.loc[ind,col] = val
    return df
def eduINT_to_str(df,edu):
    """
    Converting international "edu" column (ISCED2011) values to strings
    
    Parameters
    ----------
    df : DataFrame
    
    edu : string

    """
    edu_conv = {0:"Early childhood education",
                1:"Primary education",
                2:"Lower secondary education",
                3:"Upper secondary education",
                4:"Post-secondary non-tertiary education",
                5:"Short-cycle tertiary education",
                6:"Bachelor's or equivalent",
                7:"Master's or equivalent",
                8:"Doctorate or equivalent"}
    df[edu] = df[edu].replace(edu_conv)

def eduINT_to_US(df,edu):
    """
    Convert ISCED2011 strings to US equivalent strings. 
    One hot encode the newly created column and drop it after.
    
    Parameters
    ----------
    df : DataFrame
    
    edu : string
    
    Returns
    ----------
    df : DataFrame
    
    """
    edu_INTtoUS = {"Early childhood education":'No Diploma',
                   "Primary education": 'No Diploma',
                   "Lower secondary education":'No Diploma',
                   "Upper secondary education":'High School Diploma',
                   "Post-secondary non-tertiary education":"Some college",
                   "Short-cycle tertiary education":"Associates Degree",
                   "Bachelor's or equivalent":"Bachelors Degree",
                   "Master's or equivalent":"Post-graduate Degree",
                   "Doctorate or equivalent": "Post-graduate Degree"}
    df['edu_convUS'] = df[edu].replace(edu_INTtoUS)
    df = pd.concat([df,pd.get_dummies(df['edu_convUS'])],axis=1)
    df.drop(columns=['edu_convUS'],inplace=True)
    return df

def make_numeric_col_from_categorical(df, 
                                      var = 'site', 
                                      add_col = True,
                                      verbose = True):
    
    # Make an integer series relating to a categorical list, example usage:
    #df, arr, key_used = make_numeric_col_from_categorical(df, var = 'site')
    
    categories = pd.Series(df[var].unique())
    
    array = np.zeros(len(df))
    
    for i,cat in enumerate(df[var]):
        
        array[i] = categories[categories==cat].index[0]
    
    array = array.astype(int)
    
    # optionally add to the array
    if add_col:
        df[var + '_num'] = array
        if verbose:
            print('-----------------------------------------------')
            print(var+'_num', 'has been added to the dataframe by encoding variable',var, 'as numeric')
            
    return df, array, categories

def harmonization_model(df,features,covariates):
    '''
    Takes in raw data frame (df) and harmonizes the features (features) while 
    preserving the covariates (covariates)
    '''
    my_data = np.array(df[features])
    
    df_covs = df[covariates].rename(columns={'site': 'SITE'})
    df_covs = df_covs.reset_index().drop(columns=['index'])
    
    model, data_adj = harmonizationLearn(my_data, df_covs)
    
    df_data_adj = pd.DataFrame(data = data_adj,columns=features)
    
    df_harm = df.copy().reset_index()
    
    df_harm = df_harm.drop(columns=['index','Unnamed: 0'])
    
    df_data_adj['subject_id'] = df_harm['subject_id'].copy()
    df_data_adj['site'] = df_harm['site'].copy()
    
    df_harm = df_harm.drop(columns=features + list(['Military_new'])).copy()
    df_harm = df_harm.merge(df_data_adj, on=['subject_id','site'], how = 'outer').copy()

    df_harm[features] += df[features].mean() - df_harm[features].mean()

    
    return model,data_adj,df_harm

def KDE_pre_post(PRE, 
                 POST,
                 NAME_OF_MEASURE,
                 XLIMITS,
                 YLIMITS,
                 KDE_THICKNESS,
                 LINE_ALPHA,
                 ):
    
    """
    KDE_pre_post(PRE = [0.703, 0.967, 0.714, 0.768, 0.116, 0.675, 0.618, 0.46, 0.119, 0.48],
                     POST = [0.782, 1.293, 0.903, 1.448, 1.333, 1.198, 1.479, 0.734, 1.307, 1.433],
                     NAME_OF_MEASURE = 'Connectivity',
                     XLIMITS = (-1,4),
                     YLIMITS = (-0.5,2),
                     KDE_THICKNESS = 0.5, #0-1
                     LINE_ALPHA=0.5)
    """

    ax = sns.distplot(PRE, fit_kws={"color":"gray"}, kde=True,
                      hist=None, label="Baseline",);
    ax = sns.distplot(POST, fit_kws={"color":"red"}, kde=True,
                      hist=None, label="Post-treatment");
    
    # Get the two lines from the axes to generate shading
    l1 = ax.lines[0]
    l2 = ax.lines[1]
    
    # Get the xy data from the lines so that we can shade
    x1 = l1.get_xydata()[:,0]
    y1 = l1.get_xydata()[:,1]
    x2 = l2.get_xydata()[:,0]
    y2 = l2.get_xydata()[:,1]
    
    plt.show()
    
    plt.fill_between(1-KDE_THICKNESS*y1,x1, color="gray", alpha=0.3, linewidth=0.0)
    plt.fill_between(2+KDE_THICKNESS*y2,x2, color="red", alpha=0.3, linewidth=0.0)
        
    plt.scatter(np.ones(len(PRE)),PRE, color='gray', s=11, alpha=0.95,zorder=3)
    plt.scatter(2*np.ones(len(POST)),POST, color='red', s=11, alpha=0.95,zorder=2)
    
    for i in zip(PRE,POST):
        plt.plot([1, 2], [i[0], i[1]], 
                 color='red',zorder=1, alpha=LINE_ALPHA, linewidth=1.5)
        
    left_PRE_median = 1 - KDE_THICKNESS*y1[closestVal(x1,np.median(PRE))[0]]
    right_POST_median = 2 + KDE_THICKNESS*y2[closestVal(x2,np.median(POST))[0]]
    
    plt.plot([left_PRE_median, 1], [np.median(PRE),np.median(PRE)],
             color = "gray", linestyle='dotted')
    
    plt.plot([2, right_POST_median], [np.median(POST),np.median(POST)],
             color = "red", linestyle='dotted')
    
    plt.plot(1-KDE_THICKNESS*y1,x1, color="gray", linewidth=1.5)
    plt.plot(2+KDE_THICKNESS*y2,x2, color="red", linewidth=1.5)
    
    # plt.grid(axis='x',zorder=0)
    plt.grid(None)
    plt.xticks([1,2],["Baseline", "Post-treatment"], rotation = 90)
    
    plt.xlim(XLIMITS)
    plt.ylim(YLIMITS)
    plt.ylabel(NAME_OF_MEASURE)
    plt.tick_params(axis='y', which='both', direction='in', length=4, left=True)
    sns.despine()
    
def closestVal(Vec,p):
    
    # find the index of the closest value to p contained in the vector 'Vec'
    
    tempVec = np.abs(Vec - p)
    minInd = np.where(tempVec == np.min(tempVec))[0][0]
    minVal = Vec[minInd]
    return minInd,minVal

def results_summary_to_dataframe(results):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame({"pvals":pvals,
                               "coeff":coeff,
                               "conf_lower":conf_lower,
                               "conf_higher":conf_higher
                                })

    #Reordering...
    results_df = results_df[["coeff","pvals","conf_lower","conf_higher"]]
    return results_df

def load_plotting_settings(plotsize = 'normal'):
    
    # Coarse recommender for DPI based on screen resolution
    # approx_dpi = 180
    # if GetSystemMetrics(0)>3000:   
    approx_dpi = 350
        
    # plot settings
    plt.rcParams["font.family"] = "arial"
    mpl.rcParams['figure.dpi'] = approx_dpi
    mpl.rcParams.update({'errorbar.capsize': 2})
    plt.set_cmap('viridis')
         
    increase_value = 0
    if plotsize == 'normal':
        increase_value = 0
    elif plotsize == 'big':
        increase_value = 1
    elif plotsize == 'small':
        increase_value = -1
    elif plotsize == 'tiny':
        increase_value = -2
    
    # Style options for figures
    sns.set_style("ticks")
    plt.style.use(['seaborn-whitegrid']) # ,'dark_background'])
    mpl.rcParams['figure.figsize']   = (4 + increase_value, 4  + increase_value)
    mpl.rcParams['axes.titlesize']   = 12 + increase_value
    mpl.rcParams['axes.labelsize']   = 12 + increase_value
    mpl.rcParams['lines.linewidth']  = 1
    mpl.rcParams['lines.markersize'] = 8
    mpl.rcParams['xtick.labelsize']  = 10 + increase_value
    mpl.rcParams['ytick.labelsize']  = 10 + increase_value
    mpl.rcParams['axes.linewidth'] = 1

def match_pairs(treated, untreated,):

    """
    "match_pairs" - a function for simple 1NN matching of a clinical/treated
    group to untreated/controls, without propensity scores.
    example: inds, distances, = match_pairs(treated, untreated,)
    """

    # Input handling, convert dataframes to numpy arrays
    if isinstance(treated, pd.DataFrame):
        treated = treated.values
    if isinstance(untreated, pd.DataFrame):
        untreated = untreated.values
    
    # fit standard scaler and overwrite arrays with scaled data
    scaler = StandardScaler()
    scaler.fit(treated)
    untreated = scaler.transform(untreated)
    treated = scaler.transform(treated)
        
    # fit untreated group nearest neighbors, get treated distances
    N = NearestNeighbors(
        n_neighbors=1, algorithm = 'ball_tree').fit(untreated)
    distances, indices = N.kneighbors(treated)
    
    # return the closest fitting untreated inds and distances
    indices = indices.reshape(indices.shape[0])
    return indices, distances

def find_unique_matches(control, clinical, 
                        num_iterations=20, 
                        rand_samples = 3000):
    
    # # Settings
    # np.random.seed = 42
    # N_clinical = 1000
    # N_control = 10000
    # num_iterations = 20
    # rand_samples = 3000

    # # Create simulated data
    # control = pd.DataFrame()
    # control['age'] = np.round(100*np.random.rand(N_control))
    # control['edu'] = np.round(100*np.random.rand(N_control))
    # control.set_index(2*np.arange(N_control), inplace = True) # creating awkward indices

    # clinical = pd.DataFrame()
    # clinical['age'] = np.round(100*np.random.rand(N_clinical))
    # clinical['edu'] = np.round(100*np.random.rand(N_clinical))
    # clinical.set_index(3*np.arange(N_clinical), inplace = True) # creating awkward indices

    # unique_inds = find_unique_matches(control, clinical, 
    #                         num_iterations = num_iterations, 
    #                         rand_samples = rand_samples)

    # plt.hist(control.loc[unique_inds].values - clinical.values) # works
    
    N_control = len(control)
    N_clinical = len(clinical)
    
    if rand_samples >= N_control:
        print('Error, the random subsample count should be smaller than N controls')
    
    match_matrix = np.zeros((N_clinical, num_iterations)).astype(int)
    
    ind_permute_matrix = np.zeros((rand_samples, num_iterations)).astype(int)
    
    control_inds = np.zeros((N_clinical, num_iterations)).astype(int)
    
    # Find num_iteration different matches per person, each subsampled
    for i in range(num_iterations):
        
        ind_permute_matrix[:,i] = np.random.permutation(np.arange(N_control))[0:rand_samples]
        
        match_matrix[:,i], distances = match_pairs(clinical, control.values[ind_permute_matrix[:,i],:])
        
        control_inds[:,i] = control.index[ind_permute_matrix[:,i][match_matrix[:,i]]]
        
        # control.loc[control_inds[:,i]].age.values - clinical.age.values # confirmed working
        
    # slice the other way and find unique values over iteration
    unique_inds = np.zeros(N_clinical).astype(int)
    
    checklist_of_inds = []
    
    for j in np.arange(N_clinical):
        
        if j == 0:
            
            unique_inds[0] = control_inds[j,0]
            
            checklist_of_inds.append(unique_inds[0])
            
        else:
            
            unique_inds[j] = next((x for x in control_inds[j,:] if (x not in checklist_of_inds)), None)
            
            checklist_of_inds.append(unique_inds[j])
            
    return unique_inds

