# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:09:41 2020

@author: Eamonn Kennedy, Senior Research Scientist, TORCH lab, Univ. Utah, USA

Use Cholesky decomposition to generate multivariate correlated pseudo 
observations of some common TBI-associated variables, for algorithm testing.

The simulation population statistics (means, standard deviations) can be 
defined in the included file: 'random_data_covariates.csv' 
Current parameters are loosely based on typical values found in the literature.

The code contains hard-coded variable curation/truncation, and is not designed 
for generic feature generation. It outputs 'gen_data.csv'

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm
from scipy.linalg import cholesky

mpl.rcParams['figure.dpi'] = 200 # Define output figure display/write DPI

np.random.seed(seed=42) # Hard code the random seed seed

n_samples = 1000 # How many pseudo observations to generate

# Load desired means, stds, and covariances for the pseudo data variables
covs = pd.read_csv('random_data_covariates.csv')

# Extract the covariance matrix and simulation means, stds, and freqs.
r = covs.values[2::,1::].astype(int) # Extract the matrix of covariates
var_names = covs.columns[1::].values # Extract example variable names
var_inds = np.arange(len(var_names)) # Extract list of variable indices
mean_freq = covs.iloc[0].values[1::] # Extract example variable means
std = covs.iloc[1].values[1::] # Extract example variable standard deviations

# Generate normalized random variables
x = norm.rvs(size=(np.shape(r)[0], n_samples)) 

# Cholesky decomposition: Get the lower-triangular L satisfying A = LL* of cov
c = cholesky(r, lower=True)

# Convert the random variables to correlated variables.
y = np.dot(c, x) 

# Rescale correlated variables according to the csv-file means and stds:
norm_means = np.mean(y,axis=1) # values from decomposition
norm_std = np.std(y,axis=1) # values from decomposition
std_ratio = std/norm_std # renormalization constants
for i in var_inds:
    y[i,:] = y[i,:]*std[i]/norm_std[i] + mean_freq[i] - norm_means[i]

# Irreversible operations, data cleaning and scaling to 'true' variable ranges:
# First defining the indices in the csv for each variable
AGE = np.where(var_names == 'AGE')[0][0]
EDU = np.where(var_names == 'EDUCATION')[0][0]
FEM = np.where(var_names == 'FEMALE')[0][0]
RAC = np.where(var_names == 'RACE')[0][0]
TSD = np.where(var_names == 'PCLM_SCORE')[0][0]
LOR = np.where(var_names == 'LOC_REPORTED')[0][0]
LOD = np.where(var_names == 'LOC_DUR')[0][0]
NBI = np.where(var_names == 'N_TBI')[0][0]
EPI = np.where(var_names == 'EPILEPSY')[0][0]
TRA = np.where(var_names == 'TRAILS_A')[0][0]
TRB = np.where(var_names == 'TRAILS_B')[0][0]

# Set the AGE and years of EDUCATION vars to be rounded to the nearest year
y[AGE,:] = np.round(y[AGE,:]).astype(int)
y[EDU,:] = np.round(y[EDU,:]).astype(int)

# Set the TRAILS vars to be rounded to the nearest second
y[TRA,:] = np.round(y[TRA,:]).astype(int)
y[TRB,:] = np.round(y[TRB,:]).astype(int)

# Set FEMALE = 1 to be observd at the frequency defined in the csv file
y[FEM,:] = 1*(y[FEM,:]<np.percentile(y[FEM,:],100*mean_freq[FEM]))

# set RACE category variable to be rounded to the nearest integer
y[RAC,:] = np.round(y[RAC,:] - np.min(y[RAC,:])).astype(int)

# PCLM /C: Set minimum observation to 17 and max to 85, the full test range
y[TSD,np.where(y[TSD]<17)[0]] = 17
y[TSD,np.where(y[TSD]>85)[0]] = 85
y[TSD,:] = np.round(y[TSD,:]).astype(int)

# Set LOC REP and EPI as binary vars, and argmin(LOC_DUR) [minutes] to 1 sec.
y[LOR,:] = 1*(y[LOR,:]>mean_freq[LOR])
y[LOD,np.where(y[LOD]<0)[0]] = 1/60 # one second loss of consciousness min
y[LOD,np.where(y[LOR,:]==0)] = 0 # if no LOC reported, then duration = 0
y[NBI,np.where(y[NBI,:]<0)] = 0 # Negative values indicates no TBI events
y[NBI,:] = np.ceil(y[NBI,:]).astype(int) # Set as integer events

y[EPI,:] = 1*(y[EPI,:]<np.percentile(y[EPI,:],100*mean_freq[EPI])) # freq EPI

# Optionally plot the histogram of every variable
for i in var_inds:
    plt.hist(y[i,:])
    plt.xlabel(var_names[i])
    plt.ylabel('Count')
    plt.show()

corr_matrix = np.corrcoef(y)
df = pd.DataFrame(data=np.transpose(y), index =  np.arange(n_samples))
df.columns = var_names

df = df.astype(int)

# plot the correlation matrix of the output pseudodata
plt.imshow(corr_matrix,cmap='bwr',vmin=-1,vmax=1)
plt.title('Pseudodata correlation colormap')
plt.grid(False)
plt.yticks(var_inds,[i for i in var_names])
plt.xticks(var_inds,[i for i in var_names],rotation='vertical')
plt.colorbar(shrink = 0.5)
plt.xlim((-1,len(var_names)))
plt.ylim((-1,len(var_names)))
plt.show()

df.corr()
df.mean()
df.std()
df.head()


df.to_csv(r'gen_data.csv')

# Plot various projections of the samples.
#plt.plot(y[LOD],y[TRB],'.')
#plt.plot(y[TRA],y[TRB],'.')

#print('Example TBI dataset statistics:')
#print(np.transpose(covs.iloc[0:2,:])) # print pseudo means and stds
