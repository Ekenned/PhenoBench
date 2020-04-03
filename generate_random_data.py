# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:09:41 2020

@author: Eamonn Kennedy, Senior Research Scientist, TORCH lab, Univ. Utah, USA

Use Cholesky decomposition to generate multivariate correlated pseudo 
observations of some common TBI-associated variables for algorithm testing.

The simulation population statistics (means, standard deviations) can be 
defined in the included file: 'random_data_covariates.csv' The current 
parameters are loosely based on typical values found in the literature.

The code contains hard-coded variable curation/truncation, and is not designed 
for generic feature generation. It outputs 'gen_data.csv'.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm
from scipy.linalg import cholesky

# USER INPUTS:
seed = 4 # Hard coding of the random generating numpy seed (42).
n_samples = 1000 # Define how many pseudo observations to generate (1000).
output_filename = 'gen_data.csv' # Choose the output filename
input_filename = 'random_data_covariates.csv'
DPI = 100 # dots per inch for displayed figure

# Settings
mpl.rcParams['figure.dpi'] = DPI # Define output figure display/write DPI

# Load the multicovariates, means,std of the desired simulation output datra
covs = pd.read_csv(input_filename)

# Extract the covariance matrix and simulation means, stds, and freqs.
np.random.seed(seed=seed)
r = covs.values[2::,1::].astype(int) # Extract the matrix of covariates
mean_freq = covs.iloc[0].values[1::] # Extract example variable means
std = covs.iloc[1].values[1::] # Extract example variable standard deviations

# Extract variable names, properties, and indices
var_names = covs.columns[1::].values # Extract example variable names
n_vars = len(var_names)
var_inds = np.arange(n_vars) # Extract list of variable indices

# Generate normalized random variables
x = norm.rvs(size=(np.shape(r)[0], n_samples)) 

# Cholesky decomposition: Get the lower-triangular L satisfying A = LL* of cov
c = cholesky(r, lower=True)

# Convert the random variables to correlated variables.
y = np.dot(c, x)

# Generate a matrix of additive Gassian noise, scaled by noise_lvl
# noise_lvl = 0 # We could optionally add another layer of model noise here
# noise = noise_lvl*np.reshape(np.random.randn(n_samples*n_vars), np.shape(y))

# Rescale correlated variables according to the csv-file means and stds:
norm_means = np.mean(y,axis=1) # values from decomposition
norm_std = np.std(y,axis=1) # values from decomposition
std_ratio = std/norm_std # renormalization constants

for i in var_inds:
    #y[i,:] = (y[i,:] + noise[i,:])*std[i]/norm_std[i] + ... # additive noise
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
y[LOD,np.where(y[LOD]<0)[0]] = 0 # one second loss of consciousness min
y[LOD,np.where(y[LOR,:]==0)] = 0 # if no LOC reported, then duration = 0
y[NBI,np.where(y[NBI,:]<0)] = 0 # Negative values indicates no TBI events
y[NBI,:] = np.round(y[NBI,:]).astype(int) # Set as integer events
y[NBI,np.intersect1d(np.where(y[NBI]==0),np.where(y[LOR,:]>0))] = 1 
# if LOC reported, we would expect at least 1 TBI
y[LOD,:] = y[LOD,:]*(mean_freq[LOD]/np.mean(y[LOD,:]))
y[EPI,:] = 1*(y[EPI,:]<np.percentile(y[EPI,:],100*mean_freq[EPI])) # freq EPI

corr_matrix = np.corrcoef(y)
df = pd.DataFrame(data=np.transpose(y), index =  np.arange(n_samples))
df.columns = var_names

df = df.astype(int)

# Optionally plot the histogram of every variable
#for i in var_inds:
#    plt.hist(y[i,:])
#    plt.xlabel(var_names[i])
#    plt.ylabel('Count')
#    plt.show()

df.to_csv(output_filename,index=False)
print('Wrote pseudodata file:',output_filename)

# plot the correlation matrix of the output pseudodata
plt.imshow(corr_matrix,cmap='bwr',vmin=-1,vmax=1)
plt.title('Pseudodata correlation colormap')
plt.grid(False)
plt.yticks(var_inds,[i for i in var_names])
plt.xticks(var_inds,[i for i in var_names],rotation='vertical')
plt.colorbar(shrink = 0.5)
plt.xlim((-1,n_vars))
plt.ylim((-1,n_vars))
plt.show()

print('Means by feature:')
print(df.mean())

# Simulation validation checks vs csv file requests:
#print(df.corr())
#print(df.mean())
#print(df.std())
#print(df.head())

# Plot various projections of the samples.
#plt.plot(y[LOD],y[TRB],'.')
#plt.plot(y[TRA],y[TRB],'.')

#print('Example TBI dataset statistics:')
#print(np.transpose(covs.iloc[0:2,:])) # print pseudo means and stds
