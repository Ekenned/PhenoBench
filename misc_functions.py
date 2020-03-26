# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:14:02 2019

@author: Eamonn Kennedy
"""

import numpy as np
import platform
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.neighbors import KernelDensity

def logistic(x):
    
    # logit function 
    
    return 1 / (1 + np.exp(-x))

def hist_rugplot(y,n_bins=100):
    
    # Plot kernel density, histogram points, and density-rugplot of a distribution
    
    # Constants for plot production and calculations: 
    #############################################################################
    
    min_col_offset = 0.98 # set not-quite-white to the minimum color
    rgb_col = np.array([0.8,0.6,0.3]) # default coloring
    rugplot_lims = np.array([-0.2,-0.05]) # set the y-limits of the rugplot
    line_width = 6 # default line thickness
    fig_size = (12,8) # default plot size
    cut = 95 # 95% kde height scaling cutoff
    bin_spread = np.round(n_bins/5).astype(int) # kde spread width in bin units
    opacity = 0.3 # kde colour transparency
    scatter_size = 50 # size of circles tha represent histogram
    ebins = 5
    
    if len(y)<=200: # if the sample count is low, modify settings
        
        n_bins = 20
        line_width = 4*line_width
        bin_spread = 5
        ebins = 2
        
    # Distribution calculations
    #############################################################################

    count,pos = np.histogram(y,bins=n_bins)
    edge_size = ebins*(pos[1] - pos[0])
    kde = KernelDensity(bandwidth=bin_spread/n_bins, kernel='gaussian')
    x_d = np.linspace(pos[0] - edge_size, pos[-1] + edge_size, n_bins*5)
    kde.fit(y[:, None])
    logprob = kde.score_samples(x_d[:, None])
    prob = np.exp(logprob)
    height_norm_factor = np.percentile(count,cut)
    grad_color = count/np.max(count)
    kde_height = height_norm_factor*prob/np.percentile(prob,cut)
    
    # Plotting
    #############################################################################

    plt.figure(figsize=fig_size)
    
    plt.fill_between(x_d, kde_height, alpha=opacity) # plot kde
    
    plt.scatter(pos[1::],count,s=scatter_size,c='c') # plot histogram line
    
    for i in range(len(count)):
        
        # plot color rugplot
        
        color_array = min_col_offset - grad_color[i]*rgb_col
        plt.plot([pos[1 + i],pos[1 + i]],rugplot_lims*height_norm_factor,
                 linewidth=line_width,
                 solid_capstyle='butt',
                 color = color_array)
        
    # set limits and axes style
    plt.xlim((pos[0] - edge_size,pos[-1] + edge_size)) 
    plt.xlabel('Value')
    plt.ylabel('Count')
    sns.despine()
    plt.show()
    
    # example usage
    #y = np.concatenate((np.random.normal(0, 1, 300),np.random.normal(4, 1, 200)),0)
    #color_rugplot(y)

def load_plotting_settings(plotsize = 'normal'):
    
    increase_value = 0
    if plotsize == 'normal':
        increase_value = 0
    elif plotsize == 'big':
        increase_value = 6
    elif plotsize == 'small':
        increase_value = -4
    elif plotsize == 'tiny':
        increase_value = -8
    
    # Style options for figures
    plt.style.use(['seaborn-whitegrid']) # ,'dark_background'])
    a = platform.uname()
    # print(a)
    if a[1] == 'XPS15': # settings for graphs on 4K XPS15 screen only
        print('Using 4k figure settings...')
        mpl.rcParams["font.family"] = "Arial"
        mpl.rcParams['figure.figsize']   = (20+ + increase_value, 20 + increase_value)
        mpl.rcParams['axes.titlesize']   = 30 + increase_value
        mpl.rcParams['axes.labelsize']   = 30 + increase_value
        mpl.rcParams['lines.linewidth']  = 2
        mpl.rcParams['lines.markersize'] = 20
        mpl.rcParams['xtick.labelsize']  = 30 + increase_value
        mpl.rcParams['ytick.labelsize']  = 30 + increase_value
        mpl.rcParams['axes.linewidth'] = 3
    else:
        mpl.rcParams["font.family"] = "Arial"
        mpl.rcParams['figure.figsize']   = (10 + increase_value, 10  + increase_value)
        mpl.rcParams['axes.titlesize']   = 20 + increase_value
        mpl.rcParams['axes.labelsize']   = 20 + increase_value
        mpl.rcParams['lines.linewidth']  = 2
        mpl.rcParams['lines.markersize'] = 8
        mpl.rcParams['xtick.labelsize']  = 16 + increase_value
        mpl.rcParams['ytick.labelsize']  = 16 + increase_value
        mpl.rcParams['axes.linewidth'] = 3
        
    print('Modules and settings loaded')
    
def repeat_many_image_plot(image_dictionary,N):
    
    # Subplot the contents of a dictionary of images up to N dictionary elements
    # Example usage: 
    #from sklearn.datasets import load_digits
    #digits = load_digits()
    #image_dictionary = digits.images[0:100] # 100 instance dictionary
    #repeat_many_image_plot(image_dictionary,N=64) # plot first 64 instances
    
    if N>0:
        
        image_dictionary = image_dictionary[0:N]
        
    n = len(image_dictionary)
    n_rows = np.ceil(np.sqrt(n)).astype(int)
    n_cols = np.ceil(n/n_rows).astype(int)
    
    fig, ax_array = plt.subplots(n_rows, n_cols)
    axes = ax_array.flatten()
    for i, ax in enumerate(axes):
        
        if i<n:
            ax.imshow(image_dictionary[i], cmap='gray_r')
    plt.setp(axes, xticks=[], yticks=[], frame_on=False)
    plt.tight_layout(h_pad=0.5, w_pad=0.1)
    
def scatterplot_2D(arr,targets,save):

    # Color scatter plot an Y x 2 matrix with Y target labels of N classes
    
    plt.scatter(arr[:,0],arr[:,1],c=targets)
    plt.scatter(arr[:,0],arr[:,1],c=(
                    targets),facecolors='none',edgecolors='k',cmap = 'gist_ncar')
    if save == 1:
        plt.savefig('color_scatterplot.eps', format='eps')
        
    plt.colorbar()    
    plt.show()
    
def closestVal(Vec,p):
    
    # find the index of the closest value to p contained in the vector 'Vec'
    
    tempVec = np.abs(Vec - p)
    minInd = np.where(tempVec == np.min(tempVec))[0][0]
    minVal = Vec[minInd]
    return minInd,minVal

def test_train_inds(n_obs,train_split):
    
    # split indices randomly into test and train groups
    
    if (train_split != 0) and (train_split != 1):
        
        n_train = round(train_split*n_obs)
        train_inds = np.random.choice(n_obs,n_train,replace=False).astype(int)
        test_inds = np.setdiff1d(range(n_obs),train_inds).astype(int)
        
    else:
        
        # otherwise it is supervised
        train_inds = range(n_obs)
        test_inds = np.array([])

    return train_inds,test_inds

def logistic(x):
    
    # logit function 
    
    return 1 / (1 + np.exp(-x))

def closestVal(Vec,p):
    
    # find the index of the closest value to p contained in the vector 'Vec'
    
    tempVec = np.abs(Vec - p)
    minInd = np.where(tempVec == np.min(tempVec))[0][0]
    minVal = Vec[minInd]
    return minInd,minVal

def normalize_vector(vec):
    
    # Normalize a 1D vector to values between 0 and 1
    
    if (vec.ndim != 1):
        
        print('Could not normalize vector, input must be a 1D numpy array')
        norm_vec = 0
        
    else:
        
        norm_vec = ( vec - np.min(vec) ) / ( np.max(vec) - np.min(vec) )
    
    return norm_vec

def gini(array):
    
    """Calculate the Gini coefficient of a numpy array."""
    
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def plot_loglogfreq(vector):

    plt.plot(np.sort(vector)[::-1])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('# Features')
    plt.ylabel('Value')

def load_previous_analysis(fname = 'analysis'): 
    
    with open(fname,'rb') as analysis_write:  
        
        pickle.load(analysis_write)
    
def log_ax_lims(X):
    
    # Establish safe axis limits in log scale given a vector X of points
    
    min_ax = 10**np.floor(np.log10(min(X)))
    max_ax = 10**np.ceil(np.log10(max(X)))
    
    return (min_ax,max_ax)

def savegraph():
    
    plt.savefig('output_graph.eps', format='eps')

def rolling_window(a, window=3):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def train_threshold(pred_vals,labels,n_steps = 50):            
    
    # determine best threshold for a 1D vector to match its labels/pred labels
    
    if len(labels) != len(pred_vals):
        
        print('Could not train threshold, vectors are of different size')
        
    elif ( (labels.ndim != 1) or  (pred_vals.ndim) != 1):
        
        print('Could not train threshold, inputs must both be 1D numpy arrays')
        
    elif (True in [( (i != 0) and (i != 1) ) for i in labels]):
    
        print('Could not train threshold, non-binary values in labels')
        
    else:

        pred_vals = normalize_vector(pred_vals)
        
        n_error = len(labels) # initialize error rate at 100%
        
        for j in range(n_steps):
            
            pred_labels = pred_vals>(j/n_steps)
            error_sweep = np.sum(np.abs(pred_labels - labels)>0)
            
            if error_sweep<=n_error: # if # errors at j is less than prior vals
                
                n_error = error_sweep # set a new minimum error
                threshold = j/n_steps # and hold on to that specific threshold
                
        return threshold
    
def slow_PCC_matrix_with_vector(matrix,vector):
    
    # Return a slow but low memory correlation calculation of every row of a mat
    # with a 1D test vector. Matrix = (# observations, # features)
    
    n = np.shape(matrix)
    PCCs = np.zeros(n[1])
    for i in range(n[1]):
        
        PCCs[i] = np.corrcoef(matrix[:,i],vector)[1][0]
        
    return PCCs

def PCC_feature_matrix_with_label_matrix(feature_matrix,label_matrix):
    
    # Given N observations, each exhibiting F feature vals arising from L labels,
    # Calculate every feature correlation with every separate label set.
    
    
    n_feats = np.shape(feature_matrix)[1]
    n_obs = np.shape(feature_matrix)[0]
    n_label_sets = np.shape(label_matrix)[1]
    PCCs = np.zeros((n_label_sets,n_feats))
        
    for i in range(n_label_sets):
    
        PCCs[i,:] = slow_PCC_matrix_with_vector(feature_matrix,label_matrix[:,i])
        print('Label set #',i,' complete...')
              
    return PCCs
    
def PCC_matrix_with_vector(matrix,vector):
    
    # Return a fast (vectorized) correlation calculation of every row of a matrix
    # with a 1D test vector. Matrix = (# features, # observations)
    n = np.shape(matrix)
    vector_matrix = np.reshape(np.tile(vector,n[0]),(n[0],n[1]))
    
    z = np.corrcoef(matrix,vector_matrix)
    
    PCCs = np.zeros(n[0])
    for i in range(n[0]):
        PCCs[i] = z[-i-1,i]
        
    return PCCs

def define_message_for_plate_0155():
    
    # Define and return the encryption key for plate 0155
    
    message = np.array(
          [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0,
           0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0,
           0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0,
           1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
           1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1,
           1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1,
           0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0,
           0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
           1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1,
           0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1,
           1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1,
           0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0,
           0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0,
           1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
           0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0,
           1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    
    return message