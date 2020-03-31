# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:12:43 2020

@author: Eamonn
"""
import numpy as np
import pandas as pd
import sklearn
import os
import seaborn
import umap
import hdbscan
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyclustertend import hopkins

class PhenoBench():
    
    """
    A class that loads a phenotype TBI dataset and performs 
    standardized phenotyping of the data.
    
    Includes:
        
        Core features: 
            
        - A data dictionary loader which tolerates csv and sas file read-in
        
        Phenotyping options leveraging:
        - Unsupervised dimensionality reduction with Gen. low rank models
        - Unsupervised clustering with K-means
        - Unsupervised dimensionality reduction with UMAP
        - Unsupervised clustering with HDBSCAN
        
        Secondary features:
        
        - Methods for quick, easy imputation
        - Methods for categorical data encoding and string handling
        - Methods for estimating the optimal number of phenotypes
        - Methods for autogeneration of figures and tables
        
        Example usage 1: 
        # Phenotyping with recommended settings and file writing
        #----------------------------------------
            
        Benchmark = PhenoBench() 
        Benchmark.load(file = r'.../Data/gen_data.csv')
        Benchmark.run()
        Benchmark.output()
        
        Example usage 2:
        # Phenotyping with custom settings and no file writing
        #----------------------------------------
            
        Benchmark = PhenoBench() 
        Benchmark.load(file = r'.../Data/TBI_data.sas7bdat')
        Benchmark.run(dim_reduce = 'UMAP',
                      cluster = 'HDBSCAN',
                      reduce_nn =  30,
                      cluster_nn = 20,
                     )
        Benchmark.output(write_settings = 1,
                         write_figures = 0,
                         write_tables = 1)
        
    """
    
    def __init__(self):
        
        # Assign default settings
        defaults = {'norm_vars':0, # optionally standard score all variables
                    'dim_reduce':'UMAP', # or GLRM, or PCA
                    'cluster':'HDBSCAN', # or KMEANS
                    'KMEANS_clusters':3, # Only valid for cluster=KMEANS
                    'metric':'correlation', # UMAP settings
                    'reduce_nn':40, # UMAP settings
                    'min_d':0.1, # UMAP settings
                    'min_c':20, # HDBSCAN settings
                    'cluster_nn':30, # HDBSCAN settings
                    'write_settings':1, # output toggle
                    'write_figures':1, # output toggle
                    'write_tables':1, # output toggle
                    'rand_state':0 # default state for random seeds
                    }
        
        # Inherit defaults to class
        self.norm_vars = defaults['norm_vars'] 
        self.dim_reduce = defaults['dim_reduce']
        self.cluster = defaults['cluster']
        self.KMEANS_clusters = defaults['KMEANS_clusters']
        self.min_d = defaults['min_d']
        self.min_c = defaults['min_c']
        self.reduce_nn = defaults['reduce_nn']
        self.cluster_nn = defaults['cluster_nn']
        self.metric = defaults['metric']
        self.write_settings = defaults['write_settings']
        self.write_figures= defaults['write_figures']
        self.write_tables = defaults['write_tables']
        self.rand_state = defaults['rand_state']
        self.settings = defaults
        
    def print_settings(self):
        
        [print(i[0],':',i[1]) for i in zip(
                self.settings.keys(),self.settings.values())][0]
        
    def load(self,file):
        
        self.matrix = pd.read_csv(file)
        # Check if data has nans or strings, perform imputation, 
        self.validate_data()
    
    def validate_data(self):
        0
        # Check if data has nans or strings, perform imputation, 
        
    def hopkins_test(self):
        
        hopkins_val = hopkins(
                self.matrix.values,np.shape(self.matrix.values)[0])
        
        print('Hopkins (>0.5 implies no clusters): ')
        print('%.4f' % hopkins_val)
        
    def run(self):
        
        if self.norm_vars == 0:
            print("Not normalizing features...")
        
        self.perform_UMAP_embedding()
        
        self.perform_HDBSCAN()
        
    def perform_UMAP_embedding(self):
        
        embedding = umap.UMAP(random_state=self.rand_state,
                              n_neighbors = self.reduce_nn,
                              min_dist = self.min_d,
                              metric=self.metric).fit_transform(self.matrix)
        
        self.UMAP_embedding = embedding
        
    def perform_HDBSCAN(self):
        
        clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.cluster_nn, 
                min_samples=self.min_c).fit(self.UMAP_embedding)
        
        self.clust_pred_labels = clusterer.labels_

    def plot_unsupervised_embedding(self,save=1,nn=20,min_d=0.5,n_clu=10,min_c=1):
        
        # Perform UMAP dimensionality reduction and HDBSCAN clustering...
        # on the raw spectral matrices, plot cluster-colors of the embedded output
        # Settings:
        # nn = nearest neighbours of each observation to consider in the embedding
        # n_clust = minimum number of observations of a single chemical type
        # min_dist is the minimum distance between lower-space embedding points
        # min_c should be set higher if we think there are noisy outsiders, e.g.10
        try:
            self.UMAP_embedding
        except:
            print('No embedding found, performing embedding in plot function...')
            self.perform_UMAP_embedding(nn=nn, min_d=min_d)
        
        try:
            self.clust_pred_labels
        except:
            print('No clustering found, performing HDBSCAN in plot function...')
            self.perform_HDBSCAN(n_clu=n_clu, min_c=min_c)
            
        scatterplot_2D(self.UMAP_embedding,self.clust_pred_labels,save=1)
        
    def plot_clusters(self,labels):
        
        scatterplot_2D(self.UMAP_embedding,labels,save=0)
        
def scatterplot_2D(arr,targets,save):

    # Color scatter plot an Y x 2 matrix with Y target labels of N classes
    
    plt.scatter(arr[:,0],arr[:,1],c=targets)
    plt.scatter(arr[:,0],arr[:,1],c=(
                    targets),facecolors='none',edgecolors='k',cmap = 'bwr')
    if save == 1:
        plt.savefig('color_scatterplot.eps', format='eps')
        
    plt.colorbar()    
    plt.show()

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
