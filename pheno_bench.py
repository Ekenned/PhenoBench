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
from sklearn import decomposition
from sklearn.cluster import KMeans

font = {'family' : 'Arial',
        'size'   : 9}
mpl.rc('font', **font)

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
        
        Example usage 2:
        # Phenotyping with custom settings
        #----------------------------------------
            
        Benchmark = PhenoBench() 
        Benchmark.load(file = r'.../Data/TBI_data.sas7bdat')
        Benchmark.run(dim_reduce = 'UMAP',
                      cluster = 'HDBSCAN',
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
                    'KMEANS_clusters':2, # Only valid for cluster=KMEANS
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
        
    def run(self, dim_reduce='UMAP', cluster='HDBSCAN'):
        
        if self.norm_vars == 0:
            print("Not normalizing features...")
            
        self.embedding = np.copy(self.matrix) # default is no embedding
        
        # Embedding control logic
        if dim_reduce == 'UMAP':
            print('Performing uniform manifold embedding...')
            self.UMAP_embedding()
            
        if dim_reduce == 'PCA':
            print('Performing PCA embedding...')
            self.PCA()
            
        # Clustering control logic
        if cluster == 'HDBSCAN':
            print('Performing hierarchial density-based clustering...')
            self.HDBSCAN_clusterer()
            
        if cluster == 'KMEANS':
            print('Performing K-MEANS clustering...')
            self.KMEANS_clusterer()

        
    def UMAP_embedding(self):
        # Perform N->2 dimensional uniform manifold embedding
        
        embedding = umap.UMAP(random_state = self.rand_state,
                                   n_neighbors = self.reduce_nn,
                                   min_dist = self.min_d,
                                   metric=self.metric).fit_transform(
                                           self.matrix)
        
        self.embedding = np.copy(embedding)
        
    def PCA(self):
        # Perform N->2 dimensional principal component reduction
        
        pca = decomposition.PCA(n_components=2)
        pca.fit(self.matrix.values)
        embedding = pca.transform(self.matrix.values)
        
        self.embedding = np.copy(embedding)

    def HDBSCAN_clusterer(self):
        
        clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.cluster_nn, 
                min_samples=self.min_c).fit(self.embedding)
        
        self.clust_pred_labels = clusterer.labels_
        
    def KMEANS_clusterer(self):

        self.clust_pred_labels = KMeans(
                n_clusters=self.KMEANS_clusters).fit_predict(self.embedding)
        
    def plot_clusters(self,labels=0):
        
        try:
            if labels == 0:
                labels = self.clust_pred_labels
        except:
            0
            
        scatterplot_2D(self.embedding,labels,save=0)
        
    def plot_clusters_example(self):
        
        self.plot_clusters(labels=self.matrix['LOC_REPORTED'])
        plt.title('Colored by: At least one LOC')
        plt.show()
        
        self.plot_clusters(labels=self.matrix['LOC_DUR'])
        plt.title('Colored by: LOC duration (s)')
        plt.show()
        
        self.plot_clusters(labels=self.matrix['TRAILS_B'])
        plt.title('Colored by: TRAILS B score')
        plt.show()
        
        self.plot_clusters(labels=self.clust_pred_labels)
        plt.title('Colored by: Clusterer predicted labels')
        plt.show()
    
        # Color scatter plot an Y x 2 matrix with Y target labels of N classes
        # Performed 4 times with different variables as colors
#        x = self.UMAP_embedding[:,0]
#        y = self.UMAP_embedding[:,1]
#        
#        fig, axs = plt.subplots(2, 2)
#        axs[0, 0].set_title('Colored by: At least one LOC')
#        axs[0, 0].set_xticks(ticks=[])
#        axs[0, 0].scatter(x,y,c = self.matrix['LOC_REPORTED'],
#           facecolors='none',edgecolors='k',cmap = 'bwr')
#        #axs[0, 0].colorbar(shrink = 0.5)
#        
#        axs[1, 0].set_title('Colored by: LOC duration (s)')
#        axs[1, 0].set_xticks(ticks=[])
#        axs[1, 0].scatter(x,y,c = self.matrix['LOC_DUR'],
#           facecolors='none',edgecolors='k',cmap = 'bwr')
#        #axs[1, 0].colorbar(shrink = 0.5)
#        
#        axs[0, 1].set_title('Colored by: TRAILS B results')
#        axs[0, 1].set_xticks(ticks=[])
#        axs[0, 1].scatter(x,y,c = self.matrix['TRAILS_B'],
#           facecolors='none',edgecolors='k',cmap = 'bwr')
#        #axs[0, 1].colorbar(shrink = 0.5)
#        
#        axs[1, 1].set_title('Colored by: HDBSCAN clustering')
#        axs[1, 1].set_xticks(ticks=[])
#        axs[1, 1].scatter(x,y,c = self.clust_pred_labels,
#           facecolors='none',edgecolors='k',cmap = 'bwr')
#        plt.show()
        
def scatterplot_2D(arr,targets,save):

    # Color scatter plot an Y x 2 matrix with Y target labels of N classes
    
    plt.scatter(arr[:,0],arr[:,1],c=targets)
    plt.scatter(arr[:,0],arr[:,1],c=(
                    targets),facecolors='none',edgecolors='k',cmap = 'bwr')
    if save == 1:
        plt.savefig('color_scatterplot.eps', format='eps')
        
    plt.colorbar()    
    # plt.show()

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
