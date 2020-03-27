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
        
        - Methods for imputation
        - Methods for categorical data encoding and handling
        - Methods for estimating the optimal number of phenotypes
        
        Example 1: 
        # Phenotyping with recommended settings and file writing
        #----------------------------------------
            
        Benchmark = PhenoBench() 
        Benchmark.load(file = r'.../Data/gen_data.csv')
        Benchmark.run()
        Benchmark.output()
        
        Example 2:
        # Phenotyping with custom settings and no file writing
        #----------------------------------------
            
        Benchmark = PhenoBench() 
        Benchmark.load(file = r'.../Data/TBI_data.sas7bdat')
        Benchmark.run(dim_reduce = 'UMAP',
                      cluster = 'HDBSCAN',
                      reduce_nn =  30,
                      cluster_nn = 20,
                     )
        Benchmark.output(write_figures = 0,
                         write_tables = 0)
        
    """
    
    def __init__(self):
        
        # Assign default settings
        defaults = {'norm_vars':0,
                    'dim_reduce':'UMAP',
                    'cluster':'HDBSCAN',
                    'reduce_nn':20,
                    'min_d':0.1,
                    'cluster_nn':20,
                    'metric':'correlation',
                    'write_settings':1,
                    'write_figures':1,
                    'write_tables':1,
                    }
        
        # Inherit defaults
        self.norm_vars = defaults['norm_vars'] 
        self.dim_reduce = defaults['dim_reduce']
        self.cluster = defaults['cluster']
        self.min_d = defaults['min_d']
        self.reduce_nn = defaults['reduce_nn']
        self.cluster_nn = defaults['cluster_nn']
        self.metric = defaults['metric']
        self.write_settings = defaults['write_settings']
        self.write_figures= defaults['write_figures']
        self.write_tables = defaults['write_tables']
        self.settings = defaults
        
    def load(self,file):
        
        self.matrix = pd.load(file)
        # Check if data has nans or strings, perform imputation, 
        self.validate_data()
    
    def validate_data(self):
        0
        # Check if data has nans or strings, perform imputation, 


    def perform_UMAP_embedding(self,nn=20,min_d=0.5):
        
        embedding = umap.UMAP(random_state=self.rand_state,
                              n_neighbors = self.reduce_nn,
                              min_dist = self.min_d,
                              metric=self.metric).fit_transform(self.matrix)
        
        self.UMAP_embedding = embedding
        
    def perform_HDBSCAN(self,n_clu=10,min_c=1):
        
        clusterer = hdbscan.HDBSCAN(
                min_cluster_size=n_clu, 
                min_samples=min_c).fit(self.UMAP_embedding)
        
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

TMP = np.abs( X.corr().values )
TMP[np.isnan(TMP)] = 0
embedding = umap.UMAP(random_state=42,
                      n_neighbors = 30,
                      min_dist = 0.1,
                      metric='correlation').fit_transform(TMP)

clusterer = hdbscan.HDBSCAN(min_cluster_size=10,
                            min_samples=15).fit(embedding)

Z = clusterer.single_linkage_tree_.to_numpy()

plt.scatter(embedding[:,0],
            embedding[:,1],
            s=6,
            c=clusterer.labels_,
            )
plt.title("Embedding of item correlations")
plt.show()

plt.scatter(embedding[:,0],
             embedding[:,1],
             s=12,
             c=clusterer.single_linkage_tree_.get_clusters(3, min_cluster_size=5));
plt.colorbar();
plt.show()

fig, axes = plt.subplots(1, 1, figsize=(8, 4))
plt.title("Dendrogram of item correlations")
dn = scipy.cluster.hierarchy.dendrogram(Z,ax=axes)
plt.ylim((0,15))
plt.show()
