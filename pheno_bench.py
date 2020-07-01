# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:12:43 2020

@author: Eamonn
"""
import numpy as np
import pandas as pd
import umap
import hdbscan
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyclustertend import hopkins
from pyclustertend import vat
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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
        Phenotyping with default settings:
        #----------------------------------------
            
        Benchmark = PhenoBench() 
        Benchmark.load(file = r'gen_data.csv')
        Benchmark.run()
        Benchmark.plot_clusters()
        Benchmark.report_statistics()
        
        
        Example usage 2:
        Phenotyping with custom settings and mixed methods:
        #----------------------------------------
            
        Benchmark = PhenoBench() 
        Benchmark.load(file = r'gen_data.csv')
        Benchmark.reduce_nn = 5
        Benchmark.KMEANS_clusters = 5
        Benchmark.run(dim_reduce = 'UMAP',
                      cluster = 'KMEANS',
                      )
        Benchmark.plot_clusters_example()
        Benchmark.report_statistics()
        Benchmark.plot_vat() # visual assessment of tendency takes several mins
        
    """
    
    def __init__(self):
        
        # Assign default settings in a dictionary
        defaults = {'norm_vars':1, # optionally standard score all variables
                    'dim_reduce':'UMAP', # or GLRM, or PCA
                    'cluster':'HDBSCAN', # or KMEANS
                    'KMEANS_clusters':2, # Only valid for cluster=KMEANS
                    'split':0, # Choose a split fraction e.g. stability testing
                    'metric':'euclidean', # UMAP settings 'correlation'
                    'reduce_nn':20, # UMAP settings
                    'min_d':0.05, # UMAP settings
                    'min_c':20, # HDBSCAN settings
                    'cluster_nn':30, # HDBSCAN settings
                    'write_settings':1, # output toggle
                    'write_figures':1, # output toggle
                    'write_tables':1, # output toggle
                    'rand_state':0 # default state for random seeds
                    }
        
        # Inherit defaults as class variables
        self.norm_vars = defaults['norm_vars'] 
        self.dim_reduce = defaults['dim_reduce']
        self.cluster = defaults['cluster']
        self.KMEANS_clusters = defaults['KMEANS_clusters']
        self.split = defaults['split']
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
        
    
    # Class method definitions
    ######################################################  
    
    def print_settings(self): # try "Benchmark.print_settings()"
        
        [print(i[0],':',i[1]) for i in zip(
                self.settings.keys(),self.settings.values())][0]

    def load(self,file):
        
        self.matrix = pd.read_csv(file)
  
        # Check if data has nans or strings, perform imputation, 
        self.validate_data()
        
    def set_data(self,df):
        
        self.matrix = df
  
        # Check if data has nans or strings, perform imputation, 
        self.validate_data()
        
    def split_matrix(self):
        
        split_inds = test_train_inds(self.n_obs,self.split)
        self.left_split_inds = np.sort(split_inds[0])
        self.right_split_inds = np.sort(split_inds[1])

        self.orig_matrix = self.matrix.copy()
        self.matrix = self.orig_matrix.iloc[self.left_split_inds]
        self.remainder_matrix = self.orig_matrix.iloc[self.right_split_inds]
    
    def validate_data(self):
        
        self.n_obs = np.shape(self.matrix)[0] # get the basic data dimensions
        self.n_vars = np.shape(self.matrix)[1]
        # Check if data has nans or strings, perform imputation,
        
    def rescale_data(self):
        
        scaler = StandardScaler()
        tmp = scaler.fit_transform(self.matrix) 
        self.matrix = pd.DataFrame(data=tmp, columns=self.matrix.columns)
        
    # Dimensionality reduction and clustering methods
    ######################################################  

    def run(self, dim_reduce='UMAP', cluster='HDBSCAN',split=0):
        
        # Control logic for splitting the data
        if split  != 0:
            self.split = split
            self.split_matrix()
        
        # Keep track of what has been run
        self.reduction_performed = 0
        self.normalization_performed = 0
        self.clustering_performed = 0
        
        # Standard scoring control logic
        if self.norm_vars == 0:
            print("Not normalizing features...")
        else:
            print("Normalizing features...")
            self.rescale_data()
            self.normalization_performed = 1
            
        self.embedding = np.copy(self.matrix) # default is no embedding
        
        # Embedding control logic
        if dim_reduce == 'UMAP':
            print('Performing uniform manifold embedding...')
            self.UMAP_embedding()
            self.reduction_performed = 1
            
        if dim_reduce == 'PCA':
            print('Performing PCA embedding...')
            self.PCA()
            self.reduction_performed = 1
            
        # Clustering control logic
        if cluster == 'HDBSCAN':
            print('Performing hierarchial density-based clustering...')
            self.HDBSCAN_clusterer()
            self.clustering_performed = 1
            
        if cluster == 'KMEANS':
            print('Performing K-MEANS clustering...')
            self.KMEANS_clusterer()
            self.clustering_performed = 1

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
        
        try:
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.cluster_nn, 
                min_samples=self.min_c).fit(self.embedding)
            
        except:
        
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.cluster_nn[0], 
                min_samples=self.min_c).fit(self.embedding)
        
        self.clust_pred_labels = clusterer.labels_
        
    def KMEANS_clusterer(self):

        self.clust_pred_labels = KMeans(
                n_clusters=self.KMEANS_clusters).fit_predict(self.embedding)
        
    # Evaluation methods
    ###################################################### 
    
    def plot_vat(self):
        print('Visualizing tendency, this may take several minutes...')
        vat(self.matrix)
        
    def report_statistics(self):
        
        print('----------------------')
        self.hopkins_test()
        self.silhoutte_test()
        print('----------------------')
        
    def silhoutte_test(self):
        
        ss = silhouette_score(
                self.embedding, self.clust_pred_labels, metric='euclidean')
        
        print('Silhoutte score: ','%.4f' % ss)
    
    def hopkins_test(self):
        
        hopkins_val = hopkins(
                self.matrix,np.shape(self.matrix.values)[0])
        
        print('Hopkins score: ','%.4f' % hopkins_val)
        
    # Plotting methods
    ###################################################### 
        
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
        
        self.plot_clusters(labels=self.matrix['AGE'])
        plt.title('Colored by: AGE')
        plt.show()
        
        self.plot_clusters(labels=self.matrix['PCLM_SCORE'])
        plt.title('Colored by: PCLM_SCORE')
        plt.show()
        
        self.plot_clusters(labels=self.matrix['N_TBI'])
        plt.title('Colored by: N_TBI')
        plt.show()
        
        self.plot_clusters(labels=self.matrix['EPILEPSY'])
        plt.title('Colored by: EPILEPSY')
        plt.show()
        
        self.plot_clusters(labels=self.clust_pred_labels)
        plt.title('Colored by: Clusterer predicted labels')
        plt.show()
    
# Misc functions
###################################################### 
        
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