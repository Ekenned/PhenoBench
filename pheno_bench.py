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
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

class PhenoBench():
    
    """
    A class that loads a phenotype TBI dataset and performs 
    standardized phenotyping of the data.
    
    Includes:
        
        Core features: 
            
        - A data dictionary loader which tolerates csv and sas file read-in
        
        Phenotyping options leveraging:
        - Unsupervised dimensionality reduction with PCA
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
        Phenotyping with custom settings and mixed methods
        # Here, no normalization of features is performed, and
        # we use only 50% of the data, as a split-sample example
        #----------------------------------------
            
        Benchmark = PhenoBench() 
        Benchmark.load(file = r'gen_data.csv')
        Benchmark.settings.n_neighbors = 5
        Benchmark.settings.norm_vars = 0
        Benchmark.settings.KMEANS_clusters = 5
        Benchmark.settings.split = 0.5 # example split sampling
        Benchmark.run(dim_reduce = 'UMAP',
                      cluster = 'KMEANS',
                      )
        Benchmark.plot_clusters_example()
        Benchmark.report_statistics()
        Benchmark.plot_vat() # visual assessment of tendency takes several mins
        
    """
    
    def __init__(self):

        # set font settings
        font = {'family' : 'Arial',
        'size'   : 9}
        mpl.rc('font', **font)

        # Assign default settings in a dictionary
        self.settings = {
        'norm_vars':1, # optionally standard score all variables
        'dim_reduce':'UMAP', # UMAP, PCA
        'cluster':'HDBSCAN', # HDBSCAN, KMEANS
        'KMEANS_clusters':2, # Only used when cluster=KMEANS
        'split':None, # Choose a split fraction e.g. 0.5, for stability testing
        'rand_state':42, # default state for random seed
        'split_side':'left', # Select which side of the split data to analyze
        'metric':'euclidean', # UMAP settings, euclidean or correlation
        'n_neighbors':50, # Number of observation neighbours for comparison
        'min_d':0.05, # UMAP settings
        'min_samples':30, # HDBSCAN settings
        'min_cluster_size':30, # HDBSCAN settings
        'save_outputs':1, # toggle write of settings, plots, tables, etc.
        }
    
    def print_settings(self): 
        # Print default or updated list of settings"   

        zip_settings = zip(self.settings.keys(), self.settings.values())
        [print(i[0],':',i[1]) for i in zip_settings][0]

    # Methods for loading, validating, and scaling data
    ###################################################### 

    def load(self,file):
        # Load the data from a csv file
        
        self.raw_df = pd.read_csv(file)
  
        self.validate_data()
        
    def set_data(self,df):
        # Use some pre-existing pandas dataframe, df, as the data
        
        self.raw_df = df
  
        self.validate_data()

    def validate_data(self):
        # TBD: Check if data has nans or strings, perform imputation, etc.
        
        self.n_obs = np.shape(self.raw_df)[0] # get the basic data dimensions
        self.n_vars = np.shape(self.raw_df)[1]
        self.df_means = self.raw_df.mean()
        self.traits = [i for i in self.raw_df.columns]
        
    def split_df(self,):
        
        split_inds = test_train_inds(self.n_obs,
            self.settings['split'],
            self.settings['rand_state'])

        self.left_split_inds = np.sort(split_inds[0])
        self.right_split_inds = np.sort(split_inds[1])

        self.unsplit_df = self.raw_df # Not needed, but seems sensible to hold onto this
        self.l_df = self.raw_df.iloc[self.left_split_inds]
        self.r_df = self.raw_df.iloc[self.right_split_inds]

    def rescale_data(self):
        
        if self.settings['splitting_performed'] == 1:
            prescaled_data = self.s_df # use split data
        else:
            prescaled_data = self.raw_df # use raw data
        scaler = StandardScaler()
        self.rescaled_df = pd.DataFrame(data=scaler.fit_transform(prescaled_data), 
            columns=prescaled_data.columns,)
        
    # "Run" contains 1. Splitting 2. Normalization, 3. Embedding, 4. Clustering
    ######################################################  

    def run(self, 
        dim_reduce='UMAP', 
        cluster='HDBSCAN', 
        split=None, 
        split_side='left',
        rand_state=42,
        save_outputs=1,
        ):

        # Update the settings dictionary based on user inputs
        self.settings['dim_reduce'] = dim_reduce
        self.settings['cluster'] = cluster
        self.settings['split'] = split
        self.settings['split_side'] = split_side
        self.settings['rand_state'] = rand_state
        self.settings['save_outputs'] = save_outputs
        
        # Create run variables for tracking processes performed
        self.settings['splitting_performed'] = 0
        self.settings['reduction_performed'] = 0
        self.settings['normalization_performed'] = 0
        self.settings['clustering_performed'] = 0

        """1. Splitting
        ######################################################
        Provides the option to split the data in two for stability assessment, e.g.:
        Benchmark.run(split=0.3) # Only embed using 30% of the observations
        Benchmark.run(split=0.5,split_side='left') # 50/50 split sampling comparisons
        Benchmark.run(split=0.5,split_side='right')
        Default is no splitting (split=None)
        """
        if split != None:
            self.split_df() # split the dataframe
            self.settings['splitting_performed'] = 1

            if split_side == 'left':
                self.s_df = self.l_df.copy()
            elif split_side == 'right':
                self.s_df = self.r_df.copy()
            else:
                print('Split side not selected: Defaulting to full dataframe analysis')
        
        """ 2. Normalization
        ######################################################
        Provides the option to normalize the data.
        Features have equal weight in embedding if standard scored
        Both rescaled_df and raw_df just become "df" after this step
        Default is to standard score (Benchmark.settings['norm_vars']=1)
        """
        if self.settings['norm_vars'] == 0:
            print("Not normalizing features...")
            if self.settings['splitting_performed'] == 1:
                self.arr = np.copy(self.s_df) # use split data
            else:
                self.arr = np.copy(self.raw_df) # use raw data
        else:
            print("Standard scoring features...")
            self.rescale_data()
            self.settings['normalization_performed'] = 1
            self.arr = np.copy(self.rescaled_df)

        """ 3. Embedding
        ######################################################
        Provides the option to perform a 2D embedding of the dataframe
        using different dim. reduction methods. 
        """
        if self.settings['dim_reduce'] == 'UMAP':
            print('Performing uniform manifold embedding...')
            self.UMAP_embedding()
            self.settings['reduction_performed'] = 1

        elif self.settings['dim_reduce'] == 'PCA':
            print('Performing PCA embedding...')
            self.PCA()
            self.settings['reduction_performed'] = 1

        else:
            self.embedding = np.copy(self.arr) # default is no embedding
            
        """ 4. Clustering
        ######################################################
        Provides the option to perform clustering on the embedding
        using different methods. 
        """
        if self.settings['cluster'] == 'HDBSCAN':
            print('Performing hierarchial density-based clustering...')
            self.HDBSCAN_clusterer()
            self.settings['clustering_performed'] = 1
            
        elif self.settings['cluster'] == 'KMEANS':
            print('Performing K-MEANS clustering...')
            self.KMEANS_clusterer()
            self.settings['clustering_performed'] = 1
        else:
            print('Performing hierarchial density-based clustering...')
            self.HDBSCAN_clusterer() # HDBSCAN is defult clusterer
            self.settings['clustering_performed'] = 1

        self.cluster_props() # establish cluster properties

    # Dimensionality reduction methods
    ###################################################### 

    def UMAP_embedding(self):
        # Perform N->2 dimensional uniform manifold embedding
        
        embedding = umap.UMAP(random_state = self.settings['rand_state'],
                                   n_neighbors = self.settings['n_neighbors'],
                                   min_dist = self.settings['min_d'],
                                   metric=self.settings['metric'],
                                   ).fit_transform(self.arr)
        
        self.embedding = np.copy(embedding)
        
    def PCA(self):
        # Perform N->2 dimensional principal component reduction
        
        pca = decomposition.PCA(n_components=2)
        pca.fit(self.arr)
        embedding = pca.transform(self.arr)
        
        self.embedding = np.copy(embedding)

    # Clustering methods
    ###################################################### 

    def HDBSCAN_clusterer(self):
        # Perform hierarchial density based clustering with applications to noise
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.settings['min_cluster_size'], 
            min_samples=self.settings['min_samples']).fit(self.embedding)

        self.clust_labels = clusterer.labels_
        
    def KMEANS_clusterer(self):
        # Perform K-means clustering

        self.clust_labels = KMeans(
                n_clusters=self.settings['KMEANS_clusters']).fit_predict(self.embedding)

    # Phenotyping methods
    ###################################################### 

    def cluster_props(self):
        # Define cluster properties

        self.clust_varname = "CLUSTER" # select a name for the cluster variable
        self.merge_col = "group_ID" # select a name for the post-clustered groups variable
        self.outliers = np.where(self.clust_labels<0)[0] # unclassified cases
        self.n_outliers = len(self.outliers)
        
        self.classified = np.where(self.clust_labels>=0)[0] # classified cases
        self.clusts = np.unique(self.clust_labels[self.classified])
        self.n_clusts = len(self.clusts)
        self.out_df = self.raw_df.copy()
        self.out_df[self.clust_varname] = self.clust_labels

        alphabet = 10*'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.group_letters = [i for i in alphabet[0:self.n_clusts]]
        self.subplot_l()

    def calc_phenotypes(self, selected_df = "out", func = "mean",):
        # Calculate the statistical property for each cluster for the selected dataframe


        # Select the dataframe for evaluating phenotypes, we could use the raw, or standardized, etc.
        if selected_df == "out":
            M = self.out_df
        elif selected_df == "mean_ratio": # divide by mean per variable
            self.mean_ratio_df = self.out_df.div(self.out_df.mean()).copy()
            self.mean_ratio_df[self.clust_varname] = self.clust_labels
            M = self.mean_ratio_df.copy()
        elif selected_df == "standardized":
            M = self.s_df
        else:
            M = self.raw_df # default to the raw dataframe

        # instantiate a phentype dataframe
        phenotype_df = pd.DataFrame(data = self.clusts,columns=[self.merge_col])

        # For ever trait in the raw dataframe, calculate the mean (or other stat) for each cluster
        for i in self.traits:
            
            iterate_df = group_stats(M = M,
                        variable = i, # variable to test for phenotype
                        clustcol = self.clust_varname, # cluster variable
                        func = func, # choice of func, e.g. "median" ,"std" , "min", ...
                        )
            
            phenotype_df = pd.merge(phenotype_df,iterate_df,on=self.merge_col)
            
        # phenotype_df.drop(columns = [func + "_" + self.clust_varname])

        self.phenotype_df = phenotype_df.copy()

    def subplot_l(self, l = range(1,8)):
    
        """Given n plots, output l subplot axes, with a min of 1x1 subplots
        and a max of 8x8 subplots, as defined by lengths included in l """
        
        enough_subplots = (np.square(l) - self.n_clusts)>=0
        self.subplot_length = l[np.min(np.where(enough_subplots))]

    # Evaluation methods
    ###################################################### 
    
    def plot_vat(self):
        print('Visualizing tendency, this may take several minutes...')
        vat(self.arr)
        
    def report_statistics(self):
        
        print('----------------------')
        self.hopkins_test()
        self.silhoutte_test()
        print('----------------------')
        
    def silhoutte_test(self):
        
        self.sil_score = silhouette_score(
                self.embedding, self.clust_labels, metric='euclidean')
        
        print('Silhoutte score: ','%.4f' % self.sil_score)
    
    def hopkins_test(self):
        
        hopkins_val = hopkins(
                self.arr,np.shape(self.arr)[0])
        
        print('Hopkins score: ','%.4f' % hopkins_val)
        
    # Plotting methods
    ###################################################### 
        
    def plot_clusters(self,labels=0):
        
        try:
            if labels == 0:
                labels = self.clust_labels
        except:
            0
            
        scatterplot_2D(self.embedding,labels,save=0)

    def plot_multi_radar(self, max_radial_y = 4, color=0):
        # Produce a radar chart for every cluster, for every variable in the summary df
        # Export a pdf of the figure
        my_dpi=150
        plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
        # Loop to plot
        for row in range(0, len(self.phenotype_df.index)):

            self.make_single_radar(row=row, 
                title = self.group_letters[row],
                # title=self.phenotype_df[self.merge_col][row], 
                max_radial_y = max_radial_y, 
                color=color)

        if self.settings['save_outputs'] == 1:
            plt.savefig('phenotype_radials.pdf', format='pdf')

        plt.show()

    def make_single_radar(self, row, title, max_radial_y, color=0):
        # Produce a single radar chart for 1 cluster, for every variable in summary df

        if color == 0:
            color = plt.cm.get_cmap("Set2", len(self.phenotype_df.index))
            color = color(row)

        # number of variable
        categories=list(self.phenotype_df)[1:] # dont include group_ID
        # categories = list(self.raw_df) # THIS IS A HACK
        N = len(categories)
         
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * 3.1418 for n in range(N)]
        angles += angles[:1]
         
        # Initialise the spider plot
        ax = plt.subplot(self.subplot_length,self.subplot_length,row+1, polar=True,)
         
        # If you want the first axis to be on top:
        ax.set_theta_offset(3.1418 / 2)
        ax.set_theta_direction(-1)
         
        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories, color='grey', alpha=0.5,size=4)
         
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks(np.arange(max_radial_y), [str(i) for i in np.arange(4)], alpha=0.5, color="grey", size=8)
        # plt.yticks([1], ['1'], color="black", size=6)
        plt.ylim(0,max_radial_y)
         
        # Ind1
        values=self.phenotype_df.loc[row].drop('group_ID').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color="black", linewidth=1, linestyle='solid')
        ax.fill(angles, values, color=color, alpha=0.4)
         
        # Add a title
        plt.title(title, size=11, color=color, y=1.1)

    def plot_clusters_example(self):
        # This is a quick dataset-specific example of trait prevalences mapped by cluster
        
        self.plot_clusters(labels=self.raw_df['LOC_REPORTED'])
        plt.title('Colored by: At least one LOC')
        plt.show()
        
        self.plot_clusters(labels=self.raw_df['LOC_DUR'])
        plt.title('Colored by: LOC duration (s)')
        plt.show()
        
        #self.plot_clusters(labels=self.raw_df['TRAILS_B'])
        #plt.title('Colored by: TRAILS B score')
        #plt.show()
        
        #self.plot_clusters(labels=self.raw_df['AGE'])
        #plt.title('Colored by: AGE')
        #plt.show()
        
        #self.plot_clusters(labels=self.raw_df['PCLM_SCORE'])
        #plt.title('Colored by: PCLM_SCORE')
        #plt.show()
        
        #self.plot_clusters(labels=self.raw_df['N_TBI'])
        #plt.title('Colored by: N_TBI')
        #plt.show()
        
        self.plot_clusters(labels=self.raw_df['EPILEPSY'])
        plt.title('Colored by: EPILEPSY')
        plt.show()
        
        self.plot_clusters(labels=self.clust_labels)
        plt.title('Colored by: Clusterer predicted labels')
        plt.show()
    
# Misc functions
###################################################### 
        
def scatterplot_2D(arr,targets,save):

    # Color scatter plot an Y x 2 array with Y target labels of N classes
    
    plt.scatter(arr[:,0],arr[:,1],c=targets)
    plt.scatter(arr[:,0],arr[:,1],c=(
                    targets),facecolors='none',edgecolors='k',cmap = 'bwr')
    if save == 1:
        plt.savefig('color_scatterplot.eps', format='eps')
        
    plt.colorbar()    
    # plt.show()

def test_train_inds(n_obs,train_split,seed=0):
    
    # split indices randomly into test and train groups, specifying a seed for reprodubility
    np.random.seed = seed
    if (train_split != 0) and (train_split != 1):
        
        n_train = round(train_split*n_obs)
        train_inds = np.random.choice(n_obs,n_train,replace=False).astype(int)
        test_inds = np.setdiff1d(range(n_obs),train_inds).astype(int)
        
    else:
        
        # otherwise it is supervised
        train_inds = range(n_obs)
        test_inds = np.array([])

    return train_inds,test_inds

def group_stats(M,
                variable = "var2", 
                clustcol = "clust",
                func = "mean", # "median" ,"std" , "min", ...
                ):

    """return the mean/median, etc. of one variable for each cluster
    # Make test data
    test_arr = np.array([[1, 20, 10], 
                     [1, 30, 5], 
                     [2, 40, 7], 
                     [1, 20, 12], 
                     [3, 15, 8], 
                     [2, 50, 6], 
                     [0, 30, 9],
                     ])

    M = pd.DataFrame(data = test_arr, columns=['clust','var1','var2'])
    print(M) # test dataframe form of the above array
       function accepts 3 inputs:
           M: the dataset, which can be an array or dataframe
           variable: the column which contains the variable of interest, or a string of the column name
           clustcol: the column which identifies in which cluster the observation is, or a string of the column name
    """
    # Control logic for selecting arrays or dataframes with string columns
    # Error checking for the strings/values to exist in the dataframe/array
    if isinstance(variable, str):
        try:
            data = M[variable].values
            group_inds = M[clustcol].values
        except:
            print(variable,clustcol,' not found in dataframe')
            return 0 # end function early with null output
      
    elif isinstance(variable, int):
        try:
            data = M[:, variable]
            group_inds = M[:, clustcol]
        except:
            print(variable,clustcol,' not found in array')
            return 0 # end function early with null output
            
    unique_groups = np.unique(group_inds) # get unique group indices
    n = len(unique_groups)
    out = np.zeros(n) # one mean value for every unique group

    # Make a dataframe that will be the output, starting with group IDs
    df_out = pd.DataFrame(index = np.arange(n),
                          data = unique_groups,
                          columns = ["group_ID"],
                          )
         
    for i,g in enumerate(unique_groups):
        mask = np.where(group_inds == g)[0]
        
        if func[0:3] == "med": 
            out[i] = np.median(data[mask])
        
        elif func[0:3] == "min": 
            out[i] = np.min(data[mask])

        elif func[0:3] == "max": 
            out[i] = np.max(data[mask])
            
        elif func == "std": 
            out[i] = np.std(data[mask])
            
        elif func == "mean":
            out[i] = np.mean(data[mask])
        
        else:
            out[i] = np.mean(data[mask]) # set mean as the default function
            
    df_out[func+'_'+variable] = out    
    return df_out