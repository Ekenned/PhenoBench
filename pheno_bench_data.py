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

mpl.rcParams['figure.dpi'] = 150

os.chdir(r'C:\Users\Eamonn\Box\UU_EK\Projects\PTE_C\data')
X = pd.read_csv('PTEData.PTESurvey20.csv')

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
