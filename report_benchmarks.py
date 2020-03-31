# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:21:12 2020

@author: Eamonn
"""

from pheno_bench import PhenoBench

Benchmark = PhenoBench() 

Benchmark.load(file = r'gen_data.csv')

Benchmark.KMEANS_clusters = 2 # Optionally change number of KMEANS clusters

Benchmark.run(dim_reduce='UMAP',cluster='HDBSCAN')
Benchmark.plot_clusters();plt.show()
# Benchmark.plot_clusters_example()

Benchmark.run(dim_reduce='PCA',cluster='KMEANS')
Benchmark.plot_clusters();plt.show()
# Benchmark.plot_clusters_example()

Benchmark.hopkins_test()