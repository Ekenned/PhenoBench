# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:21:12 2020

@author: Eamonn
"""

from pheno_bench import PhenoBench
import matplotlib.pyplot as plt

Benchmark = PhenoBench() 

Benchmark.load(file = r'gen_data.csv')

# Benchmark.print_settings() # Print all options which can be set manually
# Benchmark.KMEANS_clusters = 2 # Example setting change: Num KMEANS clusters

Benchmark.run(dim_reduce='PCA',cluster='KMEANS')
Benchmark.plot_clusters();plt.title('PCA and Kmeans (basic)');plt.show()
Benchmark.report_statistics()

Benchmark.run(dim_reduce='UMAP',cluster='HDBSCAN')
Benchmark.plot_clusters();plt.title('UMAP and HDBSCAN (advanced)');plt.show()
Benchmark.report_statistics()

# Benchmark.plot_clusters_example() # Explore detailed trends in clusters
