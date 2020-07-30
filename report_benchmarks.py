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
Benchmark.settings['KMEANS_clusters'] = 4 # Example setting change: n clusters
Benchmark.run(dim_reduce='PCA',cluster='KMEANS') #,split=0.5) # split to bootst
Benchmark.plot_clusters();plt.title('PCA and Kmeans');plt.show()
Benchmark.report_statistics()

# Benchmark.print_settings() # Print all options which can be set manually
Benchmark.run(dim_reduce='UMAP',cluster='HDBSCAN') #,split=0.5) # split to bootst
Benchmark.plot_clusters();plt.title('HDBSCAN and UMAP');plt.show()
Benchmark.report_statistics()

# Benchmark.plot_clusters_example() # Uncomment to explore phenotypes in clusts

Benchmark.calc_phenotypes(selected_df = "mean_ratio")

Benchmark.phenotype_df = Benchmark.phenotype_df.drop(
    columns=["mean_TRAILS_B","mean_RACE","mean_EDUCATION"]).copy()

Benchmark.plot_multi_radar()
