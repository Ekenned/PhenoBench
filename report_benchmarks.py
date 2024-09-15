# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:21:12 2020

@author: Eamonn
"""

from pheno_bench import PhenoBench
import matplotlib.pyplot as plt
import umap

Benchmark = PhenoBench() 

Benchmark.load(file = r'gen_data.csv') # load data

# Set settings:
# Benchmark.print_settings() # Print all options which can be set manually
Benchmark.settings['KMEANS_clusters'] = 4 # Example setting change: n clusters

# Run PCA / K-MEANS EXAMPLE:
Benchmark.run(dim_reduce='PCA',cluster='KMEANS') #,split=0.5) # split to bootst
Benchmark.plot_clusters();plt.title('PCA and Kmeans');plt.show()
Benchmark.report_statistics()


# Run UMAP / HDBSCAN EXAMPLE:
Benchmark.run(dim_reduce='UMAP',cluster='HDBSCAN') #,split=0.5) # split to bootst
Benchmark.plot_clusters();plt.title('HDBSCAN and UMAP');plt.show()
Benchmark.report_statistics()

# Calculate phenotypes characteristics
Benchmark.calc_phenotypes()

# Report and save outputs

Benchmark.phenotype_df.to_csv('phenotype_means.csv')

Benchmark.plot_multi_bar()
plt.savefig('phenotype_norms.pdf', format = 'pdf')
plt.show()
