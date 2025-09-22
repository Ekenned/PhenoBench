# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:21:12 2020

@author: Eamonn
"""

from pheno_bench import PhenoBench
import matplotlib.pyplot as plt
import umap

analysis = PhenoBench() 

analysis.load(file = r'gen_data.csv') # load data

# print modifiable settings:
print(analysis.print_settings()) # Print all options which can be set manually

# Run PCA / K-MEANS example:
analysis.settings['KMEANS_clusters'] = 4 # Example setting change: n clusters
analysis.run(dim_reduce='PCA',cluster='KMEANS') 
analysis.plot_clusters();plt.title('PCA and Kmeans')
plt.savefig('PCA.pdf', format = 'pdf');plt.show()
analysis.report_statistics()


# Run UMAP / HDBSCAN example:
analysis.settings['n_neighbors'] = 50 # Example setting change: n neighbors
analysis.settings['min_cluster_size'] = 20 # HDBSCAN cluster size
analysis.run(dim_reduce='UMAP',cluster='HDBSCAN')
analysis.plot_clusters();plt.title('HDBSCAN and UMAP');
plt.savefig('UMAP.pdf', format = 'pdf');plt.show()
analysis.report_statistics()

# Calculate phenotypes characteristics
analysis.calc_phenotypes()

# Report and save outputs
analysis.phenotype_df.to_csv('phenotype_means.csv')
analysis.plot_multi_bar()
# plt.savefig('phenotype_norms.pdf', format = 'pdf')
plt.show()
