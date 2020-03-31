# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:21:12 2020

@author: Eamonn
"""

from pheno_bench import PhenoBench

Benchmark = PhenoBench() 

Benchmark.load(file = r'gen_data.csv')

Benchmark.run()

Benchmark.hopkins_test()

plt.title('Colored by at least one report of loss of consciousness (1/0)')
Benchmark.plot_clusters(labels = Benchmark.matrix['LOC_REPORTED'])

plt.title('Colored by loss of consciousness duration (s)')
Benchmark.plot_clusters(labels = Benchmark.matrix['LOC_DUR'])

plt.title('Colored by TRAILS B results')
Benchmark.plot_clusters(labels = Benchmark.matrix['TRAILS_B'])

plt.title('Colored by HDBSCAN clustering')
Benchmark.plot_clusters(labels = Benchmark.clust_pred_labels)