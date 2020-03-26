# PhenoBench

This is an in-progress suite of software tools built by the Utah TORCH lab for benchmarking TBI phenotyping implementations.
The benefits of benchmarking your study are:
1. The security of cross-comparison and confirmation of your results with other reports, prior to a submission.
2. Improved confidence in your work and claims by integration of your findings with the results of other reports.
3. A useful mechanism to demonstrate why your phenotyping approach is valid and/or superior in your context to a publicly available baseline.

PhenoBench is written in Python 3.x, and requires some python scientific computing libraries (see requirements.txt and installation instructions). An R implementation will be developed at a later date depending on feedback and requests.

**Quick Start Instructions**

1. Install [Anaconda](https://www.anaconda.com/download/?lang=en-us) for Python 3.x.
2. Open Anaconda Navigator, go to Environments, create a new environment, open and activate its terminal, and run 

"conda install -c conda-forge --yes --file requirements.txt"

3. Download the Github desktop client. Clone the master branch (also see *Git Convention*).
4. Open Spyder or a python terminal. Navigate to the PhenoBench home directory (github/PhenoBench) and run the getting started script. 

# Data Repository

The github is only for code and small text/csv files. 
No real data of any kind is permitted in this repository.
The repository contains tools for generating pseudodata based on open-access-published covariates between TBI-associated variables.
The repository contains tools for running and reporting phenotyping benchmarks on the generated random data.

------

# Data Structures

The scripts are orientated around tolerating SAS files and csv files as read-in 'real' data sources.

------

# Quick examples

1. run generate_random_data.py to generate and write a csv data dictionary of correlated pseudo TBI data variables.
2. run pheno_bench_data.py to read in, encode, perform dimensionality reduction and unsupervised clustering on 'generated_data.csv'
3. run report_benchmarks.py to output tables and figures of the benchmarks achieved on the random pseudo dataset.

------

**Git Convention**

1. Make a new *branch* to keep track of your code edits and additions. Name it anything you like. Commit your changes to this branch. Please try to make things modular and well-commented.
2. When you have code edits/additions ready to share, create a *Pull Request* to propose incorporating the changes from your branch back into the master branch.
3. Please try to re-use functions, classes and common language.

------

# Contributing

Anyone can contribute, but please do not make direct changes to the master. Instead, make changes to your own branch, then create a *Pull Request* to propose incorporating the changes from your branch back into the master branch.

------
