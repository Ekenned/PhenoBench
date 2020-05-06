# PhenoBench

This is an in-progress suite of software tools built by the Utah TORCH lab for benchmarking TBI phenotyping studies.

PhenoBench is written in Python 3.x, and requires some python scientific computing libraries (see requirements.txt, or installation instructions for new users).

The benefits of developing your study with PhenoBench are:
1. *Simplicity* Once pointed to your curated data dictionary, PhenoBench can return figures in 1 line of code.
2. *Validity* Cross-compare and confirm of your results with other TBI reports which use the same analysis.
3. *Confidence* Your work and claims are built on the solid foundation of a peer-reviewed analysis pipeline.
4. *Integration* PhenoBench allows others to understand your work through the lens of a familiar set of measures.

**Quick Start Instructions**

1. Install the freeware [Anaconda](https://www.anaconda.com/download/?lang=en-us) for Python 3.x.
2. Download the free Github desktop client. Clone the master branch to your local computer (also see *Git Convention*).
3. (Search for) and open anaconda prompt, to create and activate a new environment (e.g. called Phenobench here) by running: 

"conda create --name PhenoBench" and afterwards, activate it by running: "conda activate PhenoBench"

Navigate to the PhenoBench home directory (github/PhenoBench) e.g. on windows run: "cd C:\Users\ . . . \GitHub\PhenoBench"

To install all packages run: "conda install -c conda-forge --yes --file requirements.txt" and "pip install pyclustertend"

The environment setup is now complete, and can be reactivated anytime by running "conda activate PhenoBench" in anaconda prompt.

------

# Quick examples

1. run "generate_random_data.py" to generate "gen_data.csv": A data dictionary of pseudo-correlated TBI data variables.
2. run "report_benchmarks.py" to output tables and figures of the benchmarks achieved on the random pseudo dataset "gen_data.csv".
3. replace "gen_data.csv" with your own sensitive data dictionary, and rerun "report_benchmarks.py"
4. Read the PhenoBench class comments to try out different settings and options.
------

# Data Repository

The github is only for code and simulated text/csv files. 
No real data of any kind is permitted in this repository.
The repository contains tools for generating pseudodata based on open-access-published covariates between TBI-associated variables.
The repository contains tools for running and reporting phenotyping benchmarks on any TBI-associated data.

------

**Git Convention**

1. Make a new *branch* to keep track of your code edits and additions. Name it anything you like. Commit your changes to this branch. Please try to make things modular and well-commented.
2. When you have code edits/additions ready to share, create a *Pull Request* to propose incorporating the changes from your branch back into the master branch.

------

# Contributing

Anyone can contribute, but please do not make direct changes to the master. Instead, make changes to your own branch, then create a *Pull Request* to propose incorporating the changes from your branch back into the master branch.

------
