# LearnM8
Active learning package for docking/consensus score prediction

THIS IS THE STATE OF SUBMISSION FOR MY BACHELOR THESIS

Explanation of the code and folder structure:

-The consensus directory contains all the consensus methods for calculation. I dont take credit for that, this was providied by github user https://github.com/Tonylac77. Parts of these functions are already available in https://github.com/DrugBud-Suite/DockM8.

-The data folder contains the data used in my thesis. The data is a subset of the targets available in LITPCBA https://drugdesign.unistra.fr/LIT-PCBA/ that were rescored with DockM8. For further details on how this data was obtained please reference my thesis.

-Inside the helper folder are smaller helper functions and wrappers.

-The learners folder contain the class related work, where we encapsulated different machine learning architectures into classes so they all can be used in the same way by our script.

-The results folder contains the results used to calculate the plots in my thesis.

-active_learning_function.py provides a function which encapsulates all the active learning logic.

-debug_main.py is the main script used in order to execute the experiments. This script takes a -m flag. This flag can have the values cpu, gpu, ssf1, and ss2. In order to generate the results, every flag must be called once.

-Dockerfile was used to create the docker image

-environment.yaml allows the user to recreate the used conda environment.

-evaluator.py was used to parse the data and evaluate the results. The generated outputs are mostly formatted in a way that they either represent a latex ticz picture directly or can be inserted into one. these were the exact scripts used to generate the plots for my thesis.

-all .sh and .sub files are needed to execute this pipeline on a condor hpc cluster.

