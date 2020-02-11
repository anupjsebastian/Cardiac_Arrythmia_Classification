# Cardiac Arrythmia Classification



## Description

This project involves classifying time series data obtained from ECGs. The project aims to correctly identify and classify abnormal hearbeats, to improve patient care and also to provide expert diagnosis where it is not so easily available. 

## Datasets

The data sets used for this project are:

1. MIT-BIH Arrythmia Dataset
* Number of Samples: 109446
* Number of Categories: 5
* Sampling Frequency: 125Hz
* Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]

     These correspond to

        N: Normal beat
        S: Supraventricular premature beat
        V: Premature ventricular contraction
        F: Fusion of ventricular and normal beat
        Q: Unclassifiable beat

2. PTB Diagnostic ECG Database:
* Number of Samples: 14552
* Number of Categories: 2
* Sampling Frequency: 125Hz
* Classes [0: Normal Hearbeat, 1: Abnormal Hearbeat]

The data was downloaded from: https://www.kaggle.com/shayanfazeli/heartbeat

## Additional Files Required

Before running the notebooks in this project, please download these additional files and add them to the project directory from the following link. They were uploaded separately because of the storage limitations of GitHub.

https://mega.nz/#F!gipj0AIJ!iVgMOjtbSPuzurRcqRH_Ew


The list of files that you should have are as follows:
1. data.pickle
2. mitbih_test.csv
3. mitbih_train.csv
4. ptbdb_abnormal.csv
5. ptbdb_normal.csv

## Notebook Order

The order in which the notebooks are presented are as follows:

1. Exploratory Data Analysis.ipynb
2. Model 1 KNN Classifier.ipynb
3. Model 2 KNN Classifier with Matrix Profiling.ipynb
4. Model 3 Binary Classifier.ipynb

## Utilities File

A number of custom utility functions were used throughout the project. The code for these functions can be found in the *my_utils.py* file. 


## Presentation

The slides for this project is included in the repository, named as *ECG Presentation.pdf*. It can also be found online at 
https://www.canva.com/design/DADzFLtry7Y/riyFx5tWtwPqJSTbd-5pgA/view?utm_content=DADzFLtry7Y&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink
