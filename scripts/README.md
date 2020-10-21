# Scripts for data processing

Included in the subdirectories here are the scripts necessary to prepare the BioASQ and DUC data for the fslks package. The rest of the datasets can be download in preprocessed format and stored in the proper directory, as specified by the fslks README.

## DUC  
Request access to the Duc 2004 and 2007 data from NIST at https://duc.nist.gov/. See the process_duc.py script for further instructions about which data files to request and how to prepare them for processing. Once the script has been run, the dataset files will be named duc_2004_test_collection.json and duc_2007_test_collection.json. These are the filenames that the duc.py fslks tensorflow dataset builder expects.

## BioASQ
Go to http://bioasq.org/participate and make an account. Download the training and testing data for BioASQ 8b. Then provide this data and the BioASQ PubMED articles, available from https://bionlp.nlm.nih.gov, as input into the process_bioasq.py script. See the script for further instructions.. The files produced by the script for the dataset builder will be named bioasq_train_collection.json and bioasq_test_collection.json.


