# Scripts for data processing

Included in the subdirectories here are the scripts necessary to prepare the BioASQ and DUC data for the fslks package. The rest of the datasets can be download in preprocessed format and stored in the proper directory, as specified by the fslks README.

## DUC  
Request access to the Duc 2004 and 2007 data from NIST at https://duc.nist.gov/. See the process_duc.py script for further instructions about which data files to request and how to prepare them for processing. Once the script has been run, the dataset files will be named duc_2004_test_collection.json and duc_2007_test_collection.json. These are the filenames that the duc.py fslks tensorflow dataset builder expects.

## BioASQ
Go to http://bioasq.org/participate and make an account. Download the training and testing data for BioASQ 8b. Then provide this data and the BioASQ PubMED articles, available from https://bionlp.nlm.nih.gov, as input into the process_bioasq.py script. See the script for further instructions.. The files produced by the script for the dataset builder will be named bioasq_train_collection.json and bioasq_test_collection.json. These datasets can then be placed in the proper manual tensorflow datasets directory.

## Cochrane   
To create the Cochrane review collection and the clinical answer collection, you can run the process_cochrane_reviews.py and the process_cochrane_clinical_answers.py, respectively. See the documentation inside the scripts for further guidance on their use.    

Before you can use either script, you will need to download the Cochrane citation data yourself and provide this as input to the scripts. To do so, go to https://www.cochranelibrary.com/cca and click on either the "Cochrane Reviews" or "Clinical Answers". Click the "select all" box and then click the "Export selected citations" button. Once you have done that for both the reviews and the clinical answers, you can now use these as input into the appropriate processing script. The instructions inside the script show you how to do this. Note that if you want to create the clinical answer collection, you must first create the review collection. This is because the answers use the reviews as reference and each answer can be mapped directly back to a single review.

If you check the log for the clinical answers, you may see notifications about 404 errors. This is because some of the urls in the exported citations are no longer valid. 

The scripts will produce train, validation, and test sets. 
