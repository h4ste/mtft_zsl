"""
Module for testing summarization datataset generators
"""

import tensorflow as tf
import tensorflow_datasets as tfds

#from bioasq import Bioasq
#from medlineplus_references import MedlinePlusReferences
#from medlineplus_reviews import MedlinePlusReviews
#from medinfo import Medinfo
from ebm import EBM

#tasks = ["bioasq", "medline_plus_references", "medline_plus_reviews"]
#tasks = ["medline_plus_references", "medline_plus_reviews"]
#tasks = ["medinfo"]
tasks = ["ebm"]
for task in tasks:
    if task == "bioasq":
        data_dir = "/data/LHC_kitchensink/bioasq"
        #data_dir = "/data/LHC_kitchensink/tensorflow_datasets/downloads/manual/bioasq/"
    elif "medlineplus" in task:
        data_dir = "/data/LHC_kitchensink/medlineplus/"
    elif "medinfo" in task:
        data_dir = "/data/LHC_kitchensink/medinfo/"
    elif "ebm" in task:
        data_dir = "/data/LHC_kitchensink/ebm/"
    data, info = tfds.load(task, with_info=True, split=tfds.Split.TRAIN, data_dir=data_dir)
    print(info)
    print("Successfully loaded dataset: {}".format(task))
