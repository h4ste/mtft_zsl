"""
Module for testing summarization datataset generators
"""

import tensorflow as tf
import tensorflow_datasets as tfds

from bioasq import Bioasq
from medlineplus_references import MedlineplusReferences
from medlineplus_reviews import MedlineplusReviews
from medinfo import Medinfo
from ebm import EBM

tasks = ["medlineplus_references", "medlineplus_reviews", "bioasq/single-doc", "bioasq/multi-doc", "ebm", "medinfo"]
#tasks = ["medline_plus_references", "medline_plus_reviews"]
#tasks = ["ebm"]
#tasks = ["medinfo"]
#tasks = [
#                name="multi-abs-s2a", 
#                name="multi-abs-p2a", 
#                name="multi-ext-s2a", 
#                name="multi-ext-p2a", 
#                name="single-abs-s2a", 
#                name="single-abs-p2a", 
#                name="single-ext-s2a", 
#                name="single-ext-p2a", 
for task in tasks:
    if "bioasq" in task:
        data_dir = "/data/LHC_kitchensink/tensorflow_datasets/"
    elif "medlineplus" in task:
        data_dir = "/data/LHC_kitchensink/tensorflow_datasets/"
    elif "medinfo" in task:
        data_dir = "/data/LHC_kitchensink/tensorflow_datasets/"
    elif "ebm" in task:
        data_dir = "/data/LHC_kitchensink/tensorflow_datasets/"
    data, info = tfds.load(task, with_info=True, split=tfds.Split.TRAIN, data_dir=data_dir)
    if "medinfo" in task or "ebm" in task:
        data, info = tfds.load(task, with_info=True, split=tfds.Split.VALIDATION, data_dir=data_dir)
    if task == "bioasq/single-doc" or "medinfo" in task or "ebm" in task:
        data, info = tfds.load(task, with_info=True, split=tfds.Split.TEST, data_dir=data_dir)
    
    #print(info)
    print("Successfully loaded dataset: {}".format(task))
