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
from pubmed import PubmedSumm
from chiqa import Chiqa
from cochrane_summ import CochraneSumm


tasks = ["cochrane_summ", "medlineplus_references", "medlineplus_reviews", "bioasq/single-doc", "bioasq/multi-doc", "ebm", "medinfo", "pubmed_summ"]
#tasks = ["medline_plus_references", "medline_plus_reviews"]
#tasks = ["ebm"]
#tasks = ["medinfo"]
tasks = [                
    "chiqa/multi-abs-s2a", 
    "chiqa/multi-abs-p2a", 
    "chiqa/multi-ext-s2a", 
    "chiqa/multi-ext-p2a", 
    "chiqa/single-abs-s2a", 
    "chiqa/single-abs-p2a", 
    "chiqa/single-ext-s2a", 
    "chiqa/single-ext-p2a", 
]
#tasks = ["pubmed_summ"]
tasks = ["cochrane_summ"]
data_dir = "/data/LHC_kitchensink/tensorflow_datasets/"
for task in tasks:
    print(task)
    if "chiqa" not in task:
        data, info = tfds.load(task, with_info=True, split=tfds.Split.TRAIN, data_dir=data_dir)
    if "medinfo" in task or "ebm" in task or "medlineplus" in task or "cochrane" in task:
        data, info = tfds.load(task, with_info=True, split=tfds.Split.VALIDATION, data_dir=data_dir)
    if task == "bioasq/multi-doc" not in task:
        data, info = tfds.load(task, with_info=True, split=tfds.Split.TEST, data_dir=data_dir)
    
    #print(info)
    print("Successfully loaded dataset: {}".format(task))
