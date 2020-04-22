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


#tasks = ["cochrane_summ", "medlineplus_references", "medlineplus_reviews", "bioasq/single-doc", "bioasq/multi-doc", "medinfo", "pubmed_summ", "ebm/answer", "ebm/justify"]
#tasks = ["medline_plus_references", "medline_plus_reviews"]
tasks = ['chiqa/section2answer_multi_abstractive', 'chiqa/page2answer_multi_abstractive', 'chiqa/section2answer_multi_extractive', 'chiqa/page2answer_multi_extractive', 'chiqa/section2answer_single_abstractive', 'chiqa/page2answer_single_abstractive', 'chiqa/section2answer_single_extractive', 'chiqa/page2answer_single_extractive']
#tasks = ["cochrane_summ"]
#tasks = ["medinfo"]
#tasks = ["pubmed_summ"]
#tasks = ["ebm/answer", "ebm/justify"]
data_dir = "/data/LHC_kitchensink/tensorflow_datasets_max"
for task in tasks:
    print(task)
    if "chiqa" not in task:
        data, info = tfds.load(task, with_info=True, split=tfds.Split.TRAIN, data_dir=data_dir)
    if "medinfo" in task or "ebm" in task or "medlineplus" in task or "cochrane" in task:
        data, info = tfds.load(task, with_info=True, split=tfds.Split.VALIDATION, data_dir=data_dir)
    if "bioasq/multi-doc" not in task:
        data, info = tfds.load(task, with_info=True, split=tfds.Split.TEST, data_dir=data_dir)
    
    print(info)
    print("Successfully loaded dataset: {}".format(task))
    #print(vars(data))
