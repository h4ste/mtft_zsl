"""
Module to create tensorflow dataset of publisher created summaries of articles in PubMed.
"""

import os
import glob

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """PubMed Publisher-Summary dataset, generated using 2020 PubMed baseline""" 

_CITATION = """This work is 100% plagiarized"""

_PUBMED_DOWNLOAD_INSTRUCTIONS = """Do stuff here. Or don't. Who cares."""

class PubMedSumm(tfds.core.GeneratorBasedBuilder):
    """PubMed Publisher-Summary dataset builder"""

    VERSION = tfds.core.Version("1.0.0")
    MANUAL_DOWNLOAD_INSTRUCTIONS = _PUBMED_DOWNLOAD_INSTRUCTIONS

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'article': tfds.features.Text(),
                'title': tfds.features.Text(), 
                'summary': tfds.features.Text(),
                'pubdate': tfds.features.Text(),
                'pmid': tfds.features.Text(),
            }),
            supervised_keys=('article', 'summary'),
            citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        path = dl_manager.manual_dir
        # To make split on the fly see  
        # https://github.com/tensorflow/datasets/blob/master/docs/splits.md
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(path, "pubmed_publisher_summ_collection.json")}),
        ]

    def _generate_examples(self, path):
        """Parse and yield pubmed_publisher_summ_collection.json for single document summarization"""
        with tf.io.gfile.GFile(path) as f:
            data = json.load(f)
            for i, pmid in enumerate(data):
                yield i, {
                    'article': data[pmid]['abstract'],
                    'title': data[pmid]['title'],
                    'summary': data[pmid]['summary'],
                    'pubdate': data[pmid]['pubdate'],
                    'pmid': pmid,
                }
