"""
Module to create tensorflow dataset of publisher created summaries of articles in PubMed.
"""

import os
import glob
import json

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """PubMed Publisher-Summary dataset, generated using 2020 PubMed baseline""" 

_CITATION = """None"""

_PUBMED_DOWNLOAD_INSTRUCTIONS = """This data set can be created using e-utils and parsing the text associated with nodes with the plain-language-summary attribute available in the PubMed XML."""

class PubmedSumm(tfds.core.GeneratorBasedBuilder):
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
        path = os.path.join(dl_manager.manual_dir, self.name)
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(path, "pubmed_pubsumm_train_collection.json")}),
                tfds.core.SplitGenerator(
                    name=tfds.Split.TEST,
                    gen_kwargs={
                        "path": os.path.join(path, "pubmed_pubsumm_test_collection.json")}),
        ]

    def _generate_examples(self, path):
        """Parse and yield pubmed publisher summaries for single document summarization"""
        with tf.io.gfile.GFile(path) as f:
            data = json.load(f)
            for i, pmid in enumerate(data):
                yield i, {
                    'article': data[pmid]['abstract'].strip(),
                    'title': data[pmid]['title'].strip(),
                    'summary': data[pmid]['summary'].strip(),
                    'pubdate': data[pmid]['pubdate'],
                    'pmid': pmid,
                }
