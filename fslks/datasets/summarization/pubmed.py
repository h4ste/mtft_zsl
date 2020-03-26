"""
Module to create tensorflow dataset of publisher created summaries of articles in PubMed.
"""

import os
import glob

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """PubMed Publisher-Summary dataset, generated using 2020 PubMed baseline""" 

_CITATION = """This work is 100% plagiarized"""


class PubMedSumm(tfds.core.GeneratorBasedBuilder):
    """PubMed Publisher-Summary dataset builder"""

    VERSION = tfds.core.Version("1.0.0")

    def __init__(self):
        self.single_document = single_document

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'article': tfds.features.Text(),
                'summary': tfds.features.Text(),
                'pubdate': tfds.features.Text(),
                'pmid': tfds.features.Text(),
            }),
            supervised_keys=('article', 'summary'),
            citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # Not sure if this will work
        path = dl_manager.manual_dir
        # No BioASQ split. To make split on the fly see  
        # https://github.com/tensorflow/datasets/blob/master/docs/splits.md
        # Have to figure out how to get all data into one file?
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(path, "bioasq_collection.json")}),
        ]

    def _generate_examples(self, path):
        """Parse and yield bioasq_collection.json for single and multi-document summarization"""
        with tf.io.gfile.GFile(path) as f:
            bioasq_data = json.load(f)
            for i, example in enumerate(bioasq_data):
                question = example['question']
                for snippet in example['snippets']:
                    if self.single_document:
                        yield i, {
                            'article': snippet['article'],
                            'summary': snippet['snippet'],
                            'question': question,
                            'pmid': snippet['pmid'],
                        }
                    else:
                        # Might want to do this processing later on
                        raise NotImplementedError
