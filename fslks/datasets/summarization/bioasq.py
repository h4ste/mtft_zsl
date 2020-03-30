"""
Module to create BioASQ tensorflow dataset. 
"""
import os
import json

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """BioASQ dataset for summarization, available at http://bioasq.org/participate"""

_CITATION = """This work is 100% plagiarized"""

_BIOASQ_DOWNLOAD_INSTRUCTIONS = """OOH MAGIC"""


class Bioasq(tfds.core.GeneratorBasedBuilder):
    """BioASQ dataset builder"""

    VERSION = tfds.core.Version("1.0.0")
    MANUAL_DOWNLOAD_INSTRUCTIONS = _BIOASQ_DOWNLOAD_INSTRUCTIONS

    def __init__(self, single_document=True, config=None, version=None, data_dir=None):
        self.single_document = single_document
        super().__init__(data_dir=data_dir, config=config, version=version)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'article': tfds.features.Text(),
                'summary': tfds.features.Text(),
                'question': tfds.features.Text(),
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
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(path, "bioasq_collection.json")}),
        ]

    def _generate_examples(self, path=None):
        """Parse and yield bioasq_collection.json for single and multi-document summarization"""
        with tf.io.gfile.GFile(path) as f:
            bioasq_data = json.load(f)
            example_cnt = 0
            for example in bioasq_data:
                question = bioasq_data[example]['question']
                for snippet in bioasq_data[example]['snippets']:
                    example_cnt += 1
                    if self.single_document:
                        yield example_cnt, {
                            'article': snippet['article'],
                            'summary': snippet['snippet'],
                            'question': question,
                            'pmid': snippet['pmid'],
                        }
                    else:
                        # Might want to do this processing later on
                        raise NotImplementedError
