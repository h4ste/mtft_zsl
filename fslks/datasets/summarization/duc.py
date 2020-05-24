
"""Module to building DUC tensorflow datasets
"""
import os
import json

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """DUC data for single and multi-document summarization""" 

_CITATION = """This work is 100% plagiarized"""

_BIOASQ_DOWNLOAD_INSTRUCTIONS = """To download the BioASQ data, naviage to http://bioasq.org/. Register for an account there and download the training data for Task 8b. Unzip the downloaded directory and place it in X directory. Run the X.py script, which will then create a collection that can be parsed by the dataset builder in this module.""" 


class BioasqConfig(tfds.core.BuilderConfig):
    """Builder config for Bioasq"""

    @tfds.core.disallow_positional_args
    def __init__(self,
                 single_doc,
                 **kwargs):
        #data_dir,
        """Config for Bioasq.

        Args:
          single_doc: `bool`, specify single or multi-document summarization
          **kwargs: keyword arguments forwarded to super.
        """
        super(BioasqConfig, self).__init__(
            version=tfds.core.Version(
                "1.0.0"),
            supported_versions=[
                tfds.core.Version(
                    "1.0.0"),
            ],
            **kwargs)
        self.single_doc = single_doc


class Bioasq(tfds.core.GeneratorBasedBuilder):
    """BioASQ dataset builder"""

    MANUAL_DOWNLOAD_INSTRUCTIONS = _BIOASQ_DOWNLOAD_INSTRUCTIONS

    BUILDER_CONFIGS = [
            BioasqConfig(
                name="duc_2004",
                description="DUC 2004 single document summarization"),
            BioasqConfig(
                name="duc_2007",
                description="DUC 2007 multi-document summarization"),
            ]

    def _info(self):
        if self.builder_config.single_doc:
            source_feature = tfds.features.Text() 
            pmid_feature = tfds.features.Text() 
        else: 
            source_feature = tfds.features.Sequence(tfds.features.Text())
            pmid_feature = tfds.features.Sequence(tfds.features.Text())

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'article': source_feature,
                'summary': tfds.features.Text(),
                'question': tfds.features.Text(),
                'pmid': pmid_feature,
            }),
            supervised_keys=('article', 'summary'),
            citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        path = os.path.join(dl_manager.manual_dir, self.name)
        # There is not a test set for multi-doc summarization, 
        # because ideal summaries are not provided by BioASQ 
        # in test data
        if self.builder_config.single_doc:
            return [
                tfds.core.SplitGenerator(
                        "path": os.path.join(path, "bioasq_train_collection.json")}),
                tfds.core.SplitGenerator(
                    name=tfds.Split.TEST,
                    gen_kwargs={
                        "path": os.path.join(path, "bioasq_test_collection.json")}),
            ]

    def _generate_examples(self, path=None):
        """Parse and yield DUC collection"""
        with tf.io.gfile.GFile(path) as f:
            duc_data = json.load(f)
            example_cnt = 0
            for topic_id in duc_data:
                if self.builder_config.name == "duc_2004":
                    for topic in duc_data[topic_id]:
                        example_cnt += 1
                        yield example_cnt, {
                            'document': topic['document'],
                            'summary': topic['summary'],
                        }
                else:
                    documents = []
                    example_cnt += 1
                    question = duc_data[topic_id]['question']
                    for summary in duc_data[topic_id]['summaries']:
                        yield example_cnt, {
                            'article': summary['documents'],
                            'summary': summary['summary']
                            'question': question,
                            }
