
"""Module to building DUC tensorflow datasets
"""
import os
import json

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """DUC data for single and multi-document summarization""" 

_CITATION = """DUC 2004 and 2007"""

_DUC_DOWNLOAD_INSTRUCTIONS = """Contact NIST to acquire data from prior DUC tasks""" 


class DucConfig(tfds.core.BuilderConfig):
    """Builder config for DUC"""

    @tfds.core.disallow_positional_args
    def __init__(self,
                 **kwargs):
        #data_dir,
        """Config for DUC.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DucConfig, self).__init__(
            version=tfds.core.Version(
                "1.0.0"),
            supported_versions=[
                tfds.core.Version(
                    "1.0.0"),
            ],
            **kwargs)


class Duc(tfds.core.GeneratorBasedBuilder):
    """DUC dataset builder"""

    MANUAL_DOWNLOAD_INSTRUCTIONS = _DUC_DOWNLOAD_INSTRUCTIONS

    BUILDER_CONFIGS = [
            DucConfig(
                name="2004",
                description="DUC 2004 single document summarization"),
            DucConfig(
                name="2007",
                description="DUC 2007 multi-document summarization"),
            ]

    def _info(self):
        if self.builder_config.name == "2004":
            source_feature = tfds.features.Text() 
            pmid_feature = tfds.features.Text() 
        else: 
            # For DUC 2007 multi-doc
            source_feature = tfds.features.Sequence(tfds.features.Text())

        feature_duct = {
                'document': source_feature,
                'summary': tfds.features.Text(),
                }

        if self.builder_config.name == "2007":
                feature_duct['question'] = tfds.features.Text()

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(feature_duct),
            supervised_keys=('document', 'summary'),
            citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        path = os.path.join(dl_manager.manual_dir, self.name)
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "path": os.path.join(path, "duc_{}_test_collection.json".format(self.builder_config.name))}),
        ]

    def _generate_examples(self, path=None):
        """Parse and yield DUC collection"""
        with tf.io.gfile.GFile(path) as f:
            duc_data = json.load(f)
            example_cnt = 0
            for topic_id in duc_data:
                if self.builder_config.name == "2004":
                    example_cnt += 1
                    yield example_cnt, {
                        'document': duc_data[topic_id]['document'],
                        'summary': duc_data[topic_id]['summary'],
                    }
                else:
                    # DUC 2007
                    question = duc_data[topic_id]['question']
                    # In the 'summaries' value, there are 4 summaries + 4 sets of documents
                    # The 4 summaries were created by 4 different annotators for each topic, and have keys A-J
                    # Documents have already been sorted based on n-gram matches to summary and added to list in preprocessing.
                    for summary in duc_data[topic_id]['summaries'].values():
                        example_cnt += 1
                        yield example_cnt, {
                            'document': summary['documents'],
                            'summary': summary['summary'],
                            'question': question,
                            }
