"""Module to building BioASQ tensorflow dataset
"""
import os
import json

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """BioASQ data for single and multi-document summarization""" 

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
                name="multi_doc",
                single_doc=False,
                #data_dir="",
                description="multi-document summarization"),
            BioasqConfig(
                name="single_doc",
                single_doc=True,
                description="single document summarization"),
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
                    name=tfds.Split.TRAIN,
                    gen_kwargs={
                        "path": os.path.join(path, "bioasq_train_collection.json")}),
                tfds.core.SplitGenerator(
                    name=tfds.Split.TEST,
                    gen_kwargs={
                        "path": os.path.join(path, "bioasq_test_collection.json")}),
            ]
        else:
            return [
                tfds.core.SplitGenerator(
                    name=tfds.Split.TRAIN,
                    gen_kwargs={
                        "path": os.path.join(path, "bioasq_train_collection.json")}),
            ]

    def _generate_examples(self, path=None):
        """Parse and yield BioASQ collection"""
        with tf.io.gfile.GFile(path) as f:
            bioasq_data = json.load(f)
            example_cnt = 0
            for example in bioasq_data:
                question = bioasq_data[example]['question']
                if self.builder_config.single_doc:
                    for snippet in bioasq_data[example]['snippets']:
                        example_cnt += 1
                        yield example_cnt, {
                            'article': snippet['article'],
                            'summary': snippet['snippet'],
                            'question': question,
                            'pmid': snippet['pmid'],
                        }
                else:
                    articles = []
                    pmids = []
                    example_cnt += 1
                    for snippet in bioasq_data[example]['snippets']:
                        if snippet['pmid'] not in pmids:
                            pmids.append(snippet['pmid'])
                            articles.append(snippet['article'])
                    yield example_cnt, {
                        'article': articles,
                        'summary': bioasq_data[example]['ideal_answer'],
                        'question': question,
                        'pmid': pmids,
                        }
