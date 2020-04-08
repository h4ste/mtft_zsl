"""
Module to create MEDIQA-Answer Summarization tensorflow dataset. 
"""
import os
import json

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """MEDIQA-AnS dataset for summarization, available at https://osf.io/fyg46/. See README there for further details.""" 

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
          data_dir: `string`, the path to the directory containing the Bioasq collection.
          **kwargs: keyword arguments forwarded to super.
        """
        super(BioasqConfig, self).__init__(
            version=tfds.core.Version(
                "2.1.0"),
            supported_versions=[
                tfds.core.Version(
                    "2.1.0"),
            ],
            **kwargs)
        # Currently don't need data_dir; leaving it in just in case
        #self.data_dir = data_dir
        self.single_doc = single_doc


class Bioasq(tfds.core.GeneratorBasedBuilder):
    """BioASQ dataset builder"""

    MANUAL_DOWNLOAD_INSTRUCTIONS = _BIOASQ_DOWNLOAD_INSTRUCTIONS

    BUILDER_CONFIGS = [
            BioasqConfig(
                name="multi-doc", 
                single_doc=False,
                #data_dir="",
                description="multi-document summarization"),
            BioasqConfig(
                name="single-doc",
                single_doc=True,
                #data_dir="",
                description="single document summarization"),
            ]

    #def __init__(self, single_document=True, config=None, version=None, data_dir=None):
    #    self.single_document = single_document
    #    super().__init__(data_dir=data_dir, config=config, version=version)

    def _info(self):
        # Article will be single text feature fr single doc
        # If running multi-doc articles will be provided in a list
        # Similary for pmids
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
        path = dl_manager.manual_dir
        #path = os.path.join(dl_manager.manual_dir, self.builder_config.data_dir)
        # There is not test set for multi-doc summarization, 
        # because ideal summaries are not provided by BioASQ 
        # in test data..
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
        """Parse and yield bioasq_collection.json for single and multi-document summarization"""
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
