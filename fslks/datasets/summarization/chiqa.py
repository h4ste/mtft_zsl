"""
Module to create MEDIQA-Answer Summarization tensorflow dataset. 
"""
import os
import json

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """MEDIQA-AnS dataset for summarization, available at https://osf.io/fyg46/. 
Supports eight summarization tasks including abstract and extractive, long document and 
short document, and multi and single document. See README at OSF url for further details.
"""

_CITATION = """This work is 100% plagiarized"""

_MEDIQA_ANS_DOWNLOAD_INSTRUCTIONS = """Temp until upload""" 


class AnsConfig(tfds.core.BuilderConfig):
    """Builder config for MEDIQA-Answer Summarization"""

    @tfds.core.disallow_positional_args
    def __init__(self,
                 single_doc,
                 extractive,
                 page2answer,
                 **kwargs):
        """Config for MEDIQA-Ans.

        Args:
          single_doc: `bool`, for single or multi-doc summarization
          extractive: `bool`, for extractive or abstractive summarization
          page2answer: `bool`, for long or short document (pages or passages) summarization
          **kwargs: keyword arguments forwarded to super.
        """
        super(AnsConfig, self).__init__(
            version=tfds.core.Version(
                "2.1.0"),
            supported_versions=[
                tfds.core.Version(
                    "2.1.0"),
            ],
            **kwargs)

        self.single_doc = single_doc
        self.page2answer = page2answer
        self.extractive = extractive


class Ans(tfds.core.GeneratorBasedBuilder):
    """MEDIQA-AnS dataset builder"""

    MANUAL_DOWNLOAD_INSTRUCTIONS = _MEDIQA_ANS_DOWNLOAD_INSTRUCTIONS

    BUILDER_CONFIGS = [
            BioasqConfig(
                name="multi-abs-s2a", 
                single_doc=False,
                extractive=False,
                page2answer=False,
                description="multi-document, abstractive, section2answer summarization"),
            BioasqConfig(
                name="multi-abs-p2a", 
                single_doc=False,
                extractive=False,
                page2answer=True,
                description="multi-document, abstractive, page2answer summarization"),
            BioasqConfig(
                name="multi-ext-s2a", 
                single_doc=False,
                extractive=True,
                page2answer=False,
                description="multi-document, extractive, section2answer summarization"),
            BioasqConfig(
                name="multi-ext-p2a", 
                single_doc=False,
                extractive=True,
                page2answer=True,
                description="multi-document, extractive, page2answer summarization"),
            BioasqConfig(
                name="single-abs-s2a", 
                single_doc=True,
                extractive=False,
                page2answer=False,
                description="single-document, abstractive, section2answer summarization"),
            BioasqConfig(
                name="single-abs-p2a", 
                single_doc=True,
                extractive=False,
                page2answer=True,
                description="single-document, abstractive, page2answer summarization"),
            BioasqConfig(
                name="single-ext-s2a", 
                single_doc=True,
                extractive=True,
                page2answer=False,
                description="single-document, extractive, section2answer summarization"),
            BioasqConfig(
                name="single-ext-p2a", 
                single_doc=True,
                extractive=True,
                page2answer=True,
                description="single-document, extractive, page2answer summarization"),
    ]

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
        return [
                tfds.core.SplitGenerator(
                    name=tfds.Split.TEST,
                    gen_kwargs={
                        "path": os.path.join(path, "mediqa_ans_test_collection.json")}),
        ]

    def _generate_examples(self, path=None):
        """Parse and yield bioasq_collection.json for single and multi-document summarization"""
        #TODO: Convert bioasq to mediqans parsing 
        with tf.io.gfile.GFile(path) as f:
            data = json.load(f)
            example_cnt = 0
            for example in data:
                question = data[example]['question']
                if self.builder_config.single_doc:
                    for article in data[example]['articles']:
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
