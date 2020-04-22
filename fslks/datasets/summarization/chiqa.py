"""
Module to create MEDIQA-Answer Summarization for Consumer Health Information Question Answering tensorflow dataset. 
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


class ChiqaConfig(tfds.core.BuilderConfig):
    """Builder config for MEDIQA-Answer Summarization"""

    @tfds.core.disallow_positional_args
    def __init__(self,
                 single_doc,
                 **kwargs):
        """Config for MEDIQA-Ans.

        Args:
          single_doc: `bool`, for single or multi-doc summarization
          **kwargs: keyword arguments forwarded to super.
        """
        super(ChiqaConfig, self).__init__(
            version=tfds.core.Version(
                "1.0.0"),
            supported_versions=[
                tfds.core.Version(
                    "1.0.0"),
            ],
            **kwargs)

        self.single_doc = single_doc


class Chiqa(tfds.core.GeneratorBasedBuilder):
    """MEDIQA-AnS dataset builder"""

    MANUAL_DOWNLOAD_INSTRUCTIONS = _MEDIQA_ANS_DOWNLOAD_INSTRUCTIONS

    BUILDER_CONFIGS = [
            ChiqaConfig(
                name="section2answer_multi_abstractive", 
                single_doc=False,
                description="multi-document, abstractive, section2answer summarization"),
            ChiqaConfig(
                name="page2answer_multi_abstractive", 
                single_doc=False,
                description="multi-document, abstractive, page2answer summarization"),
            ChiqaConfig(
                name="section2answer_multi_extractive", 
                single_doc=False,
                description="multi-document, extractive, section2answer summarization"),
            ChiqaConfig(
                name="page2answer_multi_extractive", 
                single_doc=False,
                description="multi-document, extractive, page2answer summarization"),
            ChiqaConfig(
                name="section2answer_single_abstractive", 
                single_doc=True,
                description="single-document, abstractive, section2answer summarization"),
            ChiqaConfig(
                name="page2answer_single_abstractive", 
                single_doc=True,
                description="single-document, abstractive, page2answer summarization"),
            ChiqaConfig(
                name="section2answer_single_extractive", 
                single_doc=True,
                description="single-document, extractive, section2answer summarization"),
            ChiqaConfig(
                name="page2answer_single_extractive", 
                single_doc=True,
                description="single-document, extractive, page2answer summarization"),
    ]

    def _info(self):
        if self.builder_config.single_doc:
            source_feature = tfds.features.Text() 
        else: 
            source_feature = tfds.features.Sequence(tfds.features.Text())

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'article': source_feature,
                'summary': tfds.features.Text(),
                'question': tfds.features.Text(),
            }),
            supervised_keys=('article', 'summary'),
            citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        path = os.path.join(dl_manager.manual_dir, self.name)
        return [
                tfds.core.SplitGenerator(
                    name=tfds.Split.TEST,
                    gen_kwargs={
                        "path": os.path.join(path, "{}_summ.json".format(self.builder_config.name))}),
        ]

    def _generate_examples(self, path=None):
        """Parse and yield MEDIQA-AnS collection"""
        with tf.io.gfile.GFile(path) as f:
            data = json.load(f)
            example_cnt = 0
            for example in data:
                question = data[example]['question']
                if self.builder_config.single_doc:
                    example_cnt += 1
                    yield example_cnt, {
                        'article': data[example]['articles'],
                        'summary': data[example]['summary'],
                        'question': question,
                    }
                else:
                    articles = []
                    example_cnt += 1
                    for answer_id in data[example]['articles']:
                        articles.append(data[example]['articles'][answer_id][0]),
                    yield example_cnt, {
                        'article': articles,
                        'summary': data[example]['summary'],
                        'question': question,
                        }
