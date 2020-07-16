"""Module for creating Cochrane plain language summarization dataset"""

import os
import json

import tensorflow as tf
import tensorflow_datasets.public_api as tfds


_DESCRIPTION = """The Cochrane review articles and plain language summarization dataset,
for single document summarization or question answering. Each article contains 5 sections in a dictionary:
    background, objectives, methods, results, and discussion. 

The summary sub-dataset contains only plain-language summaries and review articles. The clinical answer sub-dataset contains
expert-level questions, answers, and a corresponding review article used to answer the question.

QA dataset statistics (tokens used as unit of measurement)
text             mean       stdev
----------  ---------  ----------
question      15.9255     3.84275
answer       174.763     85.2582
background  1227.23     682.146
objectives    52.4656    69.6018
methods     2605.58    1361.12
results     5831.86    4799.15
discussion  1821.24    1224.19

Summarization dataset statistics:
text             mean       stdev
----------  ---------  ---------
summary      314.826    173.549
background  1139.68     727.078
objectives    56.1298    91.1821
methods     2226.35    1285.17
results     4203.36    4436.25
discussion  1401.34    1079.82
"""

_CITATION ="""https://www.cochranelibrary.com/cdsr/reviews"""

_COCHRANE_DOWNLOAD_INSTRUCTIONS = """Export the citations available at https://www.cochranelibrary.com/cdsr/reviews
(for reviews) or https://www.cochranelibrary.com/cca (for clinical question-answering)
and use the provided urls to collect either the body of the text and the plain-language summary,
or the clinical questions and answers. Importantly, the QA pairs can be mapped to individual reviews used to create the answers.

Note that there is a known timeout-related bug in the export feature of the cochrane database, 
which prevents the user from downloading more than ~4000-5000 articles, depending on the connection
"""


class CochraneConfig(tfds.core.BuilderConfig):
    """Builder config for Cochrane"""

    @tfds.core.disallow_positional_args
    def __init__(self, 
                 **kwargs):
        """Config for Cochrane.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CochraneConfig, self).__init__(
            version=tfds.core.Version(
                "1.0.0"),
            supported_versions=[
                tfds.core.Version(
                    "1.0.0"),
            ],
            **kwargs)
        

class Cochrane(tfds.core.GeneratorBasedBuilder):
    """Cochrane Plain-Language Summarization and QA dataset builder"""

    VERSION = tfds.core.Version("1.0.0")
    MANUAL_DOWNLOAD_INSTRUCTIONS = _COCHRANE_DOWNLOAD_INSTRUCTIONS

    BUILDER_CONFIGS = [
            CochraneConfig(
                name="clinical_answer",
                description="Cochrane clinical question answering dataset"),
            CochraneConfig(
                name="summary",
                description="Cochrane summarization dataset"),
            ]

    def _info(self):
        print(vars(self))

        if self.builder_config.name == "clinical_answer":
            feature_dict = tfds.features.FeaturesDict({
                'question': tfds.features.Text(),
                'answer': tfds.features.Text(),
                'article': tfds.features.Text(),
                })
            data_keys = ('article', 'answer')
        elif self.builder_config.name == "summary":
            feature_dict = tfds.features.FeaturesDict({
                'summary': tfds.features.Text(),
                'article': tfds.features.Text(),
                })
            data_keys = ('article', 'summary')
        else:
            # Handle dataset names here
            raise IOError("Unknown dataset name. Please specify one of 'clinical_answer' or 'summary' for valid Cochrane dataset")
            
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=feature_dict,
            supervised_keys=data_keys,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        path = os.path.join(dl_manager.manual_dir, self.name)
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(path, "cochrane_{}_train_collection.json".format(self.builder_config.name))}),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    "path": os.path.join(path, "cochrane_{}_val_collection.json".format(self.builder_config.name))}),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "path": os.path.join(path, "cochrane_{}_test_collection.json".format(self.builder_config.name))}),
        ]

    def _generate_examples(self, path):
        """Parse and yield cochrane summaries"""
        with tf.io.gfile.GFile(path) as f:
            data = json.load(f)
            if self.builder_config.name == "summary":
                for i, key in enumerate(data):
                    summary = data[key]['summary']
                    article = ""
                    for section in data[key]['article']:
                        # Each article has a background, objective, methods, results, and discussion
                        article += data[key]['article'][section]
                    yield i, {
                        'summary': summary,
                        'article': article,
                    }
            else: 
                # QA data
                for i, key in enumerate(data):
                    answer = data[key]['answer']
                    article = ""
                    for section in data[key]['article']:
                        article += data[key]['article'][section]
                    question = data[key]['question']
                    yield i, {
                        'answer': answer,
                        'article': article,
                        'question': question,
                    }
