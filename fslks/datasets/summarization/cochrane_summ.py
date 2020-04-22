"""Module for creating Cochrane plain language summarization dataset"""

import os
import json

import tensorflow as tf
import tensorflow_datasets.public_api as tfds


_DESCRIPTION = """The Cochrane review articles and plain language summarization dataset,
for single document summarization.
"""

_CITATION ="""Nothing to cite because I stole the data from Wiley publishing"""

_COCHRANE_DOWNLOAD_INSTRUCTIONS = """Be very sneaky"""


class CochraneSumm(tfds.core.GeneratorBasedBuilder):
    """Cochrane plain language summarization dataset builder"""

    VERSION = tfds.core.Version("1.0.0")
    MANUAL_DOWNLOAD_INSTRUCTIONS = _COCHRANE_DOWNLOAD_INSTRUCTIONS

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'summary': tfds.features.Text(),
                'article': tfds.features.Text(),
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
                    "path": os.path.join(path, "cochrane_summary_train_collection.json")}),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    "path": os.path.join(path, "cochrane_summary_val_collection.json")}),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "path": os.path.join(path, "cochrane_summary_test_collection.json")}),
        ]

    def _generate_examples(self, path):
        """Parse and yield cochrane summaries"""
        with tf.io.gfile.GFile(path) as f:
            data = json.load(f)
            for i, key in enumerate(data):
                summary = data[key]['summary']
                article = data[key]['article']
                yield i, {
                    'summary': summary,
                    'article': article,
                }
