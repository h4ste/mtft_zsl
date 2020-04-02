"""Module for creating MedInfo dataset"""

import os
import json

import tensorflow as tf
import tensorflow_datasets.public_api as tfds


_DESCRIPTION = """The MedInfo single document summarization dataset, built using MedInfo collection
available from https://github.com/abachaa/Medication_QA_MedInfo2019.
"""

_CITATION = """This work is 100% plagiarized"""

_MEDINFO_DOWNLOAD_INSTRUCTIONS = """Do stuff here. Or don't. Who cares."""


class Medinfo(tfds.core.GeneratorBasedBuilder):
    """MedInfo summarization dataset builder"""

    VERSION = tfds.core.Version("1.0.0")
    MANUAL_DOWNLOAD_INSTRUCTIONS = _MEDINFO_DOWNLOAD_INSTRUCTIONS

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'summary': tfds.features.Text(),
                'articles': tfds.features.Text(),
                'question': tfds.features.Text(),
                }),
            supervised_keys=('articles', 'summary'),
            citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        path = dl_manager.manual_dir
        # Make split on the fly: https://github.com/tensorflow/datasets/blob/master/docs/splits.md
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(path, "medinfo_section2answer_collection.json")}),
        ]

    def _generate_examples(self, path):
        """Parse and yield medinfo_section2answer_collection.json for single document summarization"""
        with tf.io.gfile.GFile(path) as f:
            data = json.load(f)
            for i, key in enumerate(data):
                summary = data[key]['summary']
                articles = data[key]['articles']
                question = data[key]['question']
                yield i, {
                    'summary': summary,
                    'articles': articles,
                    'question': question,
                }


