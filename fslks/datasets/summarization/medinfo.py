"""Module for creating MedInfo dataset"""

import os
import json

import tensorflow as tf
import tensorflow_datasets.public_api as tfds


_DESCRIPTION = """The MedInfo single document summarization dataset, built using MedInfo collection
available from https://github.com/abachaa/Medication_QA_MedInfo2019.
"""

_CITATION = """@inproceedings{BenAbacha:MEDINFO19, 
       author    = {Asma {Ben Abacha} and Yassine Mrabet and Mark Sharp and
   Travis Goodwin and Sonya E. Shooshan and Dina Demner{-}Fushman},    
       title     = {Bridging the Gap between Consumers Medication Questions and Trusted Answers}, 
       booktitle = {MEDINFO 2019},   
       year      = {2019},
    }
"""

_MEDINFO_DOWNLOAD_INSTRUCTIONS = """Link to medinfo and provide processing script? Or link to my github where I've done the article scraping, etc"""


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
                'article': tfds.features.Text(),
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
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(path, "medinfo_train_collection.json")}),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    "path": os.path.join(path, "medinfo_validation_collection.json")}),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "path": os.path.join(path, "medinfo_test_collection.json")}),
        ]

    def _generate_examples(self, path):
        """Parse and yield medinfo for single document summarization"""
        with tf.io.gfile.GFile(path) as f:
            data = json.load(f)
            for i, key in enumerate(data):
                summary = data[key]['summary']
                article = data[key]['articles']
                question = data[key]['question']
                yield i, {
                    'summary': summary,
                    'article': article,
                    'question': question,
                }
