"""
Module to create MedlinePlus PubMed review summarization tensorflow dataset. 
"""
import os
import json

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """The MedlinePlus multi-document review summarization dataset, built using articles at available in xml at https://medlineplus.gov/xml.html and PubMed review articles listed relevant to the article.""" 

_CITATION = """This work is 100% plagiarized"""

_MEDLINEPLUS_REVIEW_DOWNLOAD_INSTRUCTIONS = """Do stuff here. Or don't. Who cares."""

class MedlinePlusReviews(tfds.core.GeneratorBasedBuilder):
    """MedlinePlus review summarization dataset builder"""

    VERSION = tfds.core.Version("1.0.0")
    MANUAL_DOWNLOAD_INSTRUCTIONS = _MEDLINEPLUS_REVIEW_DOWNLOAD_INSTRUCTIONS

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'summary': tfds.features.Text(),
                'url': tfds.features.Text(),
                'articles': tfds.features.Sequence(tfds.features.Text()),
                'pmids': tfds.features.Sequence(tfds.features.Text()),
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
                    "path": os.path.join(path, "medlineplus_review_collection.json")}),
        ]

    def _generate_examples(self, path):
        """Parse and yield medlineplus_review_collection.json for multi-document summarization"""
        with tf.io.gfile.GFile(path) as f:
            data = json.load(f)
            for i, url in enumerate(data):
                summary = data[url]['summary']
                articles = []
                pmids = []
                for pmid in data[url]['reviews']:
                    pmids.append(pmid)
                    articles.append(data[url]['reviews'][pmid])
                yield i, {
                    'summary': summary,
                    'url': url,
                    'articles': articles,
                    'pmids': pmids,
                }
