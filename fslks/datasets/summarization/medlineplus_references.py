"""
Module to create MedlinePlus PubMed review summarization tensorflow dataset. 
"""
import os
import json

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """The MedlinePlus multi-document reference summarization dataset, built using articles at available in in the Start Here and Learn More sections in the xml at https://medlineplus.gov/xml.html. The urls for these articles
obtained from the XML and the articles were then crawled from the web
"""

_CITATION = """This work is 100% plagiarized"""

_MEDLINEPLUS_REF_DOWNLOAD_INSTRUCTIONS = """Do stuff here. Or don't. Who cares."""


class MedlinePlusReferences(tfds.core.GeneratorBasedBuilder):
    """MedlinePlus References (Start Here and Learn More Sections) summarization dataset builder"""

    VERSION = tfds.core.Version("1.0.0")
    MANUAL_DOWNLOAD_INSTRUCTIONS = _MEDLINEPLUS_REF_DOWNLOAD_INSTRUCTIONS

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'summary': tfds.features.Text(),
                'medlineplus_url': tfds.features.Text(),
                'articles': tfds.features.Sequence(tfds.features.Text()),
                'reference_urls': tfds.features.Sequence(tfds.features.Text()),
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
                    "path": os.path.join(path, "medlineplus_reference_collection.json")}),
        ]

    def _generate_examples(self, path):
        """Parse and yield medlineplus_reference_collection.json for multi-document summarization"""
        with tf.io.gfile.GFile(path) as f:
            data = json.load(f)
            for i, mp_url in enumerate(data):
                summary = data[mp_url]['summary']
                articles = []
                urls = []
                for ref_url in data[mp_url]['articles']:
                    urls.append(ref_url)
                    articles.append(data[mp_url]['articles'][ref_url])
                yield i, {
                    'summary': summary,
                    'medlineplus_url': mp_url,
                    'articles': articles,
                    'reference_urls': urls,
                }