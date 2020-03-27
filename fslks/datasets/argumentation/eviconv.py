# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IBM Debater datasets."""
import csv
import os

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """
@inproceedings{shnarch-etal-2018-will,
    title = "Will it Blend? Blending Weak and Strong Labeled Data in a Neural Network for Argumentation Mining",
    author = "Shnarch, Eyal  and
              Alzate, Carlos  and
              Dankin, Lena  and
              Gleize, Martin  and
              Hou, Yufang  and
              Choshen, Leshem  and
              Aharonov, Ranit  and
              Slonim, Noam",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics 
                (Volume 2: Short Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P18-2095",
    doi = "10.18653/v1/P18-2095",
    pages = "599--605",
    abstract = "The process of obtaining high quality labeled data for natural language understanding tasks is often 
                slow, error-prone, complicated and expensive. With the vast usage of neural networks, this issue 
                becomes more notorious since these networks require a large amount of labeled data to produce 
                satisfactory results. We propose a methodology to blend high quality but scarce strong labeled data 
                with noisy but abundant weak labeled data during the training of neural networks. Experiments in the 
                context of topic-dependent evidence detection with two forms of weak labeled data show the advantages 
                of the blending scheme. In addition, we provide a manually annotated data set for the task of 
                topic-dependent evidence detection. We believe that blending weak and strong labeled data is a general 
                notion that may be applicable to many language understanding tasks, and can especially assist 
                researchers who wish to train a network but have a small amount of high quality labeled data for their 
                task of interest.",
}
"""

_DESCRIPTION = """
The data set contains 1,844 confirmed evidence taken from the data set of Shanrch et al. (2018). 
Out of these pieces of evidence 5,697 pairs were sampled. 
Each pair was annotated for the question of which evidence is more convincing.

The data set is split into 4,319 pairs for train and 1,378 for test (each is provided as a csv file).
Train set includes 48 topics and test set includes 21 other topics (i.e., no topic overlap between the two sets).

The dataset has 5 features:
  - topic: the debatable topic serving as context for the evidences.
  - evidence_1: a dictionary containing the following features:
    - text: the text of the evidence
    - stance: PRO for evidence supporting the topic, CON for evidence opposing the topic
    - id: an internal id of an evidence
    - wikipedia_article_name: the wikipedia article name from which the evidence was extracted
    - wikipedia_article_url: the wikipedia article's URL from which the evidence was extracted
  - evidence_2: a dictionary containing the following features:
    - text: the text of the evidence
    - stance: PRO for evidence supporting the topic, CON for evidence opposing the topic
    - id: an internal id of an evidence
    - wikipedia_article_name: the wikipedia article name from which the evidence was extracted
    - wikipedia_article_url: the wikipedia article's URL from which the evidence was extracted
  - label: 1 if evidence_1 is the more convincing one, 2 if evidence_2 is the more convincing.
  - acceptance_rate: the fraction of labelers which choose the final label.
"""

_TOPIC = "topic"
_EVIDENCE_1 = "evidence_1"
_EVIDENCE_2 = "evidence_2"
_EVIDENCE_TEXT = "text"
_EVIDENCE_STANCE = "stance"

_EVIDENCE_ARTICLE_NAME = "wikipedia_article_name"
_EVIDENCE_ARTICLE_URL = "wikipedia_article_url"
_LABEL = "label"
_ACCEPTANCE_RATE = "acceptance_rate"

_DOWNLOAD_URL = "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_EviConv-ACL-2019.v1.zip"


class EviConv(tfds.core.GeneratorBasedBuilder):
    """Scientific Papers."""
    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                _TOPIC: tfds.features.Text(),
                _EVIDENCE_1: tfds.features.FeaturesDict({
                    _EVIDENCE_TEXT: tfds.features.Text(),
                    _EVIDENCE_STANCE: tfds.features.ClassLabel(names=['PRO', 'CON']),
                    _EVIDENCE_ARTICLE_NAME: tfds.features.Text(),
                    _EVIDENCE_ARTICLE_URL: tfds.features.Text()
                }),
                _EVIDENCE_2: tfds.features.FeaturesDict({
                    _EVIDENCE_TEXT: tfds.features.Text(),
                    _EVIDENCE_STANCE: tfds.features.ClassLabel(names=['PRO', 'CON']),
                    _EVIDENCE_ARTICLE_NAME: tfds.features.Text(),
                    _EVIDENCE_ARTICLE_URL: tfds.features.Text()
                }),
                _LABEL: tfds.features.ClassLabel(num_classes=2),
                _ACCEPTANCE_RATE: tf.float32,
            }),
            supervised_keys=None,
            homepage="https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dl_dir = dl_manager.download_and_extract(_DOWNLOAD_URL)
        dl_dir = os.path.join(dl_dir, r'IBM_Debater_(R)_EviConv-ACL-2019.v1')
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"path": os.path.join(dl_dir, "train.csv")},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={"path": os.path.join(dl_dir, "test.csv")},
            ),
        ]

    def _generate_examples(self, path=None):
        """Yields examples."""
        with tf.io.gfile.GFile(path) as f:
            for i, row in enumerate(csv.DictReader(f)):
                yield str(i + 1), {
                    _TOPIC: row['topic'],
                    _EVIDENCE_1: {
                        _EVIDENCE_TEXT: row['evidence_1'],
                        _EVIDENCE_STANCE: row['evidence_1_stance'],
                        _EVIDENCE_ARTICLE_NAME: row['evidence_1_wikipedia_article_name'],
                        _EVIDENCE_ARTICLE_URL: row['evidence_1_wikipedia_url']
                    },
                    _EVIDENCE_2: {
                        _EVIDENCE_TEXT: row['evidence_2'],
                        _EVIDENCE_STANCE: row['evidence_2_stance'],
                        _EVIDENCE_ARTICLE_NAME: row['evidence_2_wikipedia_article_name'],
                        _EVIDENCE_ARTICLE_URL: row['evidence_2_wikipedia_url']
                    },
                    _LABEL: int(row['label']) - 1,
                    _ACCEPTANCE_RATE: float(row['acceptance_rate'])
                }
