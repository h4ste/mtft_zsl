"""Module for creating Evidence-based Medicine dataset"""

import os
import json

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """Evidence-based Medicine dataset for summarization, available at https://sourceforge.net/projects/ebmsumcorpus/"""

_CITATION = """This work is 100% plagiarized"""

_EBM_DOWNLOAD_INSTRUCTIONS = """OOH MAGIC"""


class EBM(tfds.core.GeneratorBasedBuilder):
    """EBM dataset builder"""

    VERSION = tfds.core.Version("1.0.0")
    MANUAL_DOWNLOAD_INSTRUCTIONS = _EBM_DOWNLOAD_INSTRUCTIONS

    def __init__(self, single_document=True, config=None, version=None, data_dir=None):
        self.single_document = single_document
        super().__init__(data_dir=data_dir, config=config, version=version)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'summary': tfds.features.Text(),
                'question': tfds.features.Text(),
                'articles': tfds.features.Sequence(tfds.features.Text()),
                'pmids': tfds.features.Sequence(tfds.features.Text()),
            }),
            supervised_keys=('article', 'summary'),
            citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        path = dl_manager.manual_dir
        # No split. To make split on the fly see  
        # https://github.com/tensorflow/datasets/blob/master/docs/splits.md
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(path, "ebm_collection.json")}),
        ]

    def _generate_examples(self, path=None):
        """Parse and yield ebm_collection.json for multi-document summarization
        
        This could also be configured to do single document summarization with
        justifications as the summary and articles as the source.
        """
        # THIS IS A WEIRD DATASET!
        with tf.io.gfile.GFile(path) as f:
            data = json.load(f)
            example_cnt = 0
            for example in data:
                question = data[example]['question']
                # Multiple answers, each answer with multiple justifications
                # Here the answer will be the summary, and the multiple
                # justifications the source text. So one question will have
                # multiple answers in the dataset.
                for answer in data[example]['answers']:
                    answer_text = answer['answer_text']
                    justifications = []
                    pmids = []
                    example_cnt += 1
                    # Iterate through justifications, and take the abstract
                    # the justication text was taken from, not the justification
                    # text itself.
                    for justification in answer['justifications']:
                        for pmid in justification[1]:
                            justifications.append(justification[1][pmid])
                            pmids.append(pmid)
                    yield example_cnt, {
                            'articles': justifications,
                            'summary': answer_text,
                            'question': question,
                            'pmids': pmids,
                    }
