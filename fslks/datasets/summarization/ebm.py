"""Module for creating Evidence-based Medicine dataset"""

import os
import json

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """Evidence-based Medicine dataset for multi-document summarization, available at https://sourceforge.net/projects/ebmsumcorpus/"""

_CITATION = """This work is 100% plagiarized"""

_EBM_DOWNLOAD_INSTRUCTIONS = """Download the Evidence-based-medicine dataset from https://sourceforge.net/projects/ebmsumcorpus/ and run the provided pre-processing script."""


class EBMConfig(tfds.core.BuilderConfig):
    """Builder config for EBM"""

    @tfds.core.disallow_positional_args
    def __init__(self, **kwargs):
        """Config for EBM.

        Args:
          **kwargs: keyword arguments forwarded to super.
        The name kwarg will be used to specify the task in the data processing
        """
        super(EBMConfig, self).__init__(
            version=tfds.core.Version(
                "1.0.0"),
            supported_versions=[
                tfds.core.Version(
                    "1.0.0"),
            ],
            **kwargs)


class EBM(tfds.core.GeneratorBasedBuilder):
    """EBM dataset builder"""

    MANUAL_DOWNLOAD_INSTRUCTIONS = _EBM_DOWNLOAD_INSTRUCTIONS

    BUILDER_CONFIGS = [
            EBMConfig(
                name="justify", 
                description="Generate justifications given question/answer/abstracts"),
            EBMConfig(
                name="answer",
                description="Generate answer given question/abstracts"),
            ]

    def _info(self):
        # Feature dict will need to be modified based on task
        feature_dict = {
                'summary': tfds.features.Text(),
                'question': tfds.features.Text(),
                'article': tfds.features.Sequence(tfds.features.Text()),
                'pmids': tfds.features.Sequence(tfds.features.Text()),
        }
        # Add key for answer text if doing justification
        if self.builder_config.name == "justify":
            feature_dict['answer'] = tfds.features.Text()

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(feature_dict),
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
                    "path": os.path.join(path, "ebm_train_collection.json")}),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    "path": os.path.join(path, "ebm_val_collection.json")}),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "path": os.path.join(path, "ebm_test_collection.json")}),
        ]

    def _generate_examples(self, path=None):
        """Parse and yield ebm_collection.json for multi-document summarization
        
        The answer task will treat the answer_text as a summary, and the 
        justifiy task will treat the justifications as a summary.
        """
        with tf.io.gfile.GFile(path) as f:
            data = json.load(f)
            example_cnt = 0
            for example in data:
                question = data[example]['question']
                # Multiple answers, each answer with multiple justifications
                # Here the answer will be the summary, and the references of the multiple
                # justifications will be the source text. So one question will have
                # multiple answers in the dataset.
                # All questions will have at least one answer
                for answer in data[example]['answers']:
                    answer_text = answer['answer_text']
                    # Some answers will have no justifications
                    if answer['justifications'] == []:
                        continue
                    # For this task iterate through justifications, and take the abstract
                    # the justication text was taken from, not the justification
                    # text itself. Use the answer text as the summary
                    if self.builder_config.name == "answer":
                        justification_abstracts = []
                        pmids = []
                        for justification in answer['justifications']:
                            # Multiple references for each justification
                            for pmid in justification[1]:
                                justification_abstracts.append(justification[1][pmid])
                                pmids.append(pmid)
                        # Some answers/justifications will have no reference texts
                        if justification_abstracts == []:
                            continue
                        example_cnt += 1
                        # Yield the list of justification abstracts and the answer text
                        yield example_cnt, {
                            'article': justification_abstracts,
                            'summary': answer_text,
                            'question': question,
                            'pmids': pmids,
                        }
                    # For this task, get the justification text as well, and make this the summary
                    elif self.builder_config.name == "justify":
                        justification_abstracts = []
                        pmids = []
                        for justification in answer['justifications']:
                            justification_text = justification[0]
                            for pmid in justification[1]:
                                justification_abstracts.append(justification[1][pmid])
                                pmids.append(pmid)
                            # Some justifications will have no abstracts
                            if justification_abstracts == []:
                                continue
                            # Yield text for each justification, NOT each answer
                            example_cnt += 1
                            yield example_cnt, {
                                'article': justification_abstracts,
                                'summary': justification_text,
                                'answer': answer_text,
                                'question': question,
                                'pmids': pmids,
                            }
                    else:
                        raise NotImplementedError("No valid task provided")
                    
