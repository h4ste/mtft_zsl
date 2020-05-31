import sys

import numpy as np

from fslks.experiments import Predictions
from fslks import eval

import abc
import importlib

from tabulate import tabulate

import logging

EMPTY_PREDICTION = '_UNK'


class Evaluator(abc.ABC):

    @abc.abstractmethod
    def evaluate(self, predictions: Predictions, limit: int) -> str:
        pass


class BasicEvaluator(Evaluator):

    def evaluate(self, predictions: Predictions, limit: int) -> str:
        headers = ['Task', 'Split', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU-4']
        results = []

        for task, task_predictions in predictions.items():
            for split, split_predictions_fn in task_predictions.items():
                split_predictions = split_predictions_fn()
                targets = [*map(str.split, split_predictions['targets'])]
                predictions = [*map(str.split, split_predictions['predictions'])]
                assert len(targets) == len(predictions), \
                    "%s[%s] only had %d predictions for %d targets!" % (task, split, len(targets), len(predictions))
                for idx, (target_, prediction_) in enumerate(zip(targets, predictions)):
                    assert len(target_) > 0, "Targets were empty for %s[%s] #%d" % (task, split, idx)
                    if len(prediction_) == 0:
                        logging.warning("Predictions were empty for %s[%s] #%d, setting predictions to [_UNK]",
                                        task, split, idx)
                        predictions[idx] = [EMPTY_PREDICTION]
                w_acc = eval.word_accuracy(references=targets, predictions=predictions)
                bleus = eval.bleu(target_corpus=targets, predicted_corpus=predictions)
                rouges = eval.rouge(references=targets, hypotheses=predictions)
                results.append([task, split,
                                rouges['rouge_1/r_score'] * 100.,
                                rouges['rouge_2/r_score'] * 100.,
                                rouges['rouge_l/r_score'] * 100.,
                                bleus[0] * 100.,
                                ])

        return tabulate(results, headers=headers)


class NlgEvaluator(Evaluator):

    def __init__(self, nlg):
        self.nlg = nlg

    def evaluate(self, predictions: Predictions, limit: int = sys.maxsize) -> str:
        headers = ['Task', 'Split',
                   'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4',
                   'ROUGE-L', 'METEOR', 'CIDEr', 'STCS', 'EACS', 'VECS', 'GMS']
        results = []

        for task, task_predictions in predictions.items():
            for split, split_predictions_fn in task_predictions.items():
                split_predictions = split_predictions_fn()
                prompts = split_predictions['prompts']
                targets = split_predictions['targets']
                predictions_ = split_predictions['predictions']

                prompt_targets = {}
                prompt_predictions = {}
                for prompt_, prediction_, target_ in zip(prompts, predictions_, targets):
                    if task.startswith('duc/2007') or task.startswith('tac'):
                        prompt_ = prompt_.split('summarize:')[0].strip()
                    if prompt_ not in prompt_targets:
                        assert prompt_ not in prompt_predictions
                        prompt_targets[prompt_] = []
                        prompt_predictions[prompt_] = prediction_.strip()
                    prompt_targets[prompt_].append(target_.strip())

                references = []
                hypotheses = []
                for idx, prompt in enumerate(prompt_predictions.keys()):
                    target = prompt_targets[prompt]
                    assert len(target) > 0, "Targets were empty for %s[%s] #%d" % (task, split, idx)
                    prediction = prompt_predictions[prompt]
                    if len(prediction) == 0:
                        logging.warning("Predictions were empty for %s[%s] #%d, setting predictions to [_UNK]",
                                        task, split, idx)
                        prediction = EMPTY_PREDICTION
                    references.append(target)
                    hypotheses.append(prediction)
                    if idx > limit:
                        break
                    # logging.info('References: %s', target)
                    # logging.info('Hypothesis: %s', prediction)
                    # logging.info('Individual metrics: %s', self.nlg.compute_individual_metrics(target, prediction))

                logging.info('Len(references) = %d; Len(hypotheses) = %d', len(references), len(hypotheses))

                metrics = self.nlg.compute_metrics(list(zip(*references)), hypotheses)
                results.append([task, split,
                                metrics['Bleu_1'] * 100.,
                                metrics['Bleu_2'] * 100.,
                                metrics['Bleu_3'] * 100.,
                                metrics['Bleu_4'] * 100.,
                                metrics['ROUGE_L'] * 100.,
                                metrics['METEOR'] * 100.,
                                metrics['CIDEr'] * 100.,
                                # metrics['SkipThoughtCS'] * 100.,
                                metrics['EmbeddingAverageCosineSimilarity'] * 100.,
                                metrics['VectorExtremaCosineSimilarity'] * 100.,
                                metrics['GreedyMatchingScore'] * 100.])

        return tabulate(results, headers=headers)


def auto_evaluate(predictions: Predictions) -> str:
    evaluator: Evaluator

    try:
        nlg_eval = importlib.import_module('nlgeval')
        evaluator = NlgEvaluator(nlg=nlg_eval.NLGEval())
    except ModuleNotFoundError:
        evaluator = BasicEvaluator()

    return evaluator.evaluate(predictions)


def get_evaluator(evaluator: str) -> Evaluator:
    if evaluator == 'basic':
        evaluator = BasicEvaluator()
    elif evaluator == 'nlg':
        nlg_eval = importlib.import_module('nlgeval')
        evaluator = NlgEvaluator(nlg=nlg_eval.NLGEval(no_skipthoughts=True))
    else:
        raise NotImplementedError('Unsupported evaluator \"' + evaluator + "\"")
    return evaluator
