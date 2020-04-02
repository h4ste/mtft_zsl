from fslks.run_experiment import Predictions
from fslks import eval

import abc
import importlib

from tabulate import tabulate


class Evaluator(abc.ABC):

    @abc.abstractmethod
    def evaluate(self, predictions: Predictions) -> str:
        pass


class BasicEvaluator(Evaluator):

    def evaluate(self, predictions: Predictions) -> str:
        headers = ['Task', 'Split', 'W. Acc.', 'BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        results = []

        for task, task_predictions in predictions.items():
            for split, split_predictions in task_predictions.items():
                targets = split_predictions['target_tokens']
                predictions = split_predictions['pred_tokens']
                w_acc = eval.word_accuracy(targets, predictions)
                bleus = eval.bleu(targets, predictions)
                rouges = eval.rouge(targets, predictions)
                results.append([task, split,
                                w_acc,
                                bleus[0] * 100.,
                                rouges['rouge_1/f_score'] * 100.,
                                rouges['rouge_2/f_score'] * 100.,
                                rouges['rouge_l/f_score'] * 100.])

        return tabulate(results, headers=headers)


class NlgEvaluator(Evaluator):

    def __init__(self):
        nlg_eval = importlib.import_module('nlgeval')
        self.nlg = nlg_eval.NLGEval()

    def evaluate(self, predictions: Predictions) -> str:
        headers = ['Task', 'Split',
                   'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4',
                   'ROUGE-L', 'METEOR', 'CIDEr', 'STCS', 'EACS', 'VECS', 'GMS']
        results = []

        for task, task_predictions in predictions.items():
            for split, split_predictions in task_predictions.items():
                targets = split_predictions['target_tokens']
                predictions = split_predictions['pred_tokens']

                metrics = self.nlg.compute_metrics(targets, predictions)
                results.append([task, split,
                                metrics['Bleu-1'],
                                metrics['Bleu-2'],
                                metrics['Bleu-3'],
                                metrics['Bleu-4'],
                                metrics['ROUGLE_L'],
                                metrics['METEOR'],
                                metrics['CIDEr'],
                                metrics['SkipThoughtsCosineSimilarity'],
                                metrics['EmbeddingAverageCosineSimilarity'],
                                metrics['VectorExtremaCosineSimilarity'],
                                metrics['GreedyMatchingScore']])

        return tabulate(results, headers=headers)

