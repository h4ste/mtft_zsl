"""
Script for processing BioASQ json data for the Multitask finetuning Zero Shot Learning for Consumer Health Answer Summarization code. 

To run:
python process_bioasq.py --train_path=<path to main bioasq json> --test_path=<path to directory with test datasets> --article_dir=<path to directory containing train and test articles>. 

To access the BioASQ data, you will need to register at http://bioasq.org/participate. We have provided all the train and testing PubMed articles at https://bionlp.nlm.nih.gov so that you don't need to use E-utils to download them from PubMed yourself. 

The training input path will be "your_dir/BioASQ-training8b/training8b.json", where the input file should
that which is provided by BioASQ. Once you unzip the BioASQ data, you will see the file.

The test input path will be to a directory that contains all the test sets for the year of the task you are using. The script will iterate over the test sets and combine the data.

Store both the train and test articles sets in the same directory and provide the path to that directory with --article_dir

For example:
python process_bioasq.py \
    --train_path=/home/data/bioasq/BioASQ-training8b/training8b.json  \
    --test_path=/home/data/bioasq/data/test_sets \
    --article_dir=/home/data/bioasq_articles
"""


import json
import sys
import os
import argparse
import glob
from collections import Counter


def get_args():
    """
    Get command line arguments
    """

    parser = argparse.ArgumentParser(description="Arguments for data exploration")
    parser.add_argument("--train_path",
                        dest="train_path",
                        help="Path to unzipped training bioasq data")
    parser.add_argument("--test_path",
                        dest="test_path",
                        help="Path to unzipped training bioasq data")
    parser.add_argument("--article_dir",
                        dest="article_dir",
                        help="Path to directory containing bioasq train and test PubMed articles")
    return parser


class BioASQ():
    """
    Class for processing and saving BioASQ data
    """

    def load_bioasq(self, split, args):
        """
        Load bioasq dataset
        """
        if split == "train":
            with open(args.train_path, "r", encoding="ascii") as f:
                bioasq_questions = json.load(f)['questions']
        elif split == "test":
            test_sets = []
            dataset_path = os.path.join(args.test_path, "*")
            for dataset in glob.iglob(dataset_path):
                with open(dataset, "r", encoding="utf-8") as f:
                    test_sets.append(json.load(f)['questions'])
            # Combine test sets, which are a list of questions
            bioasq_questions = []
            for test_set in test_sets:
                for question in test_set:
                    bioasq_questions.append(question)
        return bioasq_questions

    def prepare_bioasq(self, split, args, bioasq_questions):
        """
        Process BioASQ training data.
        Save questions, ideal answers, snippets, articles, and question types.
        """
        print("Processing downloaded articles and BioASQ data for {} split".format(split))

        article_path = os.path.join(args.article_dir, "bioasq_{}_pubmed_articles.json".format(split))
        with open(article_path, "r", encoding="ascii") as f:
            articles = json.load(f)
        # Dictionary to save condensed json of bioasq
        bioasq_collection = {}
        questions = []
        ideal_answers = []
        ideal_answer_dict = {}
        exact_answers = []
        snippet_dict = {}
        for i, q in enumerate(bioasq_questions):
            # Get the question
            bioasq_entry = {}
            bioasq_entry['question'] = q['body']
            questions.append(q['body'])
            # Get the references used to answer that question
            pmid_list= [d.split("/")[-1] for d in q['documents']]
            # Get the question type: list, summary, yes/no, or factoid
            q_type = q['type']
            bioasq_entry['q_type'] = q_type
            # The test set will not have ideal or exact answers included unfortunately.
            if split == "train":
                # Take the first ideal answer
                assert isinstance(q['ideal_answer'], list)
                assert isinstance(q['ideal_answer'][0], str)
                ideal_answer_dict[i] = q['ideal_answer'][0]
                bioasq_entry['ideal_answer'] = q['ideal_answer'][0]
                # And get the first exact answer
                if q_type != "summary":
                    # Yesno questions will have just a yes/no string in exact answer.
                    if q_type == "yesno":
                        exact_answers.append(q['exact_answer'][0])
                        bioasq_entry['exact_answer'] = q['exact_answer'][0]
                    else:
                        if isinstance(q['exact_answer'], str):
                            exact_answers.append(q['exact_answer'])
                            bioasq_collection[i]['exact_answer'] = q['exact_answer']
                        else:
                            exact_answers.append(q['exact_answer'][0])
                            bioasq_entry['exact_answer'] = q['exact_answer'][0]
            # Then handle the snippets (the text extracted from the abstract)
            bioasq_entry['snippets'] = []
            snippet_dict[q['body']] = []
            pmid_dict = {}
            unique_abs_index = 0
            for snippet in q['snippets']:
                pmid_match = False
                snippet_dict[q['body']].append(snippet['text'])
                doc_pmid = str(snippet['document'].split("/")[-1])
                try:
                    article = articles[doc_pmid]
                    # Add the data to the dictionary containing the collection.
                    bioasq_entry['snippets'].append({'snippet': snippet['text'], 'article': article, 'pmid': doc_pmid})
                except KeyError as e:
                    continue

                # Don't add if there are no snippet/abstract pairs
                if bioasq_entry['snippets'] == []:
                    continue
                else:
                    bioasq_collection[i] = bioasq_entry

        with open("bioasq_{0}_collection.json".format(split), "w", encoding="utf8") as f:
            json.dump(bioasq_collection, f, indent=4)


def process_bioasq():
    """
    Main processing function for bioasq data
    """
    args = get_args().parse_args()
    bq = BioASQ()
    for split in ["train", "test"]:
        data = bq.load_bioasq(split, args)
        bq.prepare_bioasq(split, args, data)


if __name__ == "__main__":
    process_bioasq()
