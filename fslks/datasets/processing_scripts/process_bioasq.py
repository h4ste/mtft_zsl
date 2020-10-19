"""
Script for processing BioASQ json data and saving

To run:
python process_bioasq.py --train_path=<path to main bioasq json> --test_path=<path to directory with test datasets>. The script will download the PubMed articles for the version of BioASQ you are using. This can take ~20-30 minutes, as there > 30,000 articles. To access the BioASQ data, you will need to register at http://bioasq.org/participate. 

The training input path will be something like "your_dir/BioASQ-training8b/training8b.json", where the input file should
that which is provided by BioASQ.

The test input path will be to a directory that contains all the test sets for the year of the task you are using. The script will iterate over the test sets and combine the data.

For example:
python process_bioasq.py --train_path=/home/data/bioasq/BioASQ-training8b/training8b.json --test_path=/home/data/bioasq/data/test_sets
"""


import json
import sys
import os
import argparse
import lxml.etree as le
import glob
from collections import Counter
import numpy as np
import time

import pubmed_client


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

    def prepare_bioasq(self, split, path, bioasq_questions):
        """
        Process BioASQ training data.
        Save questions, ideal answers, snippets, articles, and question types.
        """
        print("Processing downloaded articles and BioASQ data for {} split".format(split))

        with open("bioasq_{}_pubmed_articles.json".format(split), "r", encoding="ascii") as f:
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

    def get_bioasq_docs(self, split, bioasq_questions):
        """
        Download and save bioasq articles, for use while processing other parts of bioasq data.
        """
        print("Downloading articles from PubMed for {} split".format(split)) 
        documents = {}
        for i, q in enumerate(bioasq_questions):
            pmid_list= [d.split("/")[-1] for d in q['documents']]
            result = self._download_bioasq_docs(pmid_list)
            # There are documents in the collection no longer are on PubMed.
            # For example, 27924029 is the only document for one question, and
            # is no longer in pubmed.
            if result is not None:
                documents.update(result)

        with open("bioasq_{}_pubmed_articles.json".format(split), "w", encoding="utf8") as f:
            json.dump(documents, f, indent=4)

    def _download_bioasq_docs(self, pmid_list):
        """
        If command line argument included, download the documents specified by BioASQ
        """
        history = "n"
        query = "[UID] OR ".join(pmid_list)
        query += "[UID]"
        id_cnt=0
        doc_dict = {}

        downloader = pubmed_client.CitationDownloader()
        search_results = downloader.search_entrez(query, history)

        if search_results is None:
           return None

        if history == "y":
            fetched_results = downloader.fetch_with_history(search_results)
        elif history == "n":
            fetched_results = downloader.fetch_without_history(search_results)

        for citation in fetched_results:
            try:
                pmid = citation.find(".//PMID").text
                id_cnt+=1
                title = citation.find(".//ArticleTitle")
                if title is not None:
                    title = le.tostring(title, encoding='unicode', method='text').strip().replace("\n", " ")
                else:
                    title = ""
                    continue
                abstract = citation.find(".//Abstract")
                if abstract is not None:
                    abstract = le.tostring(abstract, encoding='unicode', method='text').strip().replace("\n", " ")
                else:
                    abstract = ""
                    continue
                text =  title + " " + abstract
                doc_dict[pmid] = text

            except Exception as e:
                raise e

        return doc_dict


def process_bioasq():
    """
    Main processing function for bioasq data
    """
    args = get_args().parse_args()
    bq = BioASQ()
    start_time = time.time()
    for split in ["train", "test"]:
        data = bq.load_bioasq(split, args)
        bq.get_bioasq_docs(split, data)
        bq.prepare_bioasq(split, args, data)
    print("Run time:", time.time() - start_time)

if __name__ == "__main__":
    process_bioasq()
