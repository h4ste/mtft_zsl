"""
Script to process DUC summarization data for duc.py dataset builder.
The script will generate various intermediate data files, 
including separate files for the human summaries, topics, and original documents. The final collections will be saved in processed_duc/duc<year>/duc_collection_<year>.json

Access rights to the original DUC data must be granted by NIST. This script will process only the 2004 and 2007 data. Place the data you receive from NIST (unpacking any .tgz first, as well as any nested .tgz, as will be the case for the DUC 2007 summarization documents) in the same parent directory for both years. For DUC 2007, these files will include duc2007_topics.sgml, mainEval.tar.gz, and DUC2007_Summarization_documents.tgz. For DUC 2004, this will include duc2004_results.tgz and DUC2004_Summarization_Documents.tgz.  

Note that for DUC 2004, you will have to go into /duc2004_results/ROUGE and unpack duc2004.task1.ROUGE.models.tar.gz.

To run the script
python process_duc.py --path=/path/to/<directory with duc data>
For example
python process_duc.py --path=/home/data/DUC
where DUC will contain the unpacked files listed above.

If the script runs sucessfully you should see this print output:
Processing duc 2004
Source document total: 489
Summ total 489
Total collection: 489
Processing duc 2007
Source document total: 1124
Summ total 45
Doc-topic total 1125
Total collection: 180
"""

import json
import glob
from bs4 import BeautifulSoup
import lxml.etree as le
import argparse
import re
import os
import sys


def get_args():
    """
    Get command line arguments
    """

    parser = argparse.ArgumentParser(description="Arguments for data exploration")
    parser.add_argument("--path",
                        dest="path",
                        help="Path to directory containing DUC data")
    return parser


class DocProcessor():
    """Parent class for all parsing shared between ducs"""
    
    def __init__(self, duc_year, path):
        self.duc_year = duc_year
        self.path = path

    def parse_docs(self, duc_path):
        """Get some data"""
        doc_dict = {}
        for duc_dir in glob.iglob(duc_path):
            doc_path = os.path.join(duc_dir, "*")
            for duc_doc in glob.iglob(doc_path):
                try:
                    with open(duc_doc, "r") as f:
                        soup = BeautifulSoup(f, features="lxml")
                except Exception as e:
                    print(e)
                    continue

                text = soup.find("text").text
                if len(text.split()) < 20:
                    continue
                # Lots of whitespace in there
                text = re.sub(r"\s+", " ", text)
                doc_no = soup.find("docno").text.strip()
                if doc_no not in doc_dict:
                    doc_dict[doc_no] = text.strip()

        print("Source document total:", len(doc_dict))
        self.save_duc(doc_dict, "docs")

    def save_duc(self, duc_docs, doc_type):
        """Save docs to json"""
        with open("processed_duc/duc_{y}/duc_{d}_{y}.json".format(d=doc_type, y=self.duc_year), "w") as f:
            json.dump(duc_docs, f, indent=4)


class DUC2004(DocProcessor):
    """Class to parse DUC 2004"""

    def __init__(self, duc_year, path):
        super().__init__(duc_year, path)

    def parse_summs(self):
        # Just parse the summaries for the first task
        duc_path = os.path.join(self.path, "duc2004_results/ROUGE/eval/models/1/*")
        summ_dict = {}
        for duc_summ in glob.iglob(duc_path):
            split_duc = duc_summ.split("/")[-1].split(".")
            doc_id = ".".join(split_duc[-2:])
            with open(duc_summ, "r") as f:
                summ = " ".join([s.strip() for s in f.readlines()])
            summ_dict[doc_id] = summ

        print("Summ total", len(summ_dict))
        self.save_duc(summ_dict, "summs")

    def combine(self):
        """Combine single doc summaries with their documents"""
        duc_doc_path = "processed_duc/duc_2004/duc"
        duc_collection = {}
        with open("{0}_summs_2004.json".format(duc_doc_path), "r") as f:
            duc_summs = json.load(f)
        with open("{0}_docs_2004.json".format(duc_doc_path), "r") as f:
            duc_docs = json.load(f)
        # Parse the jsons!
        for doc_id in duc_docs: 
            if doc_id not in duc_collection:
                duc_collection[doc_id] = {}
                duc_collection[doc_id]['document'] = duc_docs[doc_id]
                duc_collection[doc_id]['summary'] = duc_summs[doc_id]
        
        print("Total collection:", len(duc_collection))
        self.save_duc(duc_collection, "collection")


class DUC2007(DocProcessor):
    """Class to parse DUC 2007"""

    def __init__(self, duc_year, path):
        super().__init__(duc_year, path)

    def parse_summs(self):
        """Parse out the human generated summaries for the topics"""
        duc_path = os.path.join(self.path, "mainEval/ROUGE/models/*")
        summ_dict = {}
        for duc_summ in glob.iglob(duc_path):
            split_duc = duc_summ.split("/")[-1].split(".")
            topic_id = split_duc[0]
            summ_id = split_duc[-1]
            with open(duc_summ, "r", errors="ignore") as f:
                # Some summs may be written on multiple lines, some may not.
                summ = " ".join([s.strip() for s in f.readlines()])
                assert summ != ""
                assert len(summ) > 10
            if topic_id not in summ_dict:
                summ_dict[topic_id] = {}
                summ_dict[topic_id][summ_id] = summ
            else:
                summ_dict[topic_id][summ_id] = summ

        for topic in summ_dict:
            assert len(summ_dict[topic]) == 4
        print("Summ total", len(summ_dict))
        self.save_duc(summ_dict, "summs")

    def parse_topics(self):
        """Parse the topics"""
        topic_path = os.path.join(self.path, "duc2007_topics.sgml")

        with open(topic_path, "r") as f:
            soup = BeautifulSoup(f, "lxml")
        doc_dict = {}
        for topic in soup.find_all("topic"):
            topic_id = topic.find("num").text.strip()
            topic_statement = topic.find("narr").text.strip()
            topic_docs = topic.find("docs").text.strip().split("\n")
            for topic_doc in topic_docs:
                doc_dict[topic_doc] = {'topic_id': topic_id, 'question': topic_statement}
        print("Doc-topic total", len(doc_dict))
        self.save_duc(doc_dict, "topics")

    def combine(self):
        """Combine summaries with their documents"""
        duc_doc_path = "processed_duc/duc_2007/duc"
        duc_collection = {}
        with open("{0}_summs_2007.json".format(duc_doc_path), "r") as f:
            duc_summs = json.load(f)
        with open("{0}_docs_2007.json".format(duc_doc_path), "r") as f:
            duc_docs = json.load(f)
        with open("{0}_topics_2007.json".format(duc_doc_path), "r") as f:
            duc_topics = json.load(f)
        # Parse the jsons!
        for doc_id in duc_docs: 
            if doc_id not in duc_collection:
                duc_collection[doc_id] = {}
                # The topic id will have a last letter indicating which annotator selected the documents for that topic. 
                # However, this key in the summaries does not have this final letter.
                topic_id = duc_topics[doc_id]['topic_id']
                duc_collection[doc_id]['topic'] = topic_id
                duc_collection[doc_id]['document'] = duc_docs[doc_id]
                duc_collection[doc_id]['question'] = duc_topics[doc_id]['question']
                duc_collection[doc_id]['summaries'] = duc_summs[topic_id[:-1]]

        print("Total collection:", len(duc_collection))
        self.save_duc(duc_collection, "collection")

    def remove_token(self, sentence):
        """Remove one token from the sentence"""
        split_sentence = sentence.split()
        return " ".join(split_sentence[:-1])

    def map_summs2src(self, multi_doc_dict):
        """
        Try to map the summaries to each source document
        """
        mapped_summs = {}
        for topic_id in multi_doc_dict:
            mapped_summs[topic_id] = {}
            documents = multi_doc_dict[topic_id]['documents']
            mapped_summs[topic_id]['question'] = multi_doc_dict[topic_id]['question']
            assert documents != []
            # 4 summaries per topic, each with an annotator id A-I or maybe J.
            summaries = multi_doc_dict[topic_id]['summaries']
            mapped_summs[topic_id]['summaries'] = {}
            assert isinstance(summaries, dict)
            for summ_id in summaries:
                if summ_id in mapped_summs[topic_id]['summaries']: 
                    print("Summary ID already added to dict!")
                mapped_summs[topic_id]['summaries'][summ_id] = {}
                # Add the summmary to the new data format
                mapped_summs[topic_id]['summaries'][summ_id]['summary'] = summaries[summ_id]
                split_summ = summaries[summ_id].split(".")
                map_tracker = {}
                for i, doc in enumerate(documents):
                    inexact_map_cnt = 0
                    map_cnt = 0
                    # For that annotator's summ, split on period and see if there any sentences in that doc.
                    for sentence in split_summ:
                        if all([sentence in doc, len(sentence.split()) > 5, sentence != ""]):
                            map_cnt += 1
                        else:
                            # Remove 1 token at a time:
                            cut_sentence = sentence.lower()
                            while cut_sentence:
                                # cut_sentence will be false once [] is returned
                                cut_sentence = self.remove_token(cut_sentence)
                                if len(cut_sentence.split()) >= 2 and cut_sentence in doc:
                                    inexact_map_cnt += 1
                                    break
                                # Don't care about bigrams for now
                                elif len(cut_sentence.split()) < 2:
                                    break
                    # For each document, add the map counts to the tracking dict
                    map_tracker[i] = inexact_map_cnt + map_cnt

                # For each summary, order the documents based on the count
                # Flatten out the summaries as well. {'topic': 'summary_A': {'summary': summ, 'documents': sorted_doc}}
                sorted_indicies = {k: v for k, v in sorted(map_tracker.items(), key=lambda item: item[1], reverse=True)}
                sorted_docs = [documents[i] for i in sorted_indicies]
                mapped_summs[topic_id]['summaries'][summ_id]['documents'] = sorted_docs

        return mapped_summs

    def make_multi_doc_collection(self):
        """
        Order the documents by which maps more closely to a summary.
        In the dataset builders, the multiple documents will be concatenated 
        in the order in which they are provided in the json file for each summ.
        """
        duc_doc_path = "processed_duc/duc_2007/duc"
        with open("{0}_summs_2007.json".format(duc_doc_path), "r") as f:
            duc_summs = json.load(f)
        with open("{0}_docs_2007.json".format(duc_doc_path), "r") as f:
            duc_docs = json.load(f)
        with open("{0}_topics_2007.json".format(duc_doc_path), "r") as f:
            duc_topics = json.load(f)
        # Parse the jsons!
        multi_doc_dict = {}
        for doc_id in duc_docs: 
            # The topic id will have a last letter indicating which annotator selected the documents for that topic. 
            # However, this key in the summaries does not have this final letter.
            topic_id = duc_topics[doc_id]['topic_id']
            if topic_id not in multi_doc_dict:
                multi_doc_dict[topic_id] = {}
                multi_doc_dict[topic_id]['documents'] = []
                # 4 summaries per topic
                multi_doc_dict[topic_id]['summaries'] = duc_summs[topic_id[:-1]]
                multi_doc_dict[topic_id]['question'] = duc_topics[doc_id]['question']
            multi_doc_dict[topic_id]['documents'].append(duc_docs[doc_id])

        # Make an example for each summary, with the document that it maps to first in the list
        multi_doc_dict = self.map_summs2src(multi_doc_dict)
        print("Total collection:", 4*len(multi_doc_dict))
        for topic in multi_doc_dict:
            assert len(multi_doc_dict[topic]) == 2
        self.save_duc(multi_doc_dict, "collection")


def main_processor():
    """Process each year of duc"""
    args = get_args().parse_args()
    duc_years = ["2004", "2007"]
    for duc_year in duc_years:
        print("Processing duc {}".format(duc_year))
        os.makedirs("./processed_duc/duc_{}".format(duc_year), exist_ok=True)
        if duc_year == "2004":
            processor = DUC2004(duc_year, args.path)
            duc_path = os.path.join(args.path, "DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs/*")
        if duc_year == "2007":
            duc_path = os.path.join(args.path, "DUC2007_Summarization_Documents/duc2007_testdocs/main/*")
            processor = DUC2007(duc_year, args.path)
        processor.parse_docs(duc_path)
        processor.parse_summs()
        if duc_year == "2004":
            processor.combine()
        elif duc_year == "2007":
            processor.parse_topics()
            # Method to approximate order the summaries by ngram comparison
            processor.make_multi_doc_collection()

        
if __name__ == "__main__":
    main_processor()
