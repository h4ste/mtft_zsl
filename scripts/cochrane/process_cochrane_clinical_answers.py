"""
Script to collect the Cochrane Clinical Answers

When the script is run, provide the option to download the Cochrane articles 
-c and the option to parse them, -p, as well as the path to the exported Cochrane clinical answers. This is split into two parts to deal
with any unforseen issues while downloading the articles.

For example, depending on the name of your exported file:
python process_cochrane_clinical_answers.py -cp --citation_export=./citation-export_clinical_answers.txt

Errors related to HTTP status codes are handled by the script, 
since not all of the the urls in the exported citations will be active.
Notification of these will be written to clinical_answer.log. 
"""

import os
import random
import requests
import re
import statistics
from bs4 import BeautifulSoup
import json
import glob
import argparse
import logging


def get_args():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description="Command line options for parsing Cochrane clinical answers")
    parser.add_argument("--citation_export",
                        dest="citation_export",
                        help="Path to exported citations")
    parser.add_argument("-c",
                        dest="crawl",
                        action="store_true",
                        help="Crawl Cochrane for the clinical answers")
    parser.add_argument("-p",
                        dest="process_crawl",
                        action="store_true",
                        help="Process the crawled articles and save to json")
    return parser


def process_citations(citation_export):
    """
    Using the downloaded citations, parse out the urls for crawling
    """
    with open(citation_export, "r") as f:
        citations = f.readlines()
    urls = {}
    for line in citations:
        if line.startswith("DOI:"):
            doi = line.split(" ")[-1].strip()
        if line.startswith("US:"):
            urls[doi] = line.split(" ")[-1].strip()
    print("Total citations:", len(urls))
    with open("cochrane_clinical_answer_urls.json", "w", encoding="utf-8") as f:
        json.dump(urls, f, indent=4) 


def crawl():
    with open("cochrane_clinical_answer_urls.json", "r", encoding="utf-8") as f:
        urls = json.load(f)
    headers = {'User-Agent': "Mozilla/5.0"}
    for i, url_id in enumerate(urls):
        if i % 100 == 0:
            print("Downloading citation {0}/{1}".format(i, len(urls)))
        try:
            response = requests.get(urls[url_id], headers=headers)
            if response.status_code != 200:
                logging.info("Currently on article %s %s but an HTTP status error has occured:", url_id, urls[url_id]) 
                logging.info(response.status_code)
            soup = BeautifulSoup(response.content, 'html.parser')
            url_id = url_id.split("/")[-1].split(".")[-1]
            with open("clinical_answer_html_pages/{}.html".format(url_id), "w", encoding="utf-8") as f:
                f.write(str(soup.prettify()))
        except requests.exceptions.ConnectionError as e:
            logging.info(e)
            continue
    print("Downloading articles is complete")


def assign_new_id(article_id, collection_dict):
    """Little function to make unique id for articles that are used by multiple questions"""
    extra_ids = ["-0", "-1", "-2", "-3", "-5"]
    for i in extra_ids: 
        new_id = article_id + i
        if new_id not in collection_dict:
            return new_id
    raise ValueError("Out of IDS!")
        

def parse_crawl():
    """
    Using the pages crawled in the above fumction,
    parse out the article and the summary
    """
    # The summary collection is required because the reviews are the source documents for the collection
    if not os.path.exists("./cochrane_collections/cochrane_summary_collection.json"):
        raise OSError("cochrane_summary_collection.json does not exist. Please run process_cochrane_reviews.py")

    with open("cochrane_collections/cochrane_summary_collection.json", "r") as f:
        articles = json.load(f)
    collection_dict = {}
    error_cnt = 0
    missing_cnt = 0
    for page in glob.iglob("clinical_answer_html_pages/*html"):
        with open(page, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        page_id = page.split("/")[-1].split(".")[0]
        # Find the url of the review to map the answer -> review
        try:
            urls = soup.find_all("a", attrs={"href": True})
            question = soup.find("h1", attrs={"class": "publication-title"}).text.strip()
            answer_section = soup.find("h2", attrs={"class": "answer"})
            answer = answer_section.parent.find_all("p")
            answer = " ".join([p.text.strip() for p in answer])

            href_list = []
            for url in urls:
                href = url['href']
                if href.startswith("/cdsr/doi/10.1002/"): 
                    # Inconsistent urls sad face :(
                    if "pub" in href:
                        review_id = href.split(".")[-2]
                    else:
                        review_id = href.split(".")[-1].split("/")[0]
                    assert review_id.startswith("CD0"), (review_id, href)
                    href_list.append(review_id)

            # All the hrefs in the html should be the reference to the review article,
            # because it appears that only a single review article is used as reference
            # in the clinical answer
            if len(href_list) > 0:
                href_set = set(href_list)
                assert len(href_set) == 1, len(href_set)
                href_set = list(href_set)
                article_id = href_set[0] 
                if article_id in articles:
                    article = articles[article_id]['article']
                    assert isinstance(article, dict), article
                    assert isinstance(question, str), question
                    assert isinstance(answer, str), answer
                    # Some questions, which are very similar, use the
                    # same article as reference. Add an additional identifier to these
                    if article_id in collection_dict:
                        article_id = assign_new_id(article_id, collection_dict)
                    collection_dict[article_id] = {
                            'question': question,
                            'article': article,
                            'answer': answer
                            }
                else:
                    missing_cnt += 1

        except AttributeError as e:
            error_cnt += 1
            continue

    print("Size of collection:", len(collection_dict))
    with open("cochrane_collections/cochrane_clinical_answer_collection.json", "w", encoding="utf-8") as f:
        json.dump(collection_dict, f, indent=4)
 

def split_data():
    """Data spliter for train/val/test sets"""
    path = "cochrane_collections/cochrane_clinical_answer_collection.json"
    random.seed(13)
    with open(path, "r", encoding="utf-8") as f:
        cochrane_dict = json.load(f)
    cochrane = [q for q in cochrane_dict]
    random.shuffle(cochrane)
    l = len(cochrane)
    train_split = int(l*.7)
    valid_split = train_split + int(l*.1)
    train = cochrane[0:train_split]
    valid = cochrane[train_split:valid_split] 
    test = cochrane[valid_split:] 
    train = {q: cochrane_dict[q] for q in train}
    valid = {q: cochrane_dict[q] for q in valid}
    test = {q: cochrane_dict[q] for q in test}
    print("Length of train/val/test sets", len(train), "/", len(valid), "/", len(test))
    data_dict = {"train": train, "val": valid, "test": test}
    for s in data_dict:
        with open("cochrane_collections/cochrane_clinical_answer_{}_collection.json".format(s), "w", encoding="utf-8") as f:
            json.dump(data_dict[s], f, indent=4)


if __name__ == "__main__":
    args = get_args().parse_args()
    logging.basicConfig(filename="clinical_answer.log", filemode='w', level=logging.INFO)
    if args.crawl:
        os.makedirs("clinical_answer_html_pages", exist_ok=True)
        process_citations(args.citation_export)
        crawl()
    if args.process_crawl:
        if not os.path.exists("./cochrane_collections"):
            os.makedirs("cochrane_collections")
        print("Parsing articles")
        parse_crawl()
        print("Creating splits")
        split_data()
