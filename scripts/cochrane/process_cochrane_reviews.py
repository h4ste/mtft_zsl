"""
Script to create the Cochrane Clinical Review collection

To run:
python process_cochrane_reviews.py -c -p --citation_export=/path/to/citations.txt
Use just the -c option if you want to only collect the Cochrane articles
"""


import os
import argparse
import requests
import re
from bs4 import BeautifulSoup
import json
import glob
import random


def get_args():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description="Arg parser for Cochrane processing")
    parser.add_argument("-c",
                        dest="crawl",
                        action="store_true",
                        help="Crawl Cochrane for the clinical answers")
    parser.add_argument("-p",
                        dest="process_crawl",
                        action="store_true",
                        help="Process the crawled articles and save to json")
    parser.add_argument("--citation_export",
                        dest="citation_export",
                        help="Path to exported citations")

    return parser


def process_citations(citation_export_path):
    """
    Using the downloaded citations, parse out the urls for crawling
    """
    with open(citation_export_path, "r") as f:
        citations = f.readlines()
    urls = {}
    for line in citations:
        if line.startswith("ID:"):
            url_id = line.split(" ")[-1].strip()
        if line.startswith("US:"):
            urls[url_id] = line.split(" ")[-1].strip()
    print("Total citations:", len(urls))
    with open("cochrane_urls.json", "w", encoding="utf-8") as f:
        json.dump(urls, f, indent=4) 


def crawl():
    with open("cochrane_urls.json", "r", encoding="utf-8") as f:
        urls = json.load(f)
    headers = {'User-Agent': "Mozilla/5.0"}
    for i, url_id in enumerate(urls):
        if i % 100 == 0:
            print("Downloading citation {0}/{1}".format(i, len(urls)))
        response = requests.get(urls[url_id], headers=headers)
        if response.status_code != 200:
            print("Currently on article", url_id, urls[url_id], "but an HTTP status error has occured:")
            print(response.status_code)
            continue
        soup = BeautifulSoup(response.content, 'html.parser')
        with open("review_html_pages/{}.html".format(url_id), "w", encoding="utf-8") as f:
            f.write(str(soup.prettify()))
    print("Downloading articles is complete")


def parse_crawl():
    """
    Using the pages crawled in the above fumction,
    parse out the article and the summary
    """
    collection_dict = {}
    error_cnt = 0
    for page in glob.iglob("review_html_pages/*html"):
        with open(page, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        page_id = page.split("/")[-1].split(".")[0]
        # Find the summary and the article
        summary = ""
        objectives = ""
        methods = ""
        background = ""
        results = ""
        discussion = ""
        try:
            summary = soup.find("div", attrs={'class': "abstract abstract_plainLanguageSummary"})
            summary = summary.find_all("p")
            background = soup.find("section", attrs={'class': "background"})
            background = background.find_all(["h2", "h3", "p"])
            objectives = soup.find("section", attrs={'class': "objectives"})
            objectives = objectives.find_all(["h2", "h3", "p"])
            methods = soup.find("section", attrs={'class': "methods"})
            methods = methods.find_all(["h2", "h3", "p"])
            results = soup.find("section", attrs={'class': "results"})
            results = results.find_all(["h2", "h3", "p"])
            discussion = soup.find("section", attrs={'class': "discussion"})
            discussion = discussion.find_all(["h2", "h3", "p"])
        except AttributeError as e:
            error_cnt += 1
            continue
        summary = re.sub(r"\s+", " ", " ".join([i.text for i in summary]))
        background = re.sub(r"\s+", " ", " ".join([i.text for i in background]))
        objectives = re.sub(r"\s+", " ", " ".join([i.text for i in objectives]))
        methods = re.sub(r"\s+", " ", " ".join([i.text for i in methods]))
        results = re.sub(r"\s+", " ", " ".join([i.text for i in results]))
        discussion = re.sub(r"\s+", " ", " ".join([i.text for i in discussion]))
        article = {
                'background': background, 
                'objectives': objectives, 
                'methods': methods,
                'results': results,
                'discussion': discussion,
                }

        collection_dict[page_id] = {}
        collection_dict[page_id]['summary'] = summary
        collection_dict[page_id]['article'] = article
    print("Size of Cochrane collection: ", len(collection_dict))
    print("Number of missing attributes: ", error_cnt)
    with open("cochrane_collections/cochrane_summary_collection.json", "w", encoding="utf-8") as f:
        json.dump(collection_dict, f, indent=4)


def split_data():
        """Data spliter for train/val/test sets"""
        path = "cochrane_collections/cochrane_summary_collection.json"
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
        print("Length of train/val/test: ", len(train), "/", len(valid), "/", len(test))
        data_dict = {"train": train, "val": valid, "test": test}
        for s in data_dict:
            with open("cochrane_collections/cochrane_summary_{}_collection.json".format(s), "w", encoding="utf-8") as f:
                json.dump(data_dict[s], f, indent=4)


if __name__ == "__main__":
    args = get_args().parse_args()
    if args.crawl:
        os.makedirs("review_html_pages", exist_ok=True)
        process_citations(args.citation_export)
        crawl()
    if args.process_crawl:
        if not os.path.exists("./cochrane_collections"):
            os.makedirs("cochrane_collections")
        print("Parsing articles")
        parse_crawl()
        print("Creating splits")
        split_data()
