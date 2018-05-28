"""
Script for building resources:

1. Downloading Kaggle data.
2. Building preprocessor and pickling for later use.
3. Building topic models and pickling for later use.

"""
import os
import sys
import requests
import json
import pickle
import argparse

import pandas as pd
import numpy as np

sys.path.append("../libraries")
import configs

## Module Constants
KAGGLE_COMP = "all-the-news"
CSV_NAMES = ["articles1.csv", "articles2.csv", "articles3.csv"]
RESOURCE_PATH = "../" + configs.RESOURCE_FOLDER
FPATHS = [RESOURCE_PATH + "/" + name in CSV_NAMES]
CONTENT_COLUMN = "content"
MIN_WORDS_IN_ARTICLE = 200

def write_pickle(data, filename):
    """ Function for writing a pickle file.

        Args:
        data: file to get pickled
        filename (str): filename ending in pkl or pickle.
    """
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

def get_files():
    """ Function for downloading Kaggle files (see CSV_NAMES in module header),
        removing short articles (potential ads), and combining into one
        large table.
    """
    os.system("kaggle datasets download -d \
                snapcrack/all-the-news --force -p '{}'".format(RESOURCE_PATH))

    list_of_tables = []
    for fpath in FPATHS:
        list_of_tables.append(pd.read_csv(fpath), encoding = 'utf8')

    full_table = pd.concat(list_of_tables)
    article_lengths = full_table[CONTENT_COLUMNs].apply(lambda x: len(x.split()))
    full_table = full_table[article_lengths > MIN_WORDS_IN_ARTICLE]
    full_table.to_csv(configs.CORPUS_PATH, index=False)
    return full_table

if __name__ == "__main__":
    """ Main script used to download Kaggle files, preprocessing data,
        and builds topic models. All resources will be saved in the resources
        folder as csv files (data) or .pkl (objects).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', dest='download_files', action='store_true', required=False)
    parser.set_defaults(download_files=False)
    args = parser.parse_args()
    download_files = args.download_files

    # If corpus csv does not exist download and build.
    if ~os.path.isfile(configs.CORPUS_PATH) or download_files:
        full_table = get_files()
    else:
        full_table = pd.read_csv(configs.CORPUS_PATH)

    # Fit preprocessor
    processor = text_processing.ArticlePreprocessor()
    processor.fit(full_table[CONTENT_COLUMN])
    write_pickle(processor, configs.PREPROCESSOR_PATH)
    dtm = processor.transform(full_table[CONTENT_COLUMN])

    # Do yo thang Ryan
