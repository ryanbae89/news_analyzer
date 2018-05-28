"""
Script for building full resources:

1. Downloading Kaggle data.
2. Building preprocessor and pickling for later use.
3. Building topic models and pickling for later use.

"""
import os
import sys
import requests
import json
import pickle

sys.path.append("../libraries")
import configs

# Module Constants
KAGGLE DATASETS = ["articles1.csv", "articles2.csv", "articles3.csv"]
KAGGLE_COMP = "all-the-news"
content_column = "content"
min_word_length_per_article = 200


if __name__ == "__main__":
    """ Main script
    """
    download_from_kaggle(KAGGLE_DATASETS, KAGGLE_COMP, kaggle_info)

    fpaths = configs.fpaths
    list_of_tables = []
    for fpath in fpaths:
        list_of_tables.append(pd.read_csv(fpath), encoding = 'utf8')

    full_table = pd.concat(list_of_tables)

    article_lengths = full_table[content_columns].apply(lambda x: len(x.split()))
    full_table = full_table[article_lengths > min_word_length_per_article]

    full_table.to_csv(configs.CORPUS_PATH, index=False)

    # Fit preprocessor
    processor = text_processing.ArticlePreprocessor()
    processor.fit(full_table[content_column])
    write_pickle(processor, configs.PREPROCESSOR_PATH)
    dtm = processor.transform(full_table[content_column])




def write_pickle(data, filename):
    """ Function for writing a pickle file.
        Args:
        data: file to get pickled
        filename (str): filename ending in pkl or pickle.
    """
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


def download_from_kaggle(data_sets, competition):
    """Fetches data from Kaggle

    Parameters
    ----------
    data_sets : (array)
        list of dataset filenames on kaggle. (e.g. train.csv.zip)

    competition : (string)
        name of kaggle competition as it appears in url
        (e.g. 'rossmann-store-sales')

    """
    kaggle_username = input("Enter Kaggle Username: ")
    kaggle_password = input("Eter Kaggle Password: ")
    kaggle_login_info = {"UserName": kaggle_username,
                         "Password": kaggle_password}

    kaggle_dataset_url = "https://www.kaggle.com/c/{}/download/".format(competition)


    for data_set in data_sets:
        data_url = "".join(kaggle_dataset_url+data_set)
        print(data_url)
        data_output = "".join(configs.RESOURCE_PATH+"/"+data_set)
        # Attempts to download the CSV file. Gets rejected because we are not logged in.
        r = requests.get(data_url)
        # Login to Kaggle and retrieve the data.
        r = requests.post(r.url, data=kaggle_login_info, stream=True)
        # Writes the data to a local file one chunk at a time.
        with open(data_output, 'wb') as f:
            # Reads 512KB at a time into memory
            for chunk in r.iter_content(chunk_size=(512 * 1024)):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
