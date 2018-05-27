"""
Config file

Stores location to models and data used elsewhere.
"""
import os
DIR_NAME = os.path.dirname(os.path.realpath(__file__)).split("/")
RESOURCE_PATH = "/".join(DIR_NAME[:-1]) + "/resources"

# change CORPUS_PATH as needed
CORPUS_PATH = RESOURCE_PATH + "/" + "article1.csv"
GUIDED_MODELER_PATH = RESOURCE_PATH + "/" + "guidedlda_model.pkl"
UNGUIDED_MODELER_PATH = RESOURCE_PATH + "/" + "regularlda_model.pkl"
PREPROCESSOR_PATH = RESOURCE_PATH + "/" + "preprocessor.pickle"
GUIDED_LDA_TOPICS = ['arts', 'automobiles', 'books', 'business', 'fashion', 'food',
					                'health', 'magazine', 'movies',
					                'politics', 'realestate', 'science', 'sports', 'technology',
					                'theater', 'travel', 'world']
