"""
Config file

Stores location to models and data used elsewhere.
"""

import os

def get_absolute_path():
	dir = os.path.dirname(os.path.realpath(__file__))
	if "/" in dir:
		dir = dir.split("/")
		apath = "/".join(dir[:-1]) + "/"
	else:
		dir = dir.split("\\")
		apath = "\\".join(dir[:-1]) + "\\"
	return apath

RESOURCE_FOLDER = "resources"
DIR_NAME = get_absolute_path()
RESOURCE_PATH = DIR_NAME + RESOURCE_FOLDER

# change CORPUS_PATH as needed
CORPUS_PATH = RESOURCE_PATH + "/" + "articles.csv"
GUIDED_MODELER_PATH = RESOURCE_PATH + "/" + "guidedlda_model.pkl"
UNGUIDED_MODELER_PATH = RESOURCE_PATH + "/" + "regularlda_model.pkl"
PREPROCESSOR_PATH = RESOURCE_PATH + "/" + "preprocessor.pkl"
GUIDED_LDA_TOPICS = ['arts', 'automobiles', 'books', 'business', 'fashion', 'food',
					                'health', 'magazine', 'movies',
					                'politics', 'realestate', 'science', 'sports', 'technology',
					                'theater', 'travel', 'world']
