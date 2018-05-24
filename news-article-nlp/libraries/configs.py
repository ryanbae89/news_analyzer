"""
Config file

Stores location to models and data used elsewhere.
"""

RESOURCE_PATH = "../resources"

# change CORPUS_PATH as needed
CORPUS_PATH = RESOURCE_PATH + "/" + "article1.csv"
GUIDED_MODELER_PATH = RESOURCE_PATH + "/" + "guidedlda_model.pickle"
UNGUIDED_MODELER_PATH = RESOURCE_PATH + "/" + "regularlda_model.pickle"
PREPROCESSOR_PATH = RESOURCE_PATH + "/" + "preprocessor.pickle"
GUIDED_LDA_TOPICS = ['arts', 'automobiles', 'books', 'business', 'fashion', 'food',
					                'health', 'magazine', 'movies',
					                'politics', 'realestate', 'science', 'sports', 'technology',
					                'theater', 'travel', 'world']
