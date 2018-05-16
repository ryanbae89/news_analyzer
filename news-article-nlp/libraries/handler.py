#import sentiment        #update with proper names
#import knn_model_file   #update with proper names
#import wordcloud
import pandas as pd
import pickle

# Internal tools
import text_processing
import configs

class Handler():

	def __init__(self):
		loader = ResourceLoader()
	    self.guided_topic_model = loader.get_guided_topic_modeler()
		self.unguided_topic_model = loader.get_guided_topic_modeler()
		self.preprocessor = loader.get_preprocessor()
		self.corpus = loader.get_corpus()

	def get_topics(self, query_article, guided = True):
		# preprocess query article
		article = self.preprocessor.transform(query_article) # processed article
		vocab = self.preprocessor.get_vocab() # dictionary of word to index mapping

		# passes article into topic modeler (depending on guided or unguided)
		# returns
		return None

	def get_sentiment(self, query_article):
		return None #sentiment.get_sentiment():

	def get_recommended_articles(self, query_article):
		# do a separate topic analysis
		#query_topics = get_topics(query_article, guided = False)

		return pd.DataFrame(["article 1","article 2","article 3"]) # for testing purposes
		#knn_model_file.get_recommended_articles(query_topics, unguided_topic_model (filter to topic matrix?) )

	def get_word_cloud(self, query_article):
		return wordcloud.get_word_cloud()

class ResourceLoader():

	def get_corpus(self):
		return pd.read_csv(configs.CORPUS_URL)

	def get_preprocessor(self):
		prep = load_pickled(configs.PREPROCESSOR_PATH)
		return prep

	def get_guided_topic_modeler(self):
		modeler = load_pickled(configs.GUIDED_MODELER_PATH)
		return modeler

	def get_unguided_topic_modeler(self):
		modeler = load_pickled(configs.UNGUIDED_MODELER_PATH)
		return modeler

def load_pickled(filename):
	with open(filename, "rb") as input_file:
		object = pickle.load(input_file)
	return object
