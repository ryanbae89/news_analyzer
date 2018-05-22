#import knn_model_file   #update with proper names
import numpy as np
import pandas as pd
import pickle

# Internal tools
import text_processing
import topic_modeling
import configs
import word_cloud_generator
import sentiment_analyzer
import get_recommended_articles as article_recommender # change this into import article_reommender

class Handler():
	"""
	Usage: Call the handler  
	"""
	def __init__(self):
		loader = ResourceLoader()
		self.guided_topic_model = loader.get_guided_topic_modeler()
		self.unguided_topic_model = loader.get_unguided_topic_modeler()
		self.preprocessor = loader.get_preprocessor()
		self.corpus = loader.get_corpus()

	def get_topics(self, query_article):
		"""
		Processes query article for recommender system.

		Args:
			query_article = string, article or words being queried
		Returns:
			query_topics = list, guided and unguided topics distribution
		"""
		# convert query article to doc-term-matrix
		query_dtm = self.preprocessor.transform(query_article)
		# join the query article doc-term-matrix with models
		query_guided_topics = self.guided_topic_model.transform(np.array(query_dtm))
		query_unguided_topics = self.unguided_topic_model.transform(np.array(query_dtm))
		
		top_guided_topic_index = query_guided_topics.argsort()[0][-3:]
		print(top_guided_topic_index)
		print(type(top_guided_topic_index))
		# return the topic distribution of the query article for recommender
		query_topics = [np.asarray(configs.GUIDED_LDA_TOPICS)[top_guided_topic_index], 
		                query_unguided_topics]
		return query_topics

	def get_sentiment(self, query_article):
		return sentiment_analyzer.get_sentiment(query_article)

	def get_recommended_articles(self, query_article):
		# do a separate topic analysis
		query_vector = self.get_topics(query_article)[1]
		doc_topic_matrix = self.unguided_topic_model.doc_topic_
		recommended_articles = article_recommender.get_recommended_articles(doc_topic_matrix, 
			query_vector, self.corpus)
		return recommended_articles

	def get_word_cloud(self, query_article):
		return word_cloud_generator.generate_wordcloud(query_article)

class ResourceLoader():
	def get_corpus(self):
		return pd.read_csv(configs.CORPUS_PATH)

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
