#import knn_model_file   #update with proper names
import numpy as np
import pandas as pd
import pickle

# Internal tools
import NYTimesArticleRetriever
import text_processing
import topic_modeling
import configs
import word_cloud_generator
import sentiment_analyzer

class Handler():

	def __init__(self):
		loader = ResourceLoader()
	    self.guided_topic_model = loader.get_guided_topic_modeler()
		self.unguided_topic_model = loader.get_guided_topic_modeler()
		self.preprocessor = loader.get_preprocessor()
		self.corpus = loader.get_corpus()

	def build_topic_models(models_path):
		"""
		Builds topic models from the news articles corpus and saves
		them in desired directory.

		Args:
			articles = pandas dataframe, corpus of articles
			models_path = string, path where the models are saved
		"""
		# get doc-term-matrix of the articles corpus
		preprocessor = text_processing.ArticlePreprocessor()
		corpus_dtm = preprocessor.get_dtm(self.corpus)
		# get vocab and word2id
		vocab, word2id = topic_modeling.get_vocab(corpus_dtm)
		# get NYT seed words
		topics_raw = NYTimesArticleRetriever.get_nytimes_topic_words()
		bad_topics = ['national', 'nyregion', 'obituaries']
		topics_clean = topic_modeling.clean_topics(topics_raw, vocab, word2id, bad_topics)
		seed_topics = topic_modeling.get_seed_topics(topics_clean, word2id)
		# fit guided model
		topic_modeler = topic_modeling.TopicModeler(n_topics=len(topics_clean), n_iter=100, 
			random_state=0, refresh=20)
		guided_topic_model = topic_modeler.fit(dtm=corpus_dtm, seed_topics=seed_topics, 
			seed_confidence=0.15)
		# fit unguided model
		n_topics = corpus_dtm.shape[0]/10.0
		topic_modeler = topic_modeling.TopicModeler(n_topics=n_topics, n_iter=100, 
			random_state=0, refresh=20)
		unguided_topic_model = topic_modeler.fit(dtm=corpus_dtm)
		# save models
		with open(models_path + '/guided_topic_model.pickle', 'wb') as file_handle:
			pickle.dump(guided_topic_model, file_handle)
		with open(models_path + '/unguided_topic_model.pickle', 'wb') as file_handle:
			pickle.dump(unguided_topic_model, file_handle)
		with open(models_path + '/vocab.pickle', 'wb') as file_handle:
			pickle.dump([vocab, word2id], file_handle)

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
		# return the topic distribution of the query article for recommender
		query_topics = [query_guided_topics, query_unguided_topics]
		return query_topics

	def get_sentiment(self, query_article):
		return sentiment_analyzer.get_sentiment(query_article)

	def get_recommended_articles(self, query_article):
		# do a separate topic analysis
		#query_topics = get_topics(query_article, guided = False)

		return pd.DataFrame(["article 1","article 2","article 3"]) # for testing purposes
		#knn_model_file.get_recommended_articles(query_topics, unguided_topic_model (filter to topic matrix?) )

	def get_word_cloud(self, query_article):
		return word_cloud_generator.generate_wordcloud(query_article)

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
