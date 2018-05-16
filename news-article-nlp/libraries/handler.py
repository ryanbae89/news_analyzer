
#import sentiment        #update with proper names
#import knn_model_file   #update with proper names
#import wordcloud
import pandas as pd
import numpy as np

class Handler():
	
	def __init__(self):
	    self.guided_topic_model, self.unguided_topic_model, self.preprocessor = load_files() # to be created later
	
	def get_topics(self, query_article, guided = True):
		# helper function
		
		# preprocess query article
		# passes article into topic modeler (depending on guided or unguided)
		# returns
		return None
	
	def get_sentiment(self, query_article):
		return None #sentiment.get_sentiment():
	
	def get_recommended_articles(self, query_article):
		# do a separate topic analysis
		#query_topics = get_topics(query_article, guided = False)
		
		return pd.DataFrame([[query_article,"article 2","article 3"],
	                         ["article 1b","article 2b",str(np.random.rand(1))]]) # for testing purposes
		#knn_model_file.get_recommended_articles(query_topics, unguided_topic_model (filter to topic matrix?) )

	def get_word_cloud(self, query_article):
		return wordcloud.get_word_cloud()

def load_files():
	return ["hi","hello","hi"]
