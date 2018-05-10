""" Module for reading in an analyzing NYTimes article data. 

To get the NYTimes topic seeds, run the following code:

	import NYTimesArticleRetriever
	print(NYTimesArticleRetriever.get_nytimes_topic_words())

"""

# System dependencies
import datetime
import requests

# Standard
import pandas as pd
import numpy as np

import text_processing

def get_nytimes_topic_words():
    """
        Main function to get the NYTimes topic seed words.
        Returns:
        topic_seeds: A list of lists with topic seed words. The first item of each list is the
        topic name.
    """
    data = get_nytimes_data()
    topic_seeds = get_section_words(data)
    return topic_seeds

def get_nytimes_data(sections='all'):
	"""
		Helper function for retrieving the articles from the NYTimes by section.
		Args:
		sections: An array of sections from NYTimes to pull.

		Returns:
		NYTimesData: A pandas dataframe with section names and all the section
		content as a string.
	"""
	if sections == 'all':
		sections = ['arts', 'automobiles', 'books', 'business', 'fashion', 'food',
					'health', 'magazine', 'movies', 'national', 'nyregion', 'obituaries',
					'politics', 'realestate', 'science', 'sports', 'technology',
					'theater', 'travel', 'world']

	nytimes_data = []
	for section in sections:
		url = ('http://api.nytimes.com/svc/topstories/v2/'+ section +
			   '.json?api-key=65daae448b694b07a1dca7feb0322778')
		url_content = requests.get(url).content
		temp_data = pd.read_json(url_content)
		article_content = ""
		#merge articles together
		for i in range(len(np.asarray(temp_data.results))):
			article_content = article_content + str(np.asarray(temp_data.results)[i])

		nytimes_data.append([section, article_content])

	nytimes_data = pd.DataFrame(np.asarray(nytimes_data))
	nytimes_data.to_csv("../data/NYtimes_data_"+
						datetime.datetime.now().strftime("%Y%m%d")+
						".csv")
	return nytimes_data

def get_section_words(data):
	"""
		Helper function for identifying top topic words for each category.
		Args:
		data: A pandas dataframe with section names and all the section content as a string.

		Returns:
		topic_seeds: A list of lists with top topics for each of the inputted sections.
	"""
	processor = text_processing.ArticlePreprocessor(min_df=0)
	article_dtm = processor.get_dtm(series_of_articles=data.loc[:, 1])

	topic_seeds = []
	dtm_normalized = article_dtm**2 / np.sum(article_dtm, axis=0)

	for i in range(len(dtm_normalized)):
		min_apps = (dtm_normalized.loc[i, :] >= 5)
		min_probability = ((dtm_normalized.loc[i, :] /  np.sum(dtm_normalized, axis=0)) > .4)
		words = (dtm_normalized.loc[i, min_apps & min_probability].sort_values(ascending=False))
		category_name = list([data.loc[i, 0]])
		top_words = list(words.reset_index().iloc[:, 0])[0:15]
		topic_seeds.append((category_name + top_words))
	return topic_seeds
