# System dependencies
import requests
import datetime


# Standard
import pandas as pd
import numpy as np

import text_processing


""" Module for reading in an analyzing NYTimes article data.
"""

class NYTimesArticleRetriever():
    """ Class for reading and anlayzing NYTimes articles.
    """
    def __init__(self):
        """ Constructor.
            Args:
            None.
        """
        
        
        #self.max_features = max_features
        #self.min_df=min_df

    def get_NYTimes_data(self, sections='all'):
        """
            Function for retrieving the articles from the NYTimes by section.
            Args:
            sections: An array of sections from NYTimes to pull.

            Returns:
            NYTimesData: A pandas dataframe with section names and all the section content as a string.
        """
        if sections == 'all':
            sections = ['arts', 'automobiles', 'books', 'business', 'fashion', 'food'
                    , 'health',  'magazine', 'movies', 'national', 'nyregion', 'obituaries', 
                    'politics', 'realestate', 'science', 'sports',  'technology', 
                    'theater',  'travel',  'world']
        
        NYTimesData = []
        for section in sections:
            url = ('http://api.nytimes.com/svc/topstories/v2/'+ section + '.json?api-key=65daae448b694b07a1dca7feb0322778')
            s=requests.get(url).content
            temp_data = pd.read_json(s)
            df = ""
            for i in range(len(np.asarray(temp_data.results))):
                df = df + str(np.asarray(temp_data.results)[i])

            NYTimesData.append([section,df])

        NYTimesData = pd.DataFrame(np.asarray(NYTimesData))
        NYTimesData.to_csv("../data/NYtimes_data_"+datetime.datetime.now().strftime("%Y%m%d")+".csv")
        return NYTimesData
    
    def get_section_words(self, data):
        """
            Function for identifying top topic words for each category.
            Args:
            data: A pandas dataframe with section names and all the section content as a string.

            Returns:
            topic_seeds: A list of lists with top topics for each of the inputted sections.
        """
        processor = text_processing.ArticlePreprocessor(min_df = 0)
        d = processor.get_dtm(series_of_articles = data.loc[:,1])

        topic_seeds = []
        d2 = d**2 / np.sum(d,axis=0)

        for i in range(len(d2)):

            words = (d2.loc[i, (d2.loc[i,:]>=5) &  
                                      ((d2.loc[i,:] /  np.sum(d2,axis=0)) > .4 )].sort_values(ascending = False))
            topic_seeds.append((list([data.loc[i,0]]) + list(words.reset_index().iloc[:,0])[0:15]))
        return topic_seeds
    
