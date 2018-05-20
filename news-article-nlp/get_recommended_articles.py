# Standard
import numpy as np

# Scikit-Learn imports
from sklearn.neighbors import KDTree

def knn_prediction(doc_topic_matrix,query_vector):
    """ Generate indexes of articles that are close to query article in topic relevance

        Args:
            doc_topic_matrix = probability matrix generated from LDA model on articles corpus
            query_vector = probability vector generated from LDA model on user query article
        Returns:
            ind = index of closest 5 articles based on distance in topic relevance
    """
    tree = KDTree(doc_topic_matrix)
    dist, ind = tree.query(np.array([query_vector]), k=5)
    return(ind)

def join_process(index):
    """ Joins index to article corpus to pull out relevant artcile's titles

        Args:
            index = output of knn_prediction denoting index of the recommended articles
        Returns:
            selected_articles = five article titles based on topic relevance
    """
    article_corpus = ResourceLoader.get_corpus()
    article_corpus.rename(columns={'Unnamed: 0': 'Article_index'}, inplace=True) 
    article_corpus.set_index('Article_index', inplace=True)
    selected_articles = article_corpus.loc[index.tolist()[0]]
    return selected_articles['title']

def get_recommended_articles(doc_topic_matrix,query_vector):
	""" Links knn_prediction and join_porcess together and output final result

	    Args:
	        doc_topic_matrix = probability matrix generated from LDA model on articles corpus
            query_vector = probability vector generated from LDA model on user query article
        Returns:
            selected_articles = five article titles based on topic relevance
    """
    index = knn_prediction(doc_topic_matrix,query_vector)
    return join_porcess(index)





