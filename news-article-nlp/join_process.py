# System dependencies
import os
import sys

# Standard
import pandas as pd
import numpy as np

def join_process(csv,index):
    """
        Args:
        article_corpus: article database consisting all articles
        index: output of knn_prediction denoting index of the recommended articles
    """
    article_corpus = pd.read_csv(csv)
    article_corpus.rename(columns={'Unnamed: 0': 'Article_index'}, inplace=True) 
    article_corpus.set_index('Article_index', inplace=True)
    selected_articles = article_corpus.loc[index.tolist()[0]]
    return selected_articles['title']
