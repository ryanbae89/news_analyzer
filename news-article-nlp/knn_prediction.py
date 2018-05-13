# System dependencies
import os
import sys

# Standard
import pandas as pd
import numpy as np

# Scikit-Learn imports
from sklearn.neighbors import KDTree

def knn_prediction(doc_topic_matrix,query_vector):
    """
        Args:
        doc_topic_matrix: probability matrix generated from LDA model on articles corpus
        query_vector: probability vector generated from LDA model on user query article
    """
    tree = KDTree(doc_topic_matrix)
    dist, ind = tree.query(np.array([query_vector]), k=5)
    return(ind)
