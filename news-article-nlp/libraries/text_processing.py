# System dependencies
import os
import sys

# Standard
import pandas as pd
import numpy as np
import string

# NLTK imports
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Scikit-Learn imports
from sklearn.feature_extraction.text import CountVectorizer

""" Module for preprocessing articles.
"""

def transform_article(article):
    """ Function for transforming a single article.
        Cleans text (see clean_article function) and performs
        lemmatization.

        Args:
        article: a string of text.

        Returns:
        A string of preprocessed text.
    """
    tokens = clean_article(article)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    transformed_article = " ".join(lemmatized)
    return transformed_article

def clean_article(text):
    """ Helper function for cleaning an article.
        Converts to lowercase, removes punctuation,
        removes stopwords anything that isn't alpabetic.

        Args:
        text: A string of text.

        Returns:
        A list of cleaned words from the text.
    """
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words_list = [w for w in words if not w in stop_words]
    return words_list

class ArticlePreprocessor():
    """ Class for preprocessing articles.
    """
    def __init__(self, max_features=100000, min_df=5):
        """ Constructor.

            Args:
            max_features: maximum number of words in the vocabulary (most common).
            min_df: float in range[0.0, 1.0], default=5. This will ignore terms
                    that have a document frequency strictly lower than the given
                    threshold. If float, the parameter represents a proportion of
                    documents. If integer, the absolute counts.
        """
        self.max_features = max_features
        self.min_df=min_df

    def get_dtm(self, series_of_articles=None):
        """ Method for getting the document-term-matrix (dtm).
            If a series_of_articles is not passed, assumes
            dtm has already been constructed.

            Args:
            series_of_articles: a pandas series of articles.

            Returns:
            A document-term-matrix.

            Added/Modified Fields:
            dtm: a copy of the document-term-matrix.
            vectorizer: the sklearn CountVectorizer(). See their
                        documentation for more information.
        """
        if series_of_articles is None:
            return self.dtm.copy()
          
        vectorizer = CountVectorizer(preprocessor=transform_article,
                                     max_features=self.max_features,
                                     min_df=self.min_df)

        result = vectorizer.fit_transform(series_of_articles)

        tdm = pd.DataFrame(result.toarray().transpose(),
                            index = vectorizer.get_feature_names())

        dtm = tdm.transpose()

        self.dtm = dtm
        self.vectorizer = vectorizer
        return dtm.copy()