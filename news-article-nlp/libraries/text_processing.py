"""
Module for preprocessing articles.

Includes functions for cleaning/transforming articles,
as well the ArticlePreprocessor() class that can be used to
preprocess data and return a document-term-matrix.

Example:
    prep = ArticlePreprocessor()
    prep.fit(data)
    prep.transform(data)

    # Last two lines can be replaced with either:
    # prep.fit_transform(data) or prep.get_dtm()

    # For any new query article:
    prep.transform(article)
"""

# System dependencies
import os
import sys

# Standard imports
import pandas as pd
import numpy as np
import string
import copy

# NLTK imports
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Scikit-Learn imports
from sklearn.feature_extraction.text import CountVectorizer

# Module Constants
EXTRA_STOPWORDS = [
            "said",
            "mr",
            "like",
            "ms",
            "mrs"
]



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
        """
            Args:
            max_features: maximum number of words in the vocabulary (most common).
            min_df: float in range[0.0, 1.0], default=5. This will ignore terms
                    that have a document frequency strictly lower than the given
                    threshold. If float, the parameter represents a proportion of
                    documents. If integer, the absolute counts.
        """
        self.max_features = max_features
        self.min_df=min_df
        self.dtm = None

    def fit(self, X, y=None):
        """ Function for fitting preprocessor.

            Args:
            X: series of text (articles).
            y: kept for compatability with sklearn.
        """
        self.get_dtm(X)
        return self

    def transform(self, X, y=None):
        """ Function for transforming a set of articles
            (or a single article) into a one-hot-encoding
            with schema matching the fit dtm.

            Args:
            X: series of text (articles) or single piece of text (string).

            Returns:
            document-term-matrix (if X is series) or vector (if X is string)
        """
        # sklearn vectorizer needs an iterable list/series,
        # so convert to list of just a single string
        if isinstance(X, str):
            X = [X] # Convert to list
        result = self.vectorizer.transform(X)
        dtm = self._post_process(result)
        return dtm

    def fit_transform(self, X, y=None):
        """ Convenience function for fitting/transforming corpus.
        """
        self.fit(X)
        dtm = self.transform(X)
        return dtm

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
            if self.dtm is None:
                raise ValueError("Preprocessor has not been fit. \
                                    Provide series of articles.")
            else:
                return self.dtm.copy()
          
        vectorizer = CountVectorizer(preprocessor=transform_article,
                                     max_features=self.max_features,
                                     min_df=self.min_df)

        result = vectorizer.fit_transform(series_of_articles)
        self.vectorizer = vectorizer
        dtm = self._post_process(result)
        self.dtm = dtm.copy()

        return dtm

    def _post_process(self, vectorizer_output):
        """ Internal function for processing result of sklearn
            into a document-term-matrix.

            Args:
            vectorizer_output: output of sklearn's CountVectorizer.
        """
        tdm = pd.DataFrame(vectorizer_output.toarray().transpose(),
                            index = self.vectorizer.get_feature_names())
        dtm = tdm.transpose()
        return dtm

    def get_vocab(self):
        """ Function for returning vocabulary of fit document-term-matrix.
            Returns:
            dictionary with (word: index) for (key: value)
        """
        if self.dtm is None:
            raise ValueError("Preprocessor has not been fit. \
                                Provide series of articles.")
        return copy.deepcopy(self.vectorizer.vocabulary_)


def clean_article(text):
    """ Helper function for cleaning an article.
        Converts to lowercase, removes punctuation,
        removes stopwords and anything that isn't alpabetic.

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
    stop_words = list(set(stopwords.words('english'))) + EXTRA_STOPWORDS
    words_list = [w for w in words if not w in stop_words]
    return words_list

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
