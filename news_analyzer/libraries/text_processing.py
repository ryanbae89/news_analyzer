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
# Base imports
import string
# import copy

# Standard imports
import pandas as pd

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
    stop_words = list(set(stopwords.words('english'))) + EXTRA_STOPWORDS
    words_list = [w for w in words if not w in stop_words]
    return words_list


class ArticlePreprocessor():
    """ Class for preprocessing articles.
    """
    def __init__(self, max_features=100000, min_df=5):
        """
            Args:
            max_features: maximum number of words in the vocabulary
                    (most common).
            min_df: float in range[0.0, 1.0], default=5. This will ignore
                    terms that have a document frequency strictly lower
                    than the given threshold. If float, the parameter
                    represents a proportion of documents. If integer,
                    the absolute counts.
        """
        self.max_features = max_features
        self.min_df = min_df
        self.dtm = None
        self.vectorizer = None

    def fit(self, series_of_articles):
        """ Function for fitting preprocessor.

            Args:
            series of articles: pandas series of text (articles) or a
                    single text.

            Returns:
            self

            Raises:
            ValueError() if not enough 'clean' words (as determined by
                    clean_article) to fit model.
        """
        # Checks:
        # Convert to list if string (required by sklearn CountVectorizer)
        if isinstance(series_of_articles, str):
            series_of_articles = [series_of_articles]  # Convert to list

        # Ensure the transformation will return at least 1 word. Otherwise
        # raise ValueError.
        cleaned_words = []
        for article in series_of_articles:
            cleaned_words = cleaned_words + transform_article(article).split()
        cleaned_words = list(set(cleaned_words))
        # NOTE: This isn't the full vocabulary,
        # - just a good approximation (non-single-letters)
        num_words = len(cleaned_words)

        if num_words == 0:
            raise ValueError("Article(s) do not contain any \
                                alphabetic words to parse.")

        # Make sure the min_df (min. times of usage for a word) isn't
        # too strict. If number of words isn't 5 times min_df the min_df
        # will be set to zero. This will guarantee a dtm of with number
        # of columns equal to min(5, num_words).
        if num_words < 5*self.min_df:
            self.min_df = 0

        self.get_dtm(series_of_articles)
        return self

    def transform(self, series_of_articles):
        """ Function for transforming a set of articles
            (or a single article) into a one-hot-encoding
            with schema matching the fit dtm.

            Args:
            series_of_articles: series of text (articles) or single
            piece of text (string).

            Returns:
            document-term-matrix with the same columns as in the fit
            text document-term-matrix (under field 'self.dtm')
        """
        # sklearn vectorizer needs an iterable list/series,
        # so convert to list of just a single string
        if isinstance(series_of_articles, str):
            series_of_articles = [series_of_articles]  # Convert to list
        result = self.vectorizer.transform(series_of_articles)
        dtm = self._post_process(result)
        return dtm

    def fit_transform(self, series_of_articles):
        """ Convenience function for fitting/transforming corpus.
        """
        self.fit(series_of_articles)
        dtm = self.transform(series_of_articles)
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
                           index=self.vectorizer.get_feature_names())
        dtm = tdm.transpose()
        return dtm

    def get_vocab(self):
        """ Function for returning vocabulary of fit document-term-matrix.
            Returns:
            list mapping to columns of document-term-matrix
            of latest transformed (.transform()) text.
        """
        if self.dtm is None:
            raise ValueError("Preprocessor has not been fit. \
                                Provide series of articles.")
        return list(self.dtm.columns)
