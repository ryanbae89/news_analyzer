"""
This module is used to create the topic models from the corpus.
"""
# standard
import numpy as np
import pandas as pd
# guidedlda imports
import guidedlda


def get_vocab(dtm):
    """ Creates corpus vocabulary list and word2id dict.
        Args:
            dtm = pandas dataframe, document-term-matrix of corpus
                    output from text_processing.get_dtm()
        Returns:
            vocab = list, list of corpus vocabulary
            word2id = dict, dictionary with word as key and unique id as value
    """
    if isinstance(dtm, pd.DataFrame):
        vocab = list(dtm.columns)
        dtm = np.array(dtm)
        word2id = dict((v, idx) for idx, v in enumerate(vocab))
        return vocab, word2id
    else:
        raise ValueError('Please pass in a valid pandas dataframe.')


def clean_topics(topics, word2id, bad_topics=None):
    """ Cleans the seed topic words list.
        - Gets rid of undesired topics
        - Gets rid of words in topics that are not in corpus vocab
        Args:
            topics = list, list of lsits containing topic words (dirty)
            vocab = list, list of corpus vocabulary
            word2id = dict, dictionary with word as key and unique id as value
            bad_topics = list, list of topics to discard
        Returns:
            clean_topics = list, list of lists containing topic words (clean)
    """
    if bad_topics is None:
        return topics
    cleaned_topics = []
    for words in topics:
        topic = []
        if words[0] not in bad_topics:
            for word in words:
                if word in word2id:
                    topic.append(word)
            cleaned_topics.append(topic)
    return cleaned_topics


def get_seed_topics(topics, word2id):
    """ Creates seed topics dictionary for input into guidedlda.
        Args:
            topics = list, list of seed words for each topic
            word2id = dict, dictionary with word as key and unique id as value
        Returns:
            seed_topics = dict, input into TopicModler.fit() for guidedlda
    """
    seed_topics = {}
    for t_id, topic in enumerate(topics):
        for word in topic:
            seed_topics[word2id[word]] = t_id
    return seed_topics


def display_topics(n_words, model, vocab):
    """ Displays most relevant words in each topic.
        Args:
            n_words = int, number of words to display
            model = guidedlda model object
            vocab = list, list of words in the lda model
    """
    topic_word = model.topic_word_
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_words+1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))


class TopicModeler(object):
    """ Class for creating the topic models from the articles corpus.
    """
    def __init__(self, n_topics, n_iter=100, random_state=0, refresh=20):
        """ Constructor

            Args:
                n_topics = int, number of topics to run LDA on
                n_iter = int, number of iterations for LDA stopping condition
                random_state = int, random seed for replicating results
                refresh = int,
        """
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.random_state = random_state
        self.refresh = refresh
        self.model = None
        if np.array([n_topics, n_iter, random_state, refresh]).dtype != int:
            raise ValueError(
                'Inputs to TopicModeler must be non-negative integers!')
        if any(i < 0 for i in [n_topics, n_iter, random_state, refresh]):
            raise ValueError(
                'Inputs to TopicModeler must be non-negative integers!')

    def fit(self, dtm, seed_topics=None, seed_confidence=None):
        """ Fits topic model using guidedlda model.
            Args:
                dtm = numpy array or pandas dataframe, document-term-matrix
                guided = boolean, guided LDA or regular LDA
                seed_topics = list, list of words belonging to a topic
                seed_confidence = float, confidence of seed_topics
                n_topics = int, number of topics to model
                n_iter = int, number of iterations
                random_state = int,
                refresh = int,
            Returns:
                model = guidedlda object, fitted topic model
        """
        # check if guided
        if (bool(seed_topics) is False) and (bool(seed_confidence) is False):
            guided = False
        else:
            guided = True
        # convert dtm to numpy array if input is in pandas
        if isinstance(dtm, pd.DataFrame):
            dtm = np.array(dtm)
        if not isinstance(dtm, np.ndarray):
            raise ValueError(
                'Please input a valid pandas dataframe or numpy array for dtm!'
                )
        # fit LDA model
        if guided:
            if not isinstance(seed_topics, dict):
                raise ValueError("Please enter a dictionary for seed_topics.")
            elif not isinstance(seed_confidence, float):
                raise ValueError("Please enter a float for seed_confidence.")
            elif self.n_topics < len(seed_topics):
                raise ValueError(
                    "The number of topics must be greater than number of seed \
                    topics!")
            print("Guided LDA")
            model = guidedlda.GuidedLDA(
                n_topics=self.n_topics,
                n_iter=self.n_iter,
                random_state=self.random_state,
                refresh=self.refresh)
            model._fit(dtm, seed_topics, seed_confidence)
        elif not guided:
            print("Regular LDA")
            model = guidedlda.GuidedLDA(
                n_topics=self.n_topics,
                n_iter=self.n_iter,
                random_state=self.random_state,
                refresh=self.refresh)
            model.fit(dtm)
        self.model = model
        return model


class TopicModelerGridSearch():
    """ Class for creating the topic models from the articles corpus.
    """
    def __init__(self, n_topics_list, n_iter, random_state, refresh):
        """ Constructor
            Args:
                n_topics = list of ints, list of topic numbers for grid search
                n_iter = int, number of iterations for LDA stopping condition
                random_state = int, random seed for replicating results
                refresh = int,
        """
        self.n_topics_list = n_topics_list
        self.n_iter = n_iter
        self.random_state = random_state
        self.refresh = refresh
        self.model = None
        self.loglikelihoods = None
        self.n_topics_opt = None
        if not isinstance(n_topics_list, list):
            raise ValueError('You must enter a valid list of n_topics.')

    def gridsearch(self, dtm):
        """ Does grid search over list of n_topics values and returns
            the best model.The best model is the model with lowest
            abs(-loglikelihood).

            Args:
                dtm = np.array or pd.DataFrame, document-term-matrix
        """
        # convert dtm to numpy array if input is in pandas
        if isinstance(dtm, pd.DataFrame):
            dtm = np.array(dtm)
        # perform grid search
        ll_values = []
        for i, n_topics in enumerate(self.n_topics_list):
            print('fitting model with n_topics = {}...'.format(n_topics))
            model = TopicModeler(
                n_topics=n_topics,
                n_iter=self.n_iter,
                random_state=self.random_state,
                refresh=self.refresh)
            model = model.fit(dtm)
            if i == 0:
                best_model = model
                self.n_topics_opt = n_topics
                ll_values.append(best_model.loglikelihoods_[-1])
            else:
                if model.loglikelihoods_[-1] < best_model.loglikelihoods_[-1]:
                    best_model = model
                    self.n_topics_opt = int(n_topics)
                ll_values.append(model.loglikelihoods_[-1])
        self.model = best_model
        self.loglikelihoods = ll_values
