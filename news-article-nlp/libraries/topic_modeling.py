# system dependencis
import os

# standard
import numpy as np
import pandas as pd

# guidedlda imports
import guidedlda

#  news-articles-nlp imports
import NYTimesArticleRetriever
import text_processing 

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
        print('creating corpus vocab list...')
        vocab = list(dtm.columns)
        dtm = np.array(dtm)
        print('creating word2id dictionary...')
        word2id = dict((v, idx) for idx, v in enumerate(vocab))
        return vocab, word2id
    else:
        raise ValueError('Please pass in a valid pandas dataframe.')

def clean_topics(topics, vocab, word2id, bad_topics=None):
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
    clean_topics = []
    for i, words in enumerate(topics):
        topic = []
        if words[0] not in bad_topics:
            for word in words:
                if word in word2id:
                    topic.append(word)
            clean_topics.append(topic)
    return clean_topics
       
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
    def __init__(self, n_topics, n_iter, random_state, refresh):
        """ Constructor
        """
        # super(TopicModeler, self).__init__()
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.random_state = random_state
        self.refresh = refresh   
        
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
        if (seed_topics is not None) and (seed_confidence is not None):
            guided = True
        else:
            guided = False
        # convert dtm to numpy array if input is in pandas
        if isinstance(dtm, pd.DataFrame):
            print('converting dtm to numpy array...')
            dtm = np.array(dtm)
        if not isinstance(dtm, np.array):
            raise ValueError('please input a valid pandas dataframe or numpy array for dtm!')
        # guided case
        if guided:
            if not type(seed_topics) == dict:
                raise ValueError("Please enter a dictionary for seed_topics.")
            if not type(seed_confidence) == float:
                raise ValueError("Please enter a float for seed_confidence.")
            print("Guided LDA")
            model = guidedlda.GuidedLDA(n_topics=self.n_topics, n_iter=self.n_iter, 
                random_state=self.random_state, refresh=self.refresh)
            model._fit(dtm, seed_topics, seed_confidence)
        # not guided case
        elif not guided:
            print("Regular LDA")
            model = guidedlda.GuidedLDA(n_topics=self.n_topics, n_iter=self.n_iter, 
                random_state=self.random_state, refresh=self.refresh)
            model.fit(dtm)
        return model




