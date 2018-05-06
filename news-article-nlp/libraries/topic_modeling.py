# Standard
import numpy as np

# Guided LDA
import guidedlda

class TopicModeler(object):
    """ Class for creating the topic models from the articles corpus.
    """
    def __init__(self):
        """ Constructor
        """
        super(TopicModeler, self).__init__()

    def fit(self, dtm, guided, seed_topics, seed_confidence, n_topics, 
        n_iter, random_state, refresh):
        """ Fits topic model using guidedlda model.

            Args:
                dtm = numpy array, document-term-matrix
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
        if guided:
            if not type(seed_topics) == list:
                raise ValueError("Please enter a list for seed_topics.")
            if not type(seed_confidence) == float:
                raise ValueError("Please enter a float for seed_confidence.")
            print("Guided LDA")
            model = guidedlda.GuidedLDA(n_topics=n_topcs, n_iter=n_iter, 
                random_state=random_state, refresh=refresh)
            model.fit(dtm, seed_topics, seed_confidence)
        elif not guided:
            print("Regular LDA")
            model = guidedlda.GuidedLDA(n_topics=n_topics, n_iter=n_iter, 
                random_state=random_state, refresh=refresh)
            model.fit(dtm)
        return model

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
       

