# Component Design for News NLP

![ComponentDesignFlowChart](news-nlp-flowchart-2.png?raw=true)

### UI

The UI provides an interface for the user to interact. The interface is primarily a text box where the user can input and submit text to be analyzed or used for searching the corpus of articles. The text is passed on to other models for comparison and visualization purposes.

**Inputs:**  
* `None`

**Outputs:**  
* `article_text`: A single stream of text representing an article. 

### Preprocessing (Article Corpus version)
The Preprocessing component is in charge of transforming the data in such a way that is ingestible by the other components in the system. This would involve things like tokenizing words, converting words into lowercase, removing punctuation and stop words,  lemmatizing, as well as limiting the vocabulary.

**Inputs:**  
* `articles_corpus`: Input corpus of articles to preprocess and train on.
* `preprocess_configurations`: This would help tell the preprocessor what actions to take. For example, the maximum vocabulary size, minimum usage of a word, whether to lemmetize, whether to take tf-idf scores, etc.

**Outputs:**  
* `articles`: A document-term-matrix with words as columns and rows as article ID's, with a binary indicator for whether the word appears in the article. 

### Preprocessing (Query version)
The "Query" version of Preprocessing is the pre-trained Preprocessing component with its vocabulary built-in. It will run the same pipeline used to preprocess the `articles_corpus` given the configuration settings, and will also limit the vocabulary to match that of the `articles_corpus`. 

**Inputs:**  
* `article_text`: A single stream of text representing an article. 

**Outputs:**  
* `article`: A bag-of-words representation of the article corresponding to a row in a document-term-matrix.


### Sentiment Analyzer
The Sentiment Analyzer is a component that takes an article and extracts a sentiment score. It splits the article into sentences and uses the NLTK Vader module to analyze the sentiments for each sentence. Based on these sentiments it returns an overall sentiment for the article. It aggregates and returns the number of positive, negative and neutral sentences.

**Inputs:**  
* `article_text`: A single stream of text representing an article. 

**Outputs:**  
* `article_sentiment`: A value of 'Positive', 'Negative' or 'Neutral'.  
* `pos_sentences`: Number of positive sentences.  
* `neg_sentences`: Number of negative sentences.  
* `neu_sentences`: Number of neutral sentences.  

### Word Cloud Generator
This component generates a word cloud based on the word occurrence in the article. It uses the WordCloud module. The output is an image for the word cloud.  

**Inputs:**  
* `article_text`: A single stream of text representing an article.    

**Outputs:**  
* `image` : An image for the word cloud 


### Guided LDA

The Guided LDA is the component that creates the topic model from the articles corpus. It is a variant of the popular Latent Dirichlet Allocation model used to model topics in a corpus of text. The guided portion of the LDA allows semi-supervised learning on a normally unsupervised LDA algorithm by allowing seed words to guide the topic modeling. It was chosen due to the improvement in interpretability of the resulting topics over regular LDA.

**Dimensions**:

* n = number of unique articles in corpus
* d = numer of unique relevant words in corpus
* k = number of desired topics in corpus

**Inputs**:

* `articles`: A bag-of-words representation of articles (document-term-matrix). Each row represents an article in the corpus, while each column represents a unique word in the corpus after preprocessing. dim = (n, d)

**Outputs**:

* `doc_topic`: Matrix relating documents (articles) with topics. dim = (n, k)

* `topic_word`: Matrix relating topics to words. dim = (k, d)

**Hyperparameters**:

* `n_topics`: Number of desired topics

* `n_iter`: Total number of iterations 

* `seed_topics`: List of lists containing desired words in each topic

* `seed_confidence`: Confidence of the seeded topics

**Relation with other components**:

`topic_word` and `doc_topic` are the topic model. `topic_word` is used to tag each article in the corpus with certain related topics. `doc_topic` is used to evaluate the created topics from the corpus, and also to assign topics to the query article so that k-NN algorithm can retreive recommended articles. 


### Topic Predictor

The topic predictor outputs relevant topics to a query article by using the topic model from the Guided LDA. 

**Inputs**:

* `topic_word`: Matrix relating topics to words. dim = (k, d)

**Outputs**:

* `query_article_topics`: A vector of topic probabilities. Equivalent to a single row in `doc_topic`, just for the query article. dim = (1, k)


### k-NN

k-NN is the component that takes the query article topics generated from `topic predictor` and the doc_topic generated from the `Guided LDA` as inputs, and then outputs the article ids of the most relavant articles from doc_topic matrix which will then be used by `Join Process` to link back to the original article database. 

**Inputs**: 

* `topics`: Topics generated from Topic Predictor on query article
* `doc_topic`: Matrix relating documents (articles) with topics from Guided LDA

**Outputs**:

* `article id`: Based on K-NN algorithm, output most relevant article ids from the doc_topic matrix


### Join Process
	
The Join Process component will take the article ids generated by `K-NN` and join them back with the original article database to return the full recommended articles for visualization interface.

**Inputs**:

* `article id`: Ids generated from `K-NN` indicating the row index of most relavant articles in the doc-topic matrix
* `article`: Original articles from the corpus

**Outputs**:

* `recommended articles`: Articles from the corpus that are most recommended based on topic prediction


### Visualization

The visualization component takes in the output of the different models and displays the data in 3 graphical forms. These are 1) listing the recommended articles with links, 2) plotting the query sentiment data and 3) creating a word cloud with associated topics. A sample mockup is below.

**Inputs**:
* `Recommended articles`: Articles from the corpus that are most recommended based on topic prediction
* `Query topic results`: A vector of topic probabilities. Equivalent to a single row in `doc_topic`, just for the query article. dim = (1, k)
* `article_sentiment`: A value of 'Positive', 'Negative' or 'Neutral'.  
* `pos_sentences`: Number of positive sentences.  
* `neg_sentences`: Number of negative sentences.  
* `neu_sentences`: Number of neutral sentences.  


**Outputs**:
* `Recommended articles`: A printed list of the top recommended articles
* `Sentiment visualization`: A graph showing the sentiment of the input query
* `Topic word cloud`: A word cloud of the top topics for the input query
