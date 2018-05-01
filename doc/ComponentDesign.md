# Component Design for News NLP


### UI

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
* `article`: A bag-of-words representation of the article corresponding to a row in a document-term-matrix.

**Outputs:**  
* `article_sentiment`: A value of 'Positive', 'Negative' or 'Neutral'.  
* `pos_sentences`: Number of positive sentences.  
* `neg_sentences`: Number of negative sentences.  
* `neu_sentences`: Number of neutral sentences.  

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

### Word Cloud Generator
This component generates a word cloud based on the word occurrence in the article. It uses the WordCloud module. The output is an image for the word cloud.  

**Inputs:**  
* `article`: Input article to analyze   

**Outputs:**  
* `image` : An image for the word cloud  


### Topic Predictor

The topic predictor outputs relevant topics to a query article by using the topic model from the Guided LDA. 

**Inputs**:

* `topic_word`: Matrix relating topics to words. dim = (k, d)

**Outputs**:

* `query_article_topics`: A vector of topic probabilities. Equivalent to a single row in `doc_topic`, just for the query article. dim = (1, k)

### KNN Search Algorithm

This component have 3 functions: `Collect User Input` retrieve random article or keywords from the interface and feeds it into Topic Scoring and Sentiment Analysis system. `KNN Search` models input article with original article database and output recommended articles based on distance in topic and sentiment. `Output to Interface` sends the recommended articles back to the user interface.

**Name**: KNN_Search

**Functions**: Use KNN algorithm to recommend similar articles to the input article or keywords

**Inputs**: 

* topic score(int or float): topic modeling score for each potential topics
* sentiment score(binary): sentiment encoding for articles
* random article or keyword(text): user input random article or keyword
* random article topic score(int or float): topic modeling score for user input article
* random article sentiment score(binary): sentiment encoding for user input article

**Outputs**:

* recommended articles(text): based on KNN search, output similar articles from database in regards to topic score and sentiment encoding

**Relation with other components**:

* Topic Analysis from Guided LDA: KNN Search uses topic score for articles in the database as well as user input article
* Sentiment Encoding: KNN Search uses sentiment encoding as another input for modeling distance between user input article and articles in the original database
* Output recommended articles into interface: After performing KNN search algorithm, certain number of similar/related articles will be pushed to user interface

**Sub-components**:

* User Interface for random article or keyword input
* User Interface for recommended articles output

### Join Process

### Visualization

