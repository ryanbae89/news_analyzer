# Component Design for News NLP

### UI

### Preprocessing (Article Corpus version)

### Preprocessing (Query version)

### Build Sentiment Model

### Guided LDA

### Sentiment Predictor

### Topic Predictor

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


\includepdf{news-nlp-flowchart.pdf}