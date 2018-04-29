### Component Design For KNN Search Algorithm

**Name**: KNN_Search

**What it does**: Use KNN algorithm to recommend similar articles to the input article or keywords

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