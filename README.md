# NARA: News Articles Recommender and Analyzer

[![Build Status](https://travis-ci.org/heybaebae/news_analyzer.svg?branch=master)](https://travis-ci.org/heybaebae/news_analyzer)

[![Coverage Status](https://coveralls.io/repos/github/heybaebae/news_analyzer/badge.png?branch=master)](https://coveralls.io/github/heybaebae/news_analyzer?branch=master)

## Background  

Conventional news recommendation systems use a small set of keywords to identify the top recommended articles to users based on keywords frequency. We built upon this framework by enabling users to find recommended articles by providing an entire news article. The recommendation process uses article topics to evaluate which articles to recommended. 

Our system highlights topic insights through two different models: an unguided LDA for identifying topic recommended articles and a guided LDA with seed words from NYTimes to show interpretable topics. Our system also shows sentiment information and word cloud that summarize the query article for the user.

![UIScreenshot](doc/ui_screenshot.png?raw=true)  
  
NARA does 3 things:
* Recommends news articles based on topic relevance
* Analyzes sentiment of news articles
* Visualizes news articles

How does it work?
* Takes user input query article or keyword/phrases from UI
* Recommends relevant articles from corpus using LDA
* Lists relevant topics for query article
* Presents sentiment analysis based on number of positive/negative/neutral sentences in input
* Visualizes query article using word cloud

*For more details, please see*: [FunctionalDesign.md](doc/FunctionalDesign.md)

## Data

### All-the-news Kaggle dataset  
* The corpus comes from [All-the-news Kaggle dataset](https://www.kaggle.com/snapcrack/all-the-news):

It consists of over 140,000 articles from 15 US national publishers between 2015 - 2017. The distribution of the publishers is shown below:

![publishers distribution](http://i.imgur.com/QDPtuEv.png)

### The New York Times seed words dataset
* [The New York Times API](https://developer.nytimes.com):

We used labelled article information from the New York Times to seed words for the guided LDA. We aggregated and analyzed the article titles, summaries and categories from a series of days to generate the list of seed words.

## Installation

1. First clone the repo in your local directory. Then in the repo root directory, run the set up file to install the dependencies:  

```
pip install -r requirements.txt
```

2. Now, run the `setup.py` file as following to download other data dependencies:  

```
python setup.py build 
``` 
*Currently NARA only supports Python 3.4 or newer.*  


## Demo Example

Because the full news articles corpus and resulting topic models are too large to be stored in the repo, we provide a smaller corpus that is subset of the full corpus. This demo section goes over how to run Now to run the user interface to start using NARA.

1. Follow the installations steps above

2. Run the Dash UI
```
python path_to_libraries/user_interface.py
```

3. Copy the url that shows up in your command line to your browser to start the UI.

4. Enter any sample article from the examples folder or of your choice  

## Full Build

Because of the large sizes of the corpus and resulting topic models, it is recommended that you clone the repo locally and run the build scrip we provide to download the data from Kaggle and build the topic models. To do this, simply run the `build_resources.py` script as follows:

```
python path_to_scripts/build_resources.py
```

The build script will automatically download the csv files using Kaggle's API, and build the topic models. Once complete, you can simply run the UI as shown above in the demo example.

## Component Design  
![ComponentDesignFlowChart](doc/news-nlp-flowchart-2.png?raw=true)  
*For more details, please see*: [ComponentDesign.md](doc/ComponentDesign.md)

## Directory Structure

```
news_analyzer
├── LICENSE
├── README.md
├── doc
│   ├── ComponentDesign.md
│   ├── Final_presentation.pptx
│   ├── FunctionalDesign.md
│   ├── WeReadTheNews.pptx
│   ├── mockup.png
│   ├── ui_screenshot.png
│   └── news-nlp-flowchart-2.png
├── examples
│   ├── example_article1.txt
│   ├── example_article2.txt
│   └── example_article3.txt
├── news_analyzer
│   ├── __init__.py
│   ├── libraries
│   │   ├── __init__.py
│   │   ├── article_recommender.py
│   │   ├── configs.py
│   │   ├── handler.py
│   │   ├── nytimes_article_retriever.py
│   │   ├── sentiment_analyzer.py
│   │   ├── temp_wordcloud.png
│   │   ├── text_processing.py
│   │   ├── topic_modeling.py
│   │   ├── user_interface.py
│   │   └── word_cloud_generator.py
│   ├── resources
│   │   ├── articles.csv
│   │   ├── guidedlda_model.pkl
│   │   ├── nytimes_data
│   │   │   ├── NYtimes_data_20180507.csv
│   │   │   ├── NYtimes_data_20180508.csv
│   │   │   ├── NYtimes_data_20180509.csv
│   │   │   ├── NYtimes_data_20180513.csv
│   │   │   ├── NYtimes_data_20180515.csv
│   │   │   ├── NYtimes_data_20180527.csv
│   │   │   ├── NYtimes_data_20180528.csv
│   │   │   ├── NYtimes_data_20180529.csv
│   │   │   ├── NYtimes_data_20180603.csv
│   │   │   └── aggregated_seed_words.txt
│   │   ├── preprocessor.pkl
│   │   └── unguidedlda_model.pkl
│   ├── scripts
│   │   └── build_resources.py
│   └── tests
│       ├── __init__.py
│       ├── test_article_recommender.py
│       ├── test_handler.py
│       ├── test_nytimes_article_retriever.py
│       ├── test_preprocessing.py
│       ├── test_resources
│       │   ├── test_dtm.pkl
│       │   └── test_topics_raw.pkl
│       ├── test_sentiment_analyzer.py
│       ├── test_topic_modeling.py
│       ├── test_user_interface.py
│       └── test_word_cloud_generator.py
├── requirements.txt
└── setup.py
```

## Team:WeReadTheNews
MS Data Science, University of Washington  
DATA 515 Software Design for Data Science (Spring 2018)  

Team members:  
 * [Ryan Bae](http://www.linkedin.com/in/ryanbae89)    
 * [Crystal Ding](https://www.linkedin.com/in/yumeng-crystal-ding)  
 * [Charles Duze](https://www.linkedin.com/in/charlesduze)    
 * [Mohammed Helal](https://www.linkedin.com/in/mohammed-helal-78969566)   
 * [Paul Wright](https://www.linkedin.com/in/paulcharleswright)     

## References:

* [Guided LDA Module](https://medium.freecodecamp.org/how-we-changed-unsupervised-lda-to-semi-supervised-guidedlda-e36a95f3a164) 
