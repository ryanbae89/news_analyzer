# News Articles Recommender and Analyzer

[![Build Status](https://travis-ci.org/heybaebae/news_analyzer.svg?branch=master)](https://travis-ci.org/heybaebae/news_analyzer)
[![Coverage Status](https://coveralls.io/repos/github/heybaebae/news_analyzer/badge.svg?branch=master)](https://coveralls.io/github/heybaebae/news_analyzer?branch=master)

## Background  
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

## Data

The corpus comes from Kaggle dataset:
https://www.kaggle.com/snapcrack/all-the-news

It consists of over 140,000 articles from 15 US national publishers between 2015 - 2017. 

## Component Design  
![ComponentDesignFlowChart](doc/news-nlp-flowchart-2.png?raw=true)  
For more details, please see: [ComponentDesign.md](doc/ComponentDesign.md)

## Module Structure

## User Interface

## Preprocessor

## Topic Modeler

Topic modeling is done using Latent Dirichlet Allocation. 2 models are created for the recommender.

* Guided Model: 17 topics, hand picked for interpretation and topic tags
* Unguided Model: 1400 topics, LDA algorithm picks these, used for recommender

## Recommender

## Sentiment Analyzer 

## WeReadTheNews
University of Washington 
MS Data Science
DATA 515 Software Design for Data Science
Spring 2018

Team members  
 * [Ryan Bae](http://www.linkedin.com/in/ryanbae89)  
 * Crystal Ding  
 * Charles Duze  
 * Mohammed Helal  
 * Paul Wright   
 
