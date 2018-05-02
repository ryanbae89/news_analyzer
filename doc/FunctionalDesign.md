### What problem you're trying to solve. Why it's important. How your users will benefit.

News article recommendation systems are often driven by article importance and article popularity which can lead to articles not being curtailed to an individual. We aim to create an interface to display and interact with articles (including a text search) so users can understand what topics they are looking at and what other related articles they might be interested in. As part of the article analysis, we will display article sentiment and topic information.

To create this tool, we use a Kaggle dataset containing approximately 140,000 articles. The dataset includes article dates, publisher, author, title and content. Using a variant of a Latent Dirichlet Allocation (LDA) model, we categorize the articles using the title and content into predefined buckets (e.g. 'technology' or 'politics'). These topics are used both for classification and for visualization.

We use word sentiment scores from XXX to calculate the article sentiments. The sentiment is used primarily for visualization purposes so the user can see the spectrum of sentiments and how their particular article compares.

### Who will use your system? What level of computer experience do they require? What domain knowledge must they have?

The main users of this tool are the general public interested in reading news and journalists doing research.

In both cases, the tool provides an exploratory way of looking into the types of articles they are interested in (or not interested in). The user needs little programming knowledge, they simply need to input text into a text box and submit. From here, the visualizations will update with relevant topics, sentiment information and recommended articles. The level of domain knowledge is low. Aside from understanding the different pre-configured buckets, everything else is intuitive.


### Use cases. How will users interact with the system and how will the system respond. You likely want to have "mock ups" of screenshots and indicate how users will interact.

The tool provides a few insights into article attributes that can be used in different ways. The first is that it provides recommended articles based off of the inputted text. The output also includes sentiment information enabling users to search and compare article sentiments. Lastly, the output provides a list of related topics.

The user is presented with a text box where they may enter words, sentences or an article. The text is analyzed and compared to the dataset of articles to create the four main visualizations for the user to see:

#### Recommended Articles

The interface provides a list of articles that have similar topics to the inputted text. Each result (XXXX) is a link to read the associated article.

#### Sentiment Analysis

The sentiment of the inputted text is analyzed and visualized. As a stretch use case, we may also provide a histogram of the sentiments of all of the articles in the dataset grouped by publisher.

#### Topic Insights

The top sentiments of the inputted text are listed/visualized to enable the user to identify and interact with (XXX) the different topics. 

#### Word Cloud Summary

The most common words of the inputted text are provided as a word cloud, so at a glance the user can get a quick idea of what the article or text is about.  

### Mockups:

(Paul will draft something and add in later Monday)
