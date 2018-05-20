"""
word_cloud_generator

This module has 1 function:
    generate_wordcloud (see more details below)
"""
from wordcloud import WordCloud, STOPWORDS
STOP_WORDS = set(STOPWORDS)


def generate_wordcloud(input_string):
    """
    A function that generates a Word Cloud based on the input string.

    Args:
            input_string (string): string representing the query article.

    Returns:
            image: representing the word cloud image

    """
    wordcloud = WordCloud(
        background_color='white',
        stopwords=STOP_WORDS,
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=1
    ).generate(str(input_string))

    return wordcloud
