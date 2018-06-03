import os
import nltk
from setuptools import setup, find_packages
PACKAGES = find_packages()


opts = dict(
    name='news_analyzer',
    version='0.1',
    url='https://github.com/heybaebae/news_analyzer',
    license='MIT',
    author='WeReadTheNews',
    description='National news articles recommender and analyzer using LDA',
    packages=PACKAGES,
    package_data={'news_analyzer': ['data/*', 'tests/test_resources']}

)

if __name__ == '__main__':
    setup(**opts)
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    nltk.download('stopwords')
    nltk.download('wordnet')