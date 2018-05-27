
# Setup module for news-articles-nlp

import os
from setuptools import setup, find_packages
PACKAGES = find_packages()

opts = dict(name='news-articles-nlp',
            maintainer='',
            maintainer_email='',
            description='News Article Recommendation System',
            long_description=('news-articles-nlp'),
            url='https://github.com/heybaebae/news-articles-nlp',
            # download_url="DOWNLOAD_URL",
            license='MIT',
            # classifiers="CLASSIFIERS",
            author='',
            author_email='',
            version='1',
            packages=PACKAGES,
            package_data={'news-articles-nlp': ['resources/*', 'tests/test_resources/*']},
            # install_requires="REQUIRES",
            # requires="REQUIRES"
            )

if __name__ == '__main__':
    setup(**opts)
	import nltk
	nltk.download('punkt')
	nltk.download('vader_lexicon')
	nltk.download('stopwords')
	nltk.download('wordnet')