# conda install -c conda-forge wordcloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def generate_wordcloud(data):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=1
    ).generate(str(data))

    #fig = plt.figure(1, figsize=(20, 12))
    #plt.axis('off')
    #fig.show()
    return wordcloud


    #plt.imshow(wordcloud)
    #plt.show()