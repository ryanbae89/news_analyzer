# -*- coding: utf-8 -*-
"""
UI Python Module

This module uses dash and the handler script to create and update the UI.

The main components handled are the text box for input, the submit button, the recommended articles
box, the word cloud, the article sentiment display and the top topics. Each has their own call to
the handler class to retrieve the data.

"""

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd
import numpy as np
import handler

app = dash.Dash()
my_handler = handler.Handler()

# Reusable
def make_dash_table(data_frame):
    ''' Return a dash definition of an HTML table for a Pandas dataframe '''
    table = []
    for index, row in data_frame.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table

# Main layout
app.layout = html.Div([
    html.H2(["News Articles Recommender and Analyzer"],
            className="padded"),
    dcc.Textarea(value="trump White House government",
                 #placeholder = "Enter text / article here...",
                 style={'width': '100%'},
                 id='input-1-state'),
    html.Button(id='submit-button', n_clicks=0, children='Submit'),
    #html.Div(id='output-state'),
    #html.Div(id='output-state2'),
    html.Div([
        html.Div([
            html.H6(["Recommended Articles"],
                    className="gs-header gs-table-header padded"),
            html.Table(make_dash_table(
                pd.DataFrame(np.asarray([]))))
            ], id="recommended_articles", className="six columns"),
        html.Div([
            #html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
            #         width='100%')
        ], id="wc_image", className="six columns")
    ], className="row "),
    html.Div([
        html.Div([
            html.H6(["Article Sentiment"],
                    className="gs-header gs-table-header padded"),
            html.Table(make_dash_table(pd.DataFrame(np.asarray([]))))
            ], id='article_sentiment', className="six columns"),
        html.Div([
            html.H6(["Top Topics"],
                    className="gs-header gs-table-header padded"),
            html.Table(make_dash_table(pd.DataFrame(np.asarray([]))))
            ], id='top_topics', className="six columns")
    ], className="row ")
])#, className="page")

# Recommended Articles Data
@app.callback(Output('recommended_articles', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('input-1-state', 'value')])

def update_recommended_articles(n_clicks, query_article):
    """ Updates recommended articles.
    inputs:
    	n_clicks: button clicks, used for trigger in dash
    	query_article: string, article or words being queried
    """
    recommended_articles = my_handler.get_recommended_articles(query_article)
    return [
        html.H5(["Recommended Articles"],
                className="gs-header gs-table-header padded"),
        html.Table(make_dash_table(pd.DataFrame(recommended_articles)))
        ]

# Article Sentiment Data
@app.callback(Output('article_sentiment', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('input-1-state', 'value')])

def update_sentiment_information(n_clicks, query_article):
    """ Updates sentiment component with positive, neutral and negative sentence counts.
    inputs:
    	n_clicks: button clicks, used for trigger in dash
    	query_article: string, article or words being queried
    """
    sentiments = my_handler.get_sentiment(query_article)
    sentiment_df = pd.DataFrame(np.asarray([
        ['Positive Sentences:',
         str(round(sentiments['Positive_Sentences']/sentiments['Total_Sentences'], 3)*100)[:4]+'%'],
        ['Neutral Sentences:',
         str(round(sentiments['Neutral_Sentences']/sentiments['Total_Sentences'], 3)*100)[:4]+'%'],
        ['Negative Sentences:',
         str(round(sentiments['Negative_Sentences']/sentiments['Total_Sentences'], 3)*100)[:4]+'%']
    ]))
    return [
        html.H5(["Article Sentiment"],
                className="gs-header gs-table-header padded"),
        html.Table(make_dash_table(sentiment_df))
        ]

# Top Topics
@app.callback(Output('top_topics', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('input-1-state', 'value')])

def update_top_topics(n_clicks, query_article):
    """ Updates top topics from query article.
    inputs:
    	n_clicks: button clicks, used for trigger in dash
    	query_article: string, article or words being queried
    """
    top_topics = pd.DataFrame(my_handler.get_topics(query_article)[0])
    return [
        html.H5(["Top Topics"],
                className="gs-header gs-table-header padded"),
        html.Table(make_dash_table(top_topics))
        ]

# Word Cloud Image
@app.callback(Output('wc_image', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('input-1-state', 'value')])

def update_word_cloud_image(n_clicks, query_article):
    """ Updates word cloud image.
    inputs:
    	n_clicks: button clicks, used for trigger in dash
    	query_article: string, article or words being queried
    """
    word_cloud_image = my_handler.get_word_cloud(query_article)
    return [
        html.Img(src='data:image/png;base64,{}'.format(word_cloud_image.decode()),
                 width='100%')
        ]


# Configuration and stye dependencies:
external_css = ["https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
                "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
                "//fonts.googleapis.com/css?family=Raleway:400,300,600",
                "https://codepen.io/bcd/pen/KQrXdb.css",
                "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"]

for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ["https://code.jquery.com/jquery-3.2.1.min.js",
               "https://codepen.io/bcd/pen/YaXojL.js"]

for js in external_js:
    app.scripts.append_script({"external_url": js})

if __name__ == '__main__':
    app.run_server(debug=True)
