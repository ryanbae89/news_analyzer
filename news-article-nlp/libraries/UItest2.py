# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_html_components as html
import base64
import pandas as pd
import os
import numpy as np

import handler

app = dash.Dash()
image_filename = '/Users/paulwright/Dropbox/UW/2018_s_DATA515/Project/UI_Testing/WC_Image.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())
encoded_image2 = base64.b64encode(open('img_test2.jpg', 'rb').read())
my_handler = handler.Handler()

image_filename = 'WC_Image.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app.layout = html.Div([
    
])

global rec_art_test

# Reusable
def make_dash_table(df):
    ''' Return a dash definitio of an HTML table for a Pandas dataframe '''
    table = []
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table

def get_logo():
    logo = html.Div([

        html.Div([
            html.Img(src='./WC_Image.png')
        ], className="ten columns padded"),

        html.Div([
            dcc.Link('Full View   ', href='/full-view')
        ], className="two columns page-view no-print")

    ], className="row gs-header")
    return logo

app.layout = html.Div([
    dcc.Textarea(placeholder = "Enter text / article here...",
        style = {'width': '100%'}, id='input-1-state'),#, type='text', value='Montréal'),
    html.Button(id='submit-button', n_clicks=0, children='Submit'),
    html.Div(id='output-state'),
    html.Div(id='output-state2'),
    #html.Img(src='data:image/png;base64,{}'.format(encoded_image)),
    #html.Img(src='data:image/png;base64,{}'.format(encoded_image)),
    html.Div([
        html.Div([
            html.H6(["Recommended Articles"],
                    className="gs-header gs-table-header padded"),
            html.Table(make_dash_table(
                            pd.DataFrame(np.asarray([['first article title','NYTimes'],
                                                    ['second article title','Breitbart'],
                                                    ['third article title','NY Post']]))
                                       ))
            ], id = "table_data", className="six columns"),
        html.Div([
            html.Img(src='https://www.w3schools.com/images/w3schools_green.jpg', 
                 #height='142', 
                 width='100%', 
                 alt = "test image")
        ], className="six columns")
    ], className="row "),
    html.Div([
        html.Div([
            html.H6(["Top Topics"],
                    className="gs-header gs-table-header padded"),
            html.Table(make_dash_table(pd.DataFrame(np.asarray([['a','aa'],
                                                    ['b','aa'],
                                                    ['c','aa']]))))
            ], className="six columns"),
        html.Div([
            html.Img(src='https://www.w3schools.com/images/w3schools_green.jpg', 
                 #height='142', 
                 width='100%', 
                 alt = "test image")
        ], className="six columns")
    ], className="row "),
    html.Div([
        html.Div([
            html.H6(["Top Topics"],
                    className="gs-header gs-table-header padded"),
            html.Table(make_dash_table(pd.DataFrame(np.asarray([['a','aa'],
                                                    ['b','aa'],
                                                    ['c','aa']]))))
            ], className="six columns"),
        html.Div([
            dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'}
            ],
            'layout': {
                'title': 'Sentiment Analysis?'
            }
        }
    )
        ], className="six columns")
    ], className="row ")
    #html.Div([
    #    html.Img(src='https://www.w3schools.com/images/w3schools_green.jpg', 
    #         height='142', 
    #         width='104', 
    #         alt = "test image")
    #], className="ten columns padded")

        
], className="page")

# WC_image.png
# img_test2.jpg


@app.callback(Output('output-state', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('input-1-state', 'value')])

def update_output(n_clicks, input1):
    return u'''
        The Button has been pressed {} times,
        Input 1 is "{}"
    '''.format(n_clicks, input1) 

#@app.callback(Output('output-state2', 'children'),
#              [Input('submit-button', 'n_clicks')],
#              [State('input-1-state', 'value')])

#def update_output2(n_clicks, input1):
#    my_image = '<img alt="test image" src="https://www.w3schools.com/images/w3schools_green.jpg" width="100%">'
#    return u'''
#        The button has been pressed {} times,
#        Input 1 is "{}"
#    '''.format(n_clicks*2, input1)

#################################

#@app.callback(Output('output-state2', 'children'),
#              [Input('submit-button', 'n_clicks')],
#              [State('input-1-state', 'value')])

#def update_output2(n_clicks, query_article):
#    return my_handler.get_recommended_articles(query_article)


@app.callback(Output('table_data', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('input-1-state', 'value')])

def update_output2(n_clicks, query_article):
    val = my_handler.get_recommended_articles(query_article)
    return [
            html.H6(["Recommended Articles"],
                    className="gs-header gs-table-header padded"),
            html.Table(make_dash_table(
                        					val
                                       ))
            ]



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
    rec_art_test = pd.DataFrame([['first article title','NYTimes'],
                    ['second article title','Breitbart'],
                    ['third article title','NY Post']])
    app.run_server(debug=True)
    
    
    