
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split,ParameterGrid
from itertools import product, chain
from tqdm import tqdm
from catboost import CatBoostClassifier, Pool, cv
from numba import jit
import tweepy
from tweepy import OAuthHandler
import matplotlib.pyplot as plt
import nltk
from collections import Counter
import plotly
from plotly import graph_objs


RANDOM_STATE = 0




data=pd.read_csv("FinalDS_AdditionalFeatures.csv")
test=pd.read_csv("tweets.csv")



data=data[["text","created_at","retweet_count","favorite_count","source","length","user_id","user_screen_name","user_name","user_created_at","user_description","user_followers_count","user_friends_count","user_location","user_statuses_count","user_verified","user_url","tokenized_text","token_texts","no_of_question_marks","no_of_exclamation_marks","no_of_hashtags","no_of_mentions","cleaned_text","no_of_colon_marks","no_of_words","no_of_uppercase_words","user_has_url?","Final Label"]]
test=test[["text","created_at","retweet_count","favorite_count","source","length","user_id","user_screen_name","user_name","user_created_at","user_description","user_followers_count","user_friends_count","user_location","user_statuses_count","user_verified","user_url","tokenized_text","token_texts","no_of_question_marks","no_of_exclamation_marks","no_of_hashtags","no_of_mentions","cleaned_text","no_of_colon_marks","no_of_words","no_of_uppercase_words","user_has_url?"]]




test['user_has_url?'] = test['user_has_url?'].map( {'Yes':1, 'No':0} )
test['user_verified'] = test['user_verified']*1
test['user_has_url?'].fillna('-999999', inplace=True)





data['Final Label'] = data['Final Label'].map( {'FAKE':1, 'REAL':0} )
data['user_has_url?'] = data['user_has_url?'].map( {'Yes':1, 'No':0} )
data['user_verified'] = data['user_verified']*1





missing_val_count_by_column = (data.isna().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])




data['created_at'].fillna('00-00-0000 00:00', inplace=True)
data['user_description'].fillna('Unknown_descr', inplace=True)
data['user_location'].fillna('Unknown_location', inplace=True)
data['user_statuses_count'].fillna('Unknown', inplace=True)
data['user_verified'].fillna('0', inplace=True)
data['user_url'].fillna('Unknown_url', inplace=True)




test['created_at'].fillna('00-00-0000 00:00', inplace=True)
test['user_description'].fillna('Unknown_descr', inplace=True)
test['user_location'].fillna('Unknown_location', inplace=True)
test['user_statuses_count'].fillna('Unknown', inplace=True)
test['user_verified'].fillna('0', inplace=True)
test['user_url'].fillna('Unknown_url', inplace=True)






Y_train=data["Final Label"]
data.drop("Final Label",axis=1,inplace=True)
X_train=data
X_test=test
cat_features=["text","created_at","source","user_screen_name","user_url","user_name","user_created_at","user_description","user_location","tokenized_text","token_texts","cleaned_text"]



#this code is to train the classifier
#here I already trained the model, saved it and have it ready for use
'''
train_pool = Pool(X_train, Y_train, cat_features=cat_features)

model = CatBoostClassifier(
    l2_leaf_reg=5,
    learning_rate=0.01,
    depth=3,
    #depth=3,
    iterations=2000,
    eval_metric='AUC',
    od_wait=50,
    random_seed=42,
    loss_function='Logloss'
)

cv_data = cv(
    train_pool,
    model.get_params(),
    fold_count=5,
    plot="True"
)

model.fit(train_pool);
model.score(X_train, Y_train)
'''

import pickle
filename = 'catboost_model.sav'
model1 = pickle.load(open(filename, 'rb'))

model1.predict(test)

#this is to get rid of emoticons and leave only text of the tweets
import re
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
 
def tokenize(s):
    return tokens_re.findall(s)

def preproces(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens
def count_by_lambda(expression, word_array):
    return len(list(filter(expression, word_array)))
def count_occurences(character, word_array):
    counter = 0
    for j, word in enumerate(word_array):
        for char in word:
            if char == character:
                counter += 1
    return counter
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# Store the tweets in a dataframe
def process_results(results):
    id_list = [tweet.id for tweet in results]
    data_set = pd.DataFrame(id_list, columns=["id"])
        # Processing Tweet Data

    data_set["text"] = [tweet.text for tweet in results] #text of tweet
    data_set["created_at"] = [tweet.created_at for tweet in results] #when the tweet was created
    data_set["retweet_count"] = [tweet.retweet_count for tweet in results] #number of retweets
    data_set["favorite_count"] = [tweet.favorite_count for tweet in results] #number of favourites
    data_set["source"] = [tweet.source for tweet in results] #source of the tweet
    data_set["length"] = [len(tweet.text) for tweet in results] #number of characters in tweet

    # Processing User Data
    data_set["user_id"] = [tweet.author.id for tweet in results] #id of the author
    data_set["user_screen_name"] = [tweet.author.screen_name for tweet in results] 
    data_set["user_name"] = [tweet.author.name for tweet in results]
    data_set["user_created_at"] = [tweet.author.created_at for tweet in results] #age of user account
    data_set["user_description"] = [tweet.author.description for tweet in results]
    data_set["user_followers_count"] = [tweet.author.followers_count for tweet in results] #number of followers
    data_set["user_friends_count"] = [tweet.author.friends_count for tweet in results] #number of friends
    data_set["user_location"] = [tweet.author.location for tweet in results] #user has a location in profile?
    data_set["user_statuses_count"] = [tweet.author.statuses_count for tweet in results] #number of statuses
    data_set["user_verified"] = [tweet.author.verified for tweet in results] #user is verified?
    data_set["user_url"] = [tweet.author.url for tweet in results] #user has a URL?
    #calculates number of ?, !, hashtags and mentions
    data_set['tokenized_text']= data_set['text'].apply(preproces)
    data_set['token_texts'] = data_set['tokenized_text'].apply(lambda x : [w for w in x if w.lower() not in stop_words])  
    data_set['no_of_question_marks'] = data_set['token_texts'].apply(lambda txt: count_occurences("?", txt)) 
    data_set['no_of_exclamation_marks'] = data_set['token_texts'].apply(lambda txt: count_occurences("!", txt)) 
    data_set['no_of_hashtags'] = data_set['token_texts'].apply(lambda txt: count_occurences("#", txt)) 
    data_set['no_of_mentions'] = data_set['token_texts'].apply(lambda txt: count_occurences("@", txt))
    
    
    #Removes URLs
    data_set['cleaned_text'] = data_set['text'].apply(lambda txt:remove_url_by_regex("http.?://[^\s]+[\s]?",txt))
    #Removes mentions
    data_set['cleaned_text'] = data_set['cleaned_text'].apply(lambda txt:remove_url_by_regex(r'(?:@[\w_]+)',txt))
    #Calculates number of colon marks
    data_set['no_of_colon_marks'] = data_set['cleaned_text'].apply(lambda txt: count_occurences(":", txt)) 
    data_set['cleaned_text'] = data_set['cleaned_text'].apply(lambda txt:remove_url_by_regex(r'[,|:|\|=|&|;|%|$|@|^|*|-|#|?|!|.]',txt))
    data_set['no_of_words'] = data_set['cleaned_text'].apply(lambda txt:len(re.findall(r'\w+',txt)))
    data_set['no_of_uppercase_words'] = data_set['tokenized_text'].apply(lambda txt: count_by_lambda(lambda word: word == word.upper(),txt))
    
    
    return data_set
def remove_url_by_regex(pattern,string):
    return re.sub(pattern,"", string)


def scrapping(number,text):
    #put your own twitter developer key here
    consumer_key = 'xxx'
    consumer_secret = 'xxx'
    access_token = 'xxx'
    access_secret = 'xxx'

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
 
    api = tweepy.API(auth)
    results = []
    i=0
    for tweet in tweepy.Cursor(api.search, q=text, lang="en").items():
        if (not tweet.retweeted) and ('RT @' not in tweet.text):
            results.append(tweet)
            id = tweet.id
            i=i+1
            if i==number:
                break
    #print ("we did a scrapping of"+len(results+" tweet"))
    df = process_results(results)
    df['tokenized_text']= df['text'].apply(preproces)
    df["user_has_url?"] = np.where(df["user_url"].isnull(), '0', '1')
    
    df['user_verified'] = df['user_verified']*1
    df['created_at'].fillna('00-00-0000 00:00', inplace=True)
    
    df['user_statuses_count'].fillna('Unknown', inplace=True)
    df['user_verified'].fillna('0', inplace=True)
    df['user_url'].fillna('Unknown_url', inplace=True)
    df["created_at"]=df["created_at"].apply(lambda x: str(x))
    df["user_created_at"]=df["user_created_at"].apply(lambda x: str(x))
    df['user_has_url?'] = df['user_has_url?'].map( {'1':1, '0':0} )
    df["user_id"]=df["user_id"].apply(lambda x: float(x))
    df=df[["text","created_at","retweet_count","favorite_count","source","length","user_id","user_screen_name","user_name","user_created_at","user_description","user_followers_count","user_friends_count","user_location","user_statuses_count","user_verified","user_url","tokenized_text","token_texts","no_of_question_marks","no_of_exclamation_marks","no_of_hashtags","no_of_mentions","cleaned_text","no_of_colon_marks","no_of_words","no_of_uppercase_words","user_has_url?"]]

    df.dropna(how='any',axis=0,inplace=True) 
    return df

#####################################################################################################################
import pickle
from nltk.tokenize import punkt
from nltk.corpus.reader import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import re


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# Colors
colors = {
    'background': '#ECECEC',  
    'text': '#696969',
    'titles': '#599ACF',
    'blocks': '#F7F7F7',
    'graph_background': '#F7F7F7',
    'banner': '#C3DCF2'

}

# Markdown text
markdown_text1 = '''
This application serves to scrape tweets about a given subject in real time. Then, each Tweet text is analyzed using Natural Language Processing to detect whether the news are real or fake based on several features. Finally, the results are visualized in a graph and a pie chart.
'''

markdown_text2 = '''
 Created by FRIJI Hamdi,Med Ben Nasser Chiheb and GHARBI Oumaima
'''



app.layout = html.Div(style={'backgroundColor':colors['background']}, children=[
    
    # Space before title
    html.H1(children=' ',
            style={'padding': '10px'}
           ),
    
    # Title
    html.Div(
        [
            html.H3(children='Tweets Classification App',
                    style={"margin-bottom": "0px"}
                   ),
            html.H6(children='by Supcom students')
        ],
        style={
            'textAlign': 'center',
            'color': colors['text'],
            #'padding': '0px',
            'backgroundColor': colors['background']
              },
        className='banner',
            ),
    

    # Space after title
    html.H1(children=' ',
            style={'padding': '1px'}),


    # Text boxes
    html.Div(
        [
            html.Div(
                [
                    html.H6(children='What does this app do?',
                            style={'color':colors['titles']}),
                    
                    html.Div(
                        [dcc.Markdown(children=markdown_text1),],
                        style={'font-size': '12px',
                               'color': colors['text']}),
                                        
                    html.Div(
                         [
                         dcc.Input(
                        
                         type="search",
                         id='checklist'
                                )
        
                        ],
                        
                        style={'font-size': '12px',
                               'margin-top': '25px'}),
                    
                    html.Div([
                       
                        html.Button('Scrape', 
                                    id='submit', 
                                    type='submit', 
                                    style={'color': colors['blocks'],
                                           'background-color': colors['titles'],
                                           'border': 'None'})],
                        style={'textAlign': 'center',
                               'padding': '20px',
                               "margin-bottom": "0px",
                               'color': colors['titles']}),
            
                    dcc.Loading(id="loading-1", children=[html.Div(id="loading-output-1")], type="circle"),
                    
                    html.Hr(),
                    html.H6(children='Fake Tweets',
                            style={'color': colors['titles']}),

                    # Headlines
                    html.A(id="textarea1a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea1b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea2a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea2b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea3a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea3b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea4a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea4b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea5a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea5b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea6a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea6b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea7a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea7b", style={'color': colors['text'], 'font-size': '11px'}),
                    html.A(id="textarea8a", target="_blank", style={'font-size': '12px'}),
                    html.P(id="textarea8b", style={'color': colors['text'], 'font-size': '11px'}),
                    
                    
                                                            
                ],
                     style={'backgroundColor': colors['blocks'],
                            'padding': '20px',
                            'border-radius': '5px',
                            'box-shadow': '1px 1px 1px #9D9D9D'},
                     className='one-half column'),
            
            html.Div(
                [
                    html.H6("Graphic summary",
                            style={'color': colors['titles']}),

                    html.Div([
                         dcc.Graph(id='graph1', style={'height': '300px'})
                         ],
                         style={'backgroundColor': colors['blocks'],
                                'padding': '20px'}
                    ),
                    
                    html.Div([
                         dcc.Graph(id='graph2', style={'height': '300px'})
                         ],
                         style={'backgroundColor': colors['blocks'],
                                'padding': '20px'}
                    )
                ],
                     style={'backgroundColor': colors['blocks'],
                            'padding': '20px',
                            'border-radius': '5px',
                            'box-shadow': '1px 1px 1px #9D9D9D'},
                     className='one-half column')

        ],
        className="row flex-display",
        style={'padding': '20px',
               'margin-bottom': '0px'}
    ),
    
        
    # Space
    html.H1(id='space2', children=' '),
        
    
    # Final paragraph
    html.Div(
            [dcc.Markdown(children=markdown_text2),],
            style={'font-size': '35px',
                   'color': colors['text']}),

    
    # Hidden div inside the app that stores the intermediate value
    html.Div(id='intermediate-value', style={'display': 'none'})
    

])


@app.callback(
    [
    Output('intermediate-value', 'children'),
    Output('loading-1', 'children')
    ],
    [Input('submit', 'n_clicks')],
    [State('checklist', 'value')])
def scrape_and_predict(n_clicks, value):
            
    dd=scrapping(50,value)
    dd.to_csv("dd.csv",index=False)
    import pickle
    filename = 'catboost_model.sav'
    model1 = pickle.load(open(filename, 'rb'))


    dd=pd.read_csv("dd.csv")
    dd['user_description'].fillna('Unknown_descr', inplace=True)
    dd['user_location'].fillna('Unknown_location', inplace=True)
    res=model1.predict(dd)
    df = pd.DataFrame(
        {'Article': dd.text.values,
         'label': res,
         "source":dd.source.values
         })
    
    # Put into dataset
    
    return df.to_json(date_format='iso', orient='split'), ' '

@app.callback(
    Output('graph1', 'figure'),
    [Input('intermediate-value', 'children')])
def update_barchart(jsonified_df):
    
    df = pd.read_json(jsonified_df, orient='split')
    df['label'] = df['label'].map( {1:'Fake', 0:'Real'} )
    # Create a summary df
    print()
    df_sum = df.groupby(['source', 'label']).count()["Article"]
    # Create x and y arrays for the bar plot for every newspaper
    if 'Twitter for Android' in df_sum.index:
    
        df_sum_epe = df_sum['Twitter for Android']
        x_epe = ['Fake', 'Real']
        y_epe = [[df_sum_epe['Fake'] if 'Fake' in df_sum_epe.index else 0][0],
                [df_sum_epe['Real'] if 'Real' in df_sum_epe.index else 0][0]
                ]   
    else:
        x_epe = ['Fake', 'Real']
        y_epe = [0,0]
    
    if 'Twitter Web App' in df_sum.index:
        
        df_sum_thg = df_sum['Twitter Web App']
        x_thg = ['Fake', 'Real']
        y_thg = [[df_sum_thg['Fake'] if 'Fake' in df_sum_thg.index else 0][0],
                [df_sum_thg['Real'] if 'Real' in df_sum_thg.index else 0][0]
                ]   
    else:
        x_thg = ['Fake', 'Real']
        y_thg = [0,0]

    if 'Twitter Web Client' in df_sum.index:
    
        df_sum_skn = df_sum['Twitter Web Client']
        x_skn = ['Fake', 'Real']
        y_skn = [[df_sum_skn['Fake'] if 'Fake' in df_sum_skn.index else 0][0],
                [df_sum_skn['Real'] if 'Real' in df_sum_skn.index else 0][0]
                ]   

    else:
        x_skn = ['Fake', 'Real']
        y_skn = [0,0]

    # Create plotly figure
    figure = {
        'data': [
            {'x': x_epe, 'y':y_epe, 'type': 'bar', 'name': 'Tweets published from Android', 'marker': {'color': 'rgb(62, 137, 195)'}},
            {'x': x_thg, 'y':y_thg, 'type': 'bar', 'name': 'Tweets published from Web App', 'marker': {'color': 'rgb(167, 203, 232)'}},
            {'x': x_skn, 'y':y_skn, 'type': 'bar', 'name': 'Tweets published from Web Client', 'marker': {'color': 'rgb(197, 223, 242)'}}
        ],
        'layout': {
            'title': 'Number of Tweets  by Source',
            'plot_bgcolor': colors['graph_background'],
            'paper_bgcolor': colors['graph_background'],
            'font': {
                    'color': colors['text'],
                    'size': '10'
            },
            'barmode': 'stack'
            
        }   
    }
    
    return figure

@app.callback(
    Output('graph2', 'figure'),
    [Input('intermediate-value', 'children')])
def update_piechart(jsonified_df):
    
    df = pd.read_json(jsonified_df, orient='split')
    
    # Create a summary df
    df['label'] = df['label'].map( {1:'Fake', 0:'Real'} )
    df_sum = df['label'].value_counts()
    # Create x and y arrays for the bar plot
    x = ['Fake','Real']
    y = [[df_sum['Fake'] if 'Fake' in df_sum.index else 0][0],
         [df_sum['Real'] if 'Real' in df_sum.index else 0][0]]
    print(y)
    
    # Create plotly figure
    figure = {
        'data': [
            {'values': y,
             'labels': x, 
             'type': 'pie',
             'hole': .4,
             'name': '% of Tweets ',
             'marker': {'colors': ['rgb(62, 137, 195)',
                                   'rgb(167, 203, 232)'
                                                        ]},

            }
        ],
        
        'layout': {
            'title': 'Fake news percentage',
            'plot_bgcolor': colors['graph_background'],
            'paper_bgcolor': colors['graph_background'],
            'font': {
                    'color': colors['text'],
                    'size': '10'
            }
        }
        
    }
    
    
    return figure
    
    
@app.callback(
    [
    
    Output('textarea1a', 'href'),
    Output('textarea1a', 'children'),
    
    Output('textarea2a', 'href'),
    Output('textarea2a', 'children'),
    
    Output('textarea3a', 'href'),
    Output('textarea3a', 'children'),
    
    Output('textarea4a', 'href'),
    Output('textarea4a', 'children'),
    
    Output('textarea5a', 'href'),
    Output('textarea5a', 'children'),
    
    Output('textarea6a', 'href'),
    Output('textarea6a', 'children'),
    
    Output('textarea7a', 'href'),
    Output('textarea7a', 'children'),
    
    Output('textarea8a', 'href'),
    Output('textarea8a', 'children'),
    
   
    
    

    ],
    [Input('intermediate-value', 'children')])
def update_textarea1(jsonified_df):
    
    df = pd.read_json(jsonified_df, orient='split')
    
    print(df)
    #df.head()
    texts = []
    label = []
    preds_newsp = []
    texts=df[df["label"]==1].Article.values
    links=df["Article"].apply(lambda x : re.findall(r'(https?://\S+)', x))
    for i in range(0,len(links)):
        try :
            links[i][0]
        except:
            links[i]=["https://twitter.com/home?lang=fr"]
            
    print(links)
    return \
         links[0][0], texts[0],\
        links[1][0], texts[1], \
        links[2][0], texts[2], \
        links[3][0], texts[3], \
        links[4][0], texts[4], \
        links[5][0], texts[5], \
        links[6][0], texts[6], \
        links[7][0], texts[7]
        
         
           
    
    
# Loading CSS
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})


if __name__ == '__main__':
    app.run_server(debug=False)



