import json
import requests
import pandas as pd
from get_btc import btc

with open('api.txt') as api:
    api = api.readlines()


def glassnode(url):
    # insert your API key here
    API_KEY = api
    # make API request
    res = requests.get(f'https://api.glassnode.com{url}',
        params={'a': 'BTC', 'api_key': API_KEY})

    # convert to pandas dataframe
    df = pd.read_json(res.text, convert_dates=['t'])
    df.columns = ['timestamp',url.split('/')[-1]]
    return df

def on_chain_df(categorie, metric):
    url =  '/v1/metrics/' + str(categorie)  +'/' + str(metric)
    df = glassnode(url).copy()
    df.set_index('timestamp',inplace = True)
    df.columns = ['metric']
    return df


def on_chain_merge(metric):
    df_btc = btc().copy()
    return df_btc

    