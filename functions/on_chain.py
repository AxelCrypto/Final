import json
import requests
import pandas as pd
from functions.get_btc import btc

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
    try:
        pd.read_csv(f'../data/datos/{metric}.csv')
    except:
        url =  '/v1/metrics/' + str(categorie)  +'/' + str(metric)
        df = glassnode(url).copy()
        df.set_index('timestamp',inplace = True)
        df.columns = ['metric']
        df.to_csv(f'../data/datos/{metric}.csv')
    return df


def on_chain_merge(categorie, metric):
    df = on_chain_df(categorie, metric).copy()
    df_btc = btc().copy()
    df_merged = df_btc.merge(df, left_index=True, right_on='timestamp')
    
    return df_merged


on_chain_merge('addresses', 'count')
    
    

    