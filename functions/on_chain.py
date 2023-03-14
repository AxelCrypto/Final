import json
import requests
import pandas as pd
from functions.get_btc import btc
import time

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
    time.sleep(1)
    df.columns = ['timestamp',url.split('/')[-1]]
    return df

def on_chain_df(categorie, metric):
    try:
        df = pd.read_csv(f'data/datos/{categorie}_{metric}.csv')
    except:
        url =  '/v1/metrics/' + str(categorie)  +'/' + str(metric)
        df = glassnode(url).copy()
        time.sleep(1)

        df.set_index('timestamp',inplace = True)
        df.columns = ['metric']
        df.to_csv(f'data/datos/{categorie}_{metric}.csv')
    return df

import os

Path = os.path.dirname( os.path.abspath(__file__))

def on_chain_merge(categorie, metric):
    
    if metric == 'stock_to_flow_ratio':
        try:
            df_merged = pd.read_csv(Path+'/../data/datos/merged_{categorie}_{metric}.csv', index_col= 'timestamp')
            return df_merged
        except:
            df_merged = glassnode('/v1/metrics/indicators/stock_to_flow_ratio')
            df_merged = df_merged.merge(pd.json_normalize(df_merged.stock_to_flow_ratio), left_index= True, right_index=True)
            del df_merged['stock_to_flow_ratio']
            del df_merged['price']
            time.sleep(1)
            df_merged.set_index('timestamp', drop=True, inplace = True)
            df_merged.columns = ['days','metric']
            df_btc = btc().copy()
            df_btc.index = pd.to_datetime(df_btc.index)
            df_merged = df_btc.merge(df_merged, left_index=True, right_index=True)
            try:
                df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'])
                df_merged.set_index('timestamp',drop=True, inplace = True)
            except:pass
            df_merged.to_csv(Path+f'/../data/datos/merged_{categorie}_{metric}.csv')
            return df_merged

    else:
        try:
            df_merged = pd.read_csv(Path+'/../data/datos/merged_{categorie}_{metric}.csv', index_col= 'timestamp')
            return df_merged

        except:
            df = on_chain_df(categorie, metric).copy()
            time.sleep(4)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            #time.sleep(2)

            df.set_index('timestamp', drop = True, inplace = True)
            df_btc = btc().copy()
            df_btc.index = pd.to_datetime(df_btc.index)
            df_merged = df_btc.merge(df, left_index=True, right_index=True)
            try:
                df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'])
                df_merged.set_index('timestamp',drop=True, inplace = True)
            except:pass
            df_merged.to_csv(Path+f'/../data/datos/merged_{categorie}_{metric}.csv')

            return df_merged


