import json
import requests
import pandas as pd

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
