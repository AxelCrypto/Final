import json
import requests
import pandas as pd


# insert your API key here
API_KEY = '2Mgz6X8H5b7DDHS2jEic8aCASDL'

# make API request
res = requests.get('https://api.glassnode.com/v2/metrics/endpoints',
    params={'a': 'BTC', 'api_key': API_KEY})

# convert to pandas dataframe
df = pd.read_json(res.text, convert_dates=['t'])

print(df.head())

