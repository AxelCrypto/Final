import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime
import os


Path = os.path.dirname( os.path.abspath(__file__))
def btc():
    try:
        df = pd.read_csv(Path+'/../data/datos/btc.csv', index_col= 'timestamp')
       # df.set_index('timestamp', drop = True, inplace=True)
       # df.index = pd.to_datetime(df.index)
        return df

    except:
        url = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
        timeframe = 86400
        start_date = 1312174800
        end_date = int(time.time())

        # set the maximum number of data points to return per request
        limit = 1000

        data_list = []
        while start_date <= end_date:

            request_end_date = min(start_date + limit*timeframe, end_date)
            response = requests.get(url + "?step=" + str(timeframe) + "&start=" + str(start_date) + "&end=" + str(request_end_date) + "&limit=" + str(limit))
            data = json.loads(response.text)
            data_list += data['data']['ohlc']
            start_date += limit*timeframe

        df = pd.DataFrame(data_list)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        df.set_index('timestamp',drop=True, inplace = True)
        df.columns = [e.capitalize() for e in df.columns]
        df.sort_index(ascending = True, inplace= True)
        df.to_csv(Path+'/../data/datos/btc.csv')

        return df
