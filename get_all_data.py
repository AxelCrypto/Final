import pandas as pd
import time
import requests
import json
import numpy as np
from datetime import datetime
import os
from functions.get_M2 import *
from functions.get_btc import *
import yfinance as yf
from functions.on_chain import *
from functions.balances import *
from git import Repo
import os


Path = os.path.dirname( os.path.abspath(__file__))


def get_all_data_btc ():
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
    df.to_csv(Path+'/data/datos/btc.csv')
    df_btc = df.copy()
    return df_btc

df_btc = get_all_data_btc().copy()

def get_all_data_M2():

    global df_btc
    # importing Bitcoin's price DF 
    df_btc.index = pd.to_datetime(df_btc.index)
    
    # importing M2_USD price 
    df = M2_usd()
    df.index = pd.to_datetime(df.index)
    df_M2_usd = df.copy()
    

    # importing M2_EUR price 
    df = M2_ecb()
    df_M2_eur = pd.DataFrame(df)

    #merging
    combined = df_M2_usd.merge(df_M2_eur, left_index = True, right_index = True) 
    
    combined.iloc[:,1] = combined.iloc[:,1]/1000
    combined.columns = ['m2_usd','m2_eur']

    combined = combined.merge(df_btc, left_index= True, right_index = True)
    combined.to_csv(Path+'/data/datos/merged_M2s.csv')

    return combined

def get_all_data_DXY():
    dxy = yf.download('DX-Y.NYB', start='2010-01-01', end=datetime.today())
    dxy = pd.DataFrame(dxy)
    #dxy.set_index('Date')
    #dxy.index = pd.to_datetime(df.index)
    dxy.to_csv('data/datos/dxy.csv')

def get_all_data_blockchain(categorie, indicateur):
    url = "https://api.blockchain.info/charts/hash-rate"
    params = {"timespan": "all", "format": "json"}
    response = requests.get(url, params=params)

    # Convert the JSON response to a DataFrame
    data = pd.DataFrame(response.json()["values"])

    # Convert the timestamp values to datetime objects
    data["x"] = pd.to_datetime(data["x"], unit="s")

    # Rename the columns to something more meaningful
    data = data.rename(columns={"x": "timestamp", "y": "metric"})
    data.set_index('timestamp', drop= True, inplace = True)

    df_btc.index = pd.to_datetime(df_btc.index)
    df = df_btc.merge(data, left_index= True, right_index= True)
    df.to_csv(f'data/datos/{categorie}_{indicateur}.csv')

def get_all_data_s2f(categorie, indicateur):
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
    df_merged.to_csv(f'data/datos/merged_{categorie}_{indicateur}.csv')
    return df_merged

def get_all_data_onchain(categorie, metric):
    global df_btc
    url =  '/v1/metrics/' + str(categorie)  +'/' + str(metric)
    df = glassnode(url).copy()
    
    time.sleep(1)
    df.set_index('timestamp',inplace = True, drop= True)
    df.columns = ['metric']
#    df.index = pd.to_datetime(df.index)
#    df.set_index('timestamp', drop = True, inplace = True)
    df_btc.index = pd.to_datetime(df_btc.index)
    df_merged = df_btc.merge(df, left_index=True, right_index=True)
    try:
        df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'])
        df_merged.set_index('timestamp',drop=True, inplace = True)
    except:pass
    df_merged.to_csv(f'data/datos/merged_{categorie}_{metric}.csv')

    return df_merged

def get_all_data_balances():
    base_url = 'https://min-api.cryptocompare.com/data/blockchain/balancedistribution/histo/day?fsym='
    df_raw = api_call(f'{base_url}btc&limit=2000')
    df_distr = prepare_balancedistribution_data(df_raw)
    f_0_0_to_0_001 = df_distr[::9] 
    f_0_001_to_0_01 = df_distr[1::9] 
    df_merged = f_0_0_to_0_001.merge(f_0_001_to_0_01, left_on='date', right_on ='date' )
    df_merged['totalVolume'] = df_merged.totalVolume_x + df_merged.totalVolume_y
    df_merged['addressesCount'] = df_merged.addressesCount_x + df_merged.addressesCount_y

    #final categories:
    less_than_0_01 = df_merged.drop(columns = df_merged.iloc[:,1:-2])
    f_0_01_to_0_1= df_distr[2::9].drop(columns='range')
    f_0_1_to_1 = df_distr[3::9].drop(columns='range')
    f_1_to_10 = df_distr[4::9] .drop(columns='range')
    f_10_to_100 = df_distr[5::9] .drop(columns='range')
    f_100_to_1000 = df_distr[6::9] .drop(columns='range')
    f_1000_to_10000 = df_distr[7::9] .drop(columns='range')
    f_10000_to_100000 = df_distr[8::9] .drop(columns='range')
    
    #Rename 
    less_than_0_01.columns= ['date','totalVolume_less_than_0_01','addressesCount_less_than_0_01']
    f_0_01_to_0_1.columns= ['date','totalVolume_f_0_01_to_0_1','addressesCount_f_0_01_to_0_1']
    f_0_1_to_1.columns= ['date','totalVolume_f_0_1_to_1','addressesCount_f_0_1_to_1']
    f_1_to_10.columns= ['date','totalVolume_f_1_to_10','addressesCount_f_1_to_10']
    f_10_to_100.columns= ['date','totalVolume_f_10_to_100','addressesCount_f_10_to_100']
    f_100_to_1000.columns= ['date','totalVolume_f_100_to_1000','addressesCount_f_100_to_1000']
    f_1000_to_10000.columns= ['date','totalVolume_f_1000_to_10000','addressesCount_f_1000_to_10000']
    f_10000_to_100000.columns= ['date','totalVolume_f_10000_to_100000','addressesCount_f_10000_to_100000']

    #merging
    df_btc.index = pd.to_datetime(df_btc.index)
    df = df_btc.merge(less_than_0_01,left_index= True, right_on='date').merge(f_0_01_to_0_1, left_on='date', right_on='date').merge(f_0_1_to_1, left_on='date', right_on='date').merge(f_0_1_to_1, left_on='date', right_on='date').merge(f_0_1_to_1, left_on='date', right_on='date').merge(f_1_to_10, left_on='date', right_on='date').merge(f_10_to_100, left_on='date', right_on='date').merge(f_100_to_1000, left_on='date', right_on='date').merge(f_1000_to_10000, left_on='date', right_on='date').merge(f_10000_to_100000, left_on='date', right_on='date')
    df.to_csv('data/datos/merged_balances.csv')

def run_all():
    get_all_data_btc()
    get_all_data_M2()
    get_all_data_DXY()
    get_all_data_blockchain('Mining', 'Hashrate')
    get_all_data_blockchain('Mining', 'Total Transaction Fees (BTC)')
    get_all_data_s2f('indicators', 'stock_to_flow_ratio')
    get_all_data_onchain('fees', 'volume_sum')
    get_all_data_balances()
    get_all_data_blockchain('On-Chain', 'unique-addresses')
    get_all_data_onchain('addresses', 'active_count')
    get_all_data_onchain('addresses', 'new_non_zero_count')
    get_all_data_onchain('indicators', 'sopr')
    get_all_data_onchain('institutions', 'purpose_etf_aum_sum')
    get_all_data_onchain('institutions', 'purpose_etf_flows_sum')
    get_all_data_onchain('institutions', 'purpose_etf_holdings_sum')
    get_all_data_onchain('market', 'marketcap_usd')
    get_all_data_onchain('market', 'marketcap_usd')
    get_all_data_onchain('supply', 'active_more_1y_percent')
    get_all_data_onchain('transactions', 'count')
    get_all_data_onchain('transactions', 'size_mean')

run_all()

PATH_OF_GIT_REPO = r'.git'  
COMMIT_MESSAGE = 'update_data'

def git_push():
#    try:
    repo = Repo(PATH_OF_GIT_REPO)
    repo.git.add(update=True)
    repo.index.commit(COMMIT_MESSAGE)
    origin = repo.remote(name='origin')
    origin.push(refspec='HEAD:developper')
#    except:
#        print('Some error occured while pushing the code')    


git_push()