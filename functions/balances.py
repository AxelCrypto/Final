import json
import cryptocompare as cc
import requests
import pandas as pd
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go



Path = os.path.dirname( os.path.abspath(__file__))

with open(Path+'/../api_cryptocompare.txt') as api:
    api = api.readlines()[0]

api_key = api
data_limit = 2000
symbol_a = 'BTC'

def api_call(url):
  # Set API Key as Header
  headers = {'authorization': 'Apikey ' + api_key,}
  session = requests.Session()
  session.headers.update(headers)

  # API call to cryptocompare
  response = session.get(url)

  # Conversion of the response to dataframe
  historic_blockdata_dict = json.loads(response.text)
  df = pd.DataFrame.from_dict(historic_blockdata_dict.get('Data').get('Data'), orient='columns', dtype=None, columns=None)
  return df

def prepare_pricedata(df):
  df['date'] = pd.to_datetime(df['time'], unit='s')
  df.drop(columns=['time', 'conversionType', 'conversionSymbol'], inplace=True)
  return df


# Prepare balance distribution dataframe
def prepare_balancedistribution_data(df):
  df['balance_distribution'] = df['balance_distribution'].apply(lambda x: [i for i in x])
  json_struct = json.loads(df[['time','balance_distribution']].to_json(orient="records"))    
  df_ = pd.json_normalize(json_struct)
  df_['date'] = pd.to_datetime(df_['time'], unit='s')
  df_flat = pd.concat([df_.explode('balance_distribution').drop(['balance_distribution'], axis=1),
           df_.explode('balance_distribution')['balance_distribution'].apply(pd.Series)], axis=1)
  df_flat.reset_index(drop=True, inplace=True)
  df_flat['range'] = ['' + str(float(df_flat['from'][x])) + '_to_' + str(float(df_flat['to'][x])) for x in range(df_flat.shape[0])]
  df_flat.drop(columns=['from','to', 'time'], inplace=True)

  # Data cleansing
  df_flat = df_flat[~df_flat['range'].isin(['100000.0_to_0.0'])]
  df_flat['range'].iloc[df_flat['range'] == '1e-08_to_0.001'] = '0.0_to_0.001'
  return df_flat


def balances_viz_addresses(df: pd.DataFrame, log_scale: bool, log_scale_metric: bool, ma : int, type_balance, indic: str = 'metric', z_score: bool = False):



    dic=  {'<0.01':'less_than_0_01', '0.01 - 0.1': 'f_0_01_to_0_1' ,'0.1 - 1': 'f_0_1_to_1', '1 - 10' : 'f_1_to_10', '10 - 100' : 'f_10_to_100','100 - 1000' : 'f_100_to_1000', '1k - 10k' : 'f_1000_to_10000','10k+' : 'f_10000_to_100000', 'all': 'all'}

    if type_balance== 'Adresses Count':
        ind = 'addressesCount_' + dic[indic]
    else: ind = 'totalVolume_' + dic[indic]

    fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}],[{"secondary_y": True}]], shared_xaxes=True, vertical_spacing=0.1, row_width=[0.1, 0.3])
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick chart'), row=1, col=1, secondary_y=False)
    

    rolling_mean = df[ind].rolling(ma).mean()
    mean = rolling_mean.mean()
    std = rolling_mean.std()
    z_scores = (df[ind] -  df[ind].rolling(ma).mean()) /  df[ind].rolling(ma).std()

    if z_score == True: 
        if log_scale:
            fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='log'), yaxis2=dict(title=ind, overlaying='y', side='right'))
            fig.add_trace(go.Bar(x=df.index, y=z_scores.rolling(ma+7).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=True)
        if log_scale == False:
            fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='linear'), yaxis2=dict(title=ind, overlaying='y', side='right'))
            fig.add_trace(go.Bar(x=df.index, y=z_scores.rolling(ma+7).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=True)

    else:
        if log_scale == True:
            if log_scale_metric == True :
                fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='log'), yaxis2=dict(title=ind, overlaying='y', side='right'))
                fig.add_trace(go.Line(x=df.index, y=df[ind].rolling(ma).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=True)
                fig.update_layout(yaxis2=dict(title=ind, type='log', overlaying='y', side='right'))

            elif log_scale_metric == False : 
                fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='log'), yaxis2=dict(title=ind, type='linear', overlaying='y', side='right'))
                fig.add_trace(go.Line(x=df.index, y=df[ind].rolling(ma).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=True)
        elif log_scale == False : 
            if log_scale_metric == True :
                fig.add_trace(go.Line(x=df.index, y=df[ind].rolling(ma).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=True)
                fig.update_layout(yaxis2=dict(title=ind, type='log', overlaying='y', side='right'))
            else:
                fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='linear'), yaxis2=dict(title=ind, type='linear', overlaying='y', side='right'))
                fig.add_trace(go.Line(x=df.index, y=df[ind].rolling(ma).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=True)
        
    
    title = indic.capitalize()
    fig.update_layout(title=title)
    fig.update_layout(width=600, height=800)
    fig['layout']['xaxis2']['title'] = 'Date'
    fig['layout']['yaxis']['title'] = 'Price'
    fig['layout']['yaxis2']['title'] = 'z-score (mean=1)'
    
    return fig 