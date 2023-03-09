import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'
import numpy as np
import json
from datetime import datetime
from scipy import stats


def on_chain_viz(df: pd.DataFrame, log_scale: bool, log_scale_metric: bool, ma : int, categorie: str = 'onchain', indic: str = 'metric'):

    fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}],[{"secondary_y": True}]], shared_xaxes=True, vertical_spacing=0.1, row_width=[0.1, 0.3])
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick chart'), row=1, col=1, secondary_y=False)
    
    fig.update_xaxes(showticklabels=True, tickformat="%Y-%m-%d")
    
    fig.add_trace(go.Line(x=df.index, y=df['metric'].rolling(ma).mean(), name= categorie + '_' + indic, yaxis='y2'), row=1, col=1, secondary_y=True)

    title = categorie.capitalize() + ' ' + indic.capitalize()

    fig.update_layout(title=title)

    if log_scale == True:
        if log_scale_metric == True :
            fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='log'), yaxis2=dict(title='metric', overlaying='y', side='right'))
        elif log_scale_metric == False : 
            fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='log'), yaxis2=dict(title='metric', type='linear', overlaying='y', side='right'))
    elif log_scale == False : 
        if log_scale_metric == True :
            fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='linear'), yaxis2=dict(title='metric', type='log', overlaying='y', side='right'))
        else:
            fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='linear'), yaxis2=dict(title='metric', type='linear', overlaying='y', side='right'))
    
    fig.update_layout(width=600, height=800)
    fig['layout']['xaxis2']['title'] = 'Date'
    fig['layout']['yaxis']['title'] = 'Price'
    fig['layout']['yaxis2']['title'] = indic
    
    return fig



def on_chain_viz_zscore(df: pd.DataFrame, log_scale: bool, log_scale_metric: bool, ma : int, categorie: str = 'onchain', indic: str = 'metric', z_score: bool = True):


    
    fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}],[{"secondary_y": True}]], shared_xaxes=True, vertical_spacing=0.1, row_width=[0.1, 0.3])
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick chart'), row=1, col=1, secondary_y=False)
    
    fig.update_xaxes(showticklabels=True, tickformat="%Y-%m-%d")
    
    fig.add_trace(go.Line(x=df.index, y=df['metric'].rolling(ma).mean(), name= categorie + '_' + indic, yaxis='y2'), row=1, col=1, secondary_y=True)

    title = categorie.capitalize() + ' ' + indic.capitalize()

    fig.update_layout(title=title)

    if log_scale == True:
        if log_scale_metric == True :
            fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='log'), yaxis2=dict(title='metric', overlaying='y', side='right'))
        elif log_scale_metric == False : 
            fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='log'), yaxis2=dict(title='metric', type='linear', overlaying='y', side='right'))
    elif log_scale == False : 
        if log_scale_metric == True :
            fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='linear'), yaxis2=dict(title='metric', type='log', overlaying='y', side='right'))
        else:
            fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='linear'), yaxis2=dict(title='metric', type='linear', overlaying='y', side='right'))
    
    fig.update_layout(width=600, height=800)
    fig['layout']['xaxis2']['title'] = 'Date'
    fig['layout']['yaxis']['title'] = 'Price'
    fig['layout']['yaxis2']['title'] = indic
    
    return fig



def on_chain_viz_zscore_test(df: pd.DataFrame, log_scale: bool, log_scale_metric: bool, ma : int, categorie: str = 'onchain', indic: str = 'metric', z_score: bool = True):

    fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}],[{"secondary_y": True}]], shared_xaxes=True, vertical_spacing=0.1, row_width=[0.1, 0.3])
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick chart'), row=1, col=1, secondary_y=False)
    
    rolling_mean = df['metric'].rolling(ma).mean()
    rolling_std = df['metric'].rolling(ma).std()
    z_score = (df['metric'] - rolling_mean) / rolling_std
    z_score_1 = stats.zscore(z_score)
    
    fig.add_trace(go.Line(x=df.index, y=z_score_1, name= categorie + '_' + indic, yaxis='y2'), row=1, col=1, secondary_y=True)

    title = categorie.capitalize() + ' ' + indic.capitalize()

    fig.update_layout(title=title)

    if log_scale == True:
        if log_scale_metric == True :
            fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='log'), yaxis2=dict(title='metric', overlaying='y', side='right'))
        elif log_scale_metric == False : 
            fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='log'), yaxis2=dict(title='metric', type='linear', overlaying='y', side='right'))
    elif log_scale == False : 
        if log_scale_metric == True :
            fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='linear'), yaxis2=dict(title='metric', type='log', overlaying='y', side='right'))
        else:
            fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='linear'), yaxis2=dict(title='metric', type='linear', overlaying='y', side='right'))
    
    fig.update_layout(width=600, height=800)
    fig['layout']['xaxis2']['title'] = 'Date'
    fig['layout']['yaxis']['title'] = 'Price'
    fig['layout']['yaxis2']['title'] = 'z-score (mean=1)'
    
    return fig
