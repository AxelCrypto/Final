
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'
import numpy as np
import json
from datetime import datetime
from scipy import stats



def macro_zscore(df: pd.DataFrame, log_scale: bool, log_scale_metric: bool,  z_score: bool , ma : int, indic: str = 'M2_sum'):

    fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}],[{"secondary_y": True}]], shared_xaxes=True, vertical_spacing=0.1, row_width=[0.1, 0.3])
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick chart'), row=1, col=1, secondary_y=False)

    rolling_mean = df['M2_sum'].rolling(ma).mean()
    mean = rolling_mean.mean()
    std = rolling_mean.std()
    z_scores = (df['M2_sum'] -  df['M2_sum'].rolling(ma).mean()) /  df['M2_sum'].rolling(ma).std()

    if z_score == True: 
        if log_scale:
            fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='log'), yaxis2=dict(title='metric', overlaying='y', side='right'))
            fig.add_trace(go.Bar(x=df.index, y=z_scores.rolling(ma+7).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=True)
        if log_scale == False:
            fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='linear'), yaxis2=dict(title='metric', overlaying='y', side='right'))
            fig.add_trace(go.Bar(x=df.index, y=z_scores.rolling(ma+7).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=True)

    else:
        if log_scale == True:
            if log_scale_metric == True :
                fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='log'), yaxis2=dict(title='metric', overlaying='y', side='right'))
                fig.add_trace(go.Line(x=df.index, y=df['M2_sum'].rolling(ma).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=True)
                fig.update_layout(yaxis2=dict(title='metric', type='log', overlaying='y', side='right'))

            elif log_scale_metric == False : 
                fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='log'), yaxis2=dict(title='metric', type='linear', overlaying='y', side='right'))
                fig.add_trace(go.Line(x=df.index, y=df['M2_sum'].rolling(ma).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=True)
        elif log_scale == False : 
            if log_scale_metric == True :
                fig.add_trace(go.Line(x=df.index, y=df['M2_sum'].rolling(ma).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=True)
                fig.update_layout(yaxis2=dict(title='metric', type='log', overlaying='y', side='right'))
            else:
                fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='linear'), yaxis2=dict(title='metric', type='linear', overlaying='y', side='right'))
                fig.add_trace(go.Line(x=df.index, y=df['M2_sum'].rolling(ma).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=True)
        
    
    title = indic.capitalize() 
    fig.update_layout(title=title)
    fig.update_layout(width=600, height=800)
    fig['layout']['xaxis2']['title'] = 'Date'
    fig['layout']['yaxis']['title'] = 'Price'
    fig['layout']['yaxis2']['title'] = indic
    
    return fig

