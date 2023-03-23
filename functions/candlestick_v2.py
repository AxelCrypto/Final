import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'
import numpy as np
import json
from datetime import datetime



def get_candlestick_plot(
        df: pd.DataFrame,
        log_scale: bool,
        ticker: str = 'btc'):

    fig = make_subplots(
        rows = 2,
        cols = 1,
        shared_xaxes = True,
        vertical_spacing = 0.1,
        subplot_titles = (f'{ticker} Stock Price', 'Volume Chart'),
        row_width = [0.1, 0.3]
    )
    
    fig.add_trace(
        go.Candlestick(
            x = df.index,
            open = df['Open'], 
            high = df['High'],
            low = df['Low'],
            close = df['Close'],
            name = 'Candlestick chart'
        ),
        row = 1,
        col = 1,
    )
    fig.update_xaxes(showticklabels=True, tickformat="%Y-%m-%d")
    

    fig.add_trace(
        go.Bar(x = df.index, y = df['Volume'], name = 'Volume'),
        row = 2,
        col = 1,
    )

    if log_scale == True:

        fig.update_layout(
            xaxis2=dict(title='Date'),
            yaxis=dict(title='Price', type='log'),
            yaxis2=dict(title='Volume'),
       )

    elif log_scale == False: 
        fig.update_layout(
            xaxis2=dict(title='Date'),
            yaxis=dict(title='Price', type='linear'),
            yaxis2=dict(title='Volume'),
    )

    fig.update_layout(
        width=600,
        height=800)

    
    fig['layout']['xaxis2']['title'] = 'Date'
    fig['layout']['yaxis']['title'] = 'Price'
    fig['layout']['yaxis2']['title'] = 'Volume'
    

    
    return fig



def get_candlestick_plot_ma(
        df: pd.DataFrame,
        log_scale: bool,
        ticker: str = 'btc'):
    
    df['20w_sma'] = df['Close'].rolling(140).mean()
    df['21w_ema'] = df['Close'].ewm(span=147, adjust=False).mean()

    fig = make_subplots(
        rows = 2,
        cols = 1,
        shared_xaxes = True,
        vertical_spacing = 0.1,
        subplot_titles = (f'{ticker} Stock Price', 'Volume Chart'),
        row_width = [0.1, 0.3]
    )
    
    fig.add_trace(
        go.Candlestick(
            x = df.index,
            open = df['Open'], 
            high = df['High'],
            low = df['Low'],
            close = df['Close'],
            name = 'Candlestick chart'
        ),
        row = 1,
        col = 1,
    )
    fig.update_xaxes(showticklabels=True, tickformat="%Y-%m-%d")
    
    fig.add_trace(
        go.Line(x=df.index, y=df['20w_sma'], name='20w SMA'),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Line(x=df.index, y=df['21w_ema'], name='21 EMA'),
        row=1,
        col=1,
    )


    fig.add_trace(
        go.Bar(x = df.index, y = df['Volume'], name = 'Volume'),
        row = 2,
        col = 1,
    )

    if log_scale == True:

        fig.update_layout(
            xaxis2=dict(title='Date'),
            yaxis=dict(title='Price', type='log'),
            yaxis2=dict(title='Volume'),)

    elif log_scale == False: 
        fig.update_layout(
            xaxis2=dict(title='Date'),
            yaxis=dict(title='Price', type='linear'),
            yaxis2=dict(title='Volume'),)

    fig.update_layout(
        width=600,
        height=800)

    
    fig['layout']['xaxis2']['title'] = 'Date'
    fig['layout']['yaxis']['title'] = 'Price'
    fig['layout']['yaxis2']['title'] = 'Volume'
    

    
    return fig


def get_candlestick_plot_EHMA(df: pd.DataFrame, log_scale: bool, ticker: str = 'btc'):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{ticker} Stock Price',  'EHMA Signal'),
        row_width=[3, 2, 3]
    )

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlestick chart'
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(showticklabels=True, tickformat="%Y-%m-%d")

    # Add EHMA indicator
    period = 20
    yukdus = True
    sqrt_period = np.sqrt(period)

    def borserman_ema(x, y):
        alpha = 2 / (y + 1)
        sum = np.array([alpha * x[0]])
        for i in range(1, len(x)):
            value = alpha * x[i] + (1 - alpha) * sum[i-1]
            sum = np.append(sum, value)
        return sum

    close_ema1 = borserman_ema(df['Close'], int(period / 2))
    close_ema2 = borserman_ema(df['Close'], period)
    ehma = borserman_ema(2 * close_ema1 - close_ema2, sqrt_period)

    # Create a copy of the EHMA trace as a stacked area chart
    ehma_series = pd.Series(ehma)
    ehma_signal = go.Scatter(
        x=df.index,
        y=np.where(ehma_series > ehma_series.shift(), ehma, np.nan),
        mode='markers',
        marker=dict(size=2, color='green'),
        name='EHMA Signal'
    )

    fig.add_trace(
        ehma_signal,
        row=1, col=1
    )

    # Add the EHMA signal as a green area
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=ehma,
            mode='lines',
            line=dict(width=0),
            fillcolor='green',
            fill='tonexty',
            name='EHMA Area'
        ),
        row=1, col=1
    )

    # Update the layout
    if log_scale:
        fig.update_layout(
            xaxis=dict(title='Date'),
            yaxis=dict(title='Price', type='log'),
            yaxis3=dict(title='EHMA Signal', showticklabels=False),
        )
    else:
        fig.update_layout(
            xaxis=dict(title='Date'),
            yaxis=dict(title='Price', type='linear'),
            yaxis3=dict(title='EHMA Signal', showticklabels=False),
        )

    fig.update_layout(
        width=900,
        height=1500,)
   
    fig['layout']['xaxis2']['title'] = 'Date'
    fig['layout']['yaxis']['title'] = 'Price'
    

    
    return fig




def e_plot_price_zscore(df):
    # Convert the date format of the index to datetime object
    df.index = pd.to_datetime(df.index)

    # Create a figure and two subplots, one for each y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add a trace for Bitcoin's price in log scale
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", yaxis='y1', mode='lines', line=dict(color='blue')), secondary_y=False)
    fig.update_yaxes(type='log', secondary_y=False)
    
    # Set the color scale for M2 Z-Score Deviation
    color_scale = [[0.0, 'red'], [0.5, 'green'], [1.0, 'green']]
    
    # Add a trace for M2 Z-Score Deviation as a bar chart with color metrics
    # Handle the case where the last column of df is non-numeric
    if pd.api.types.is_numeric_dtype(df.iloc[:, -1]):
        fig.add_trace(go.Bar(x=df.index, y=df.iloc[:, -1], name="M2 Z-Score Deviation", yaxis='y2', marker=dict(color=df.iloc[:, -1], coloraxis="coloraxis", colorscale=color_scale)), secondary_y=True)
        fig.update_yaxes(title_text="M2 Z-Score", secondary_y=True)
    else:
        fig.add_trace(go.Scatter(x=[], y=[], name="M2 Z-Score Deviation"), secondary_y=True)
    
    # Set the title and axis labels
    fig.update_layout(title='Bitcoin vs M2 eur et usd (Z-Score)')
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price", secondary_y=False)

    # Show the chart in a browser link
    fig_json = pio.to_json(fig)
    return fig_json

from scipy import stats

def z_score(df, string,  av = 14):

    rolling_mean = df[::av][string].rolling(window=av, min_periods=1).mean()

    zscore_deviation = stats.zscore(df[::av][string].sub(rolling_mean, fill_value=0))

    df[f'M2_zscore_{av}'] = zscore_deviation

    data = df[::av][['Close',f'M2_zscore_{av}']]

    fig_json = e_plot_price_zscore(data)
    
    return json.loads(fig_json)




def viz_with_indicator(df: pd.DataFrame, log_scale: bool, log_scale_metric: bool, ma : int, indic: str = 'metric', z_score: bool = False):

    fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}],[{"secondary_y": True}]], shared_xaxes=True, vertical_spacing=0.1, row_width=[0.1, 0.3])
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick chart'), row=1, col=1, secondary_y=False)
    

    rolling_mean = df['metric'].rolling(ma).mean()
    mean = rolling_mean.mean()
    std = rolling_mean.std()
    z_scores = (df['metric'] -  df['metric'].rolling(ma).mean()) /  df['metric'].rolling(ma).std()

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
                fig.add_trace(go.Line(x=df.index, y=df['metric'].rolling(ma).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=True)
                fig.update_layout(yaxis2=dict(title='metric', type='log', overlaying='y', side='right'))

            elif log_scale_metric == False : 
                fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='log'), yaxis2=dict(title='metric', type='linear', overlaying='y', side='right'))
                fig.add_trace(go.Line(x=df.index, y=df['metric'].rolling(ma).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=True)
        elif log_scale == False : 
            if log_scale_metric == True :
                fig.add_trace(go.Line(x=df.index, y=df['metric'].rolling(ma).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=True)
                fig.update_layout(yaxis2=dict(title='metric', type='log', overlaying='y', side='right'))
            else:
                fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='linear'), yaxis2=dict(title='metric', type='linear', overlaying='y', side='right'))
                fig.add_trace(go.Line(x=df.index, y=df['metric'].rolling(ma).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=True)
        
    
    title = indic.capitalize()
    fig.update_layout(title=title)
    fig.update_layout(width=600, height=800)
    fig['layout']['xaxis2']['title'] = 'Date'
    fig['layout']['yaxis']['title'] = 'Price'
    fig['layout']['yaxis2']['title'] = 'z-score (mean=1)'
    
    return fig



def viz_with_indicator_same_axis(df: pd.DataFrame, log_scale: bool, log_scale_metric: bool, ma : int, indic: str = 'metric', z_score: bool = False):

    fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}],[{"secondary_y": True}]], shared_xaxes=True, vertical_spacing=0.1, row_width=[0.1, 0.3])
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick chart'), row=1, col=1, secondary_y=False)
    

    rolling_mean = df['metric'].rolling(ma).mean()
    mean = rolling_mean.mean()
    std = rolling_mean.std()
    z_scores = (df['metric'] -  df['metric'].rolling(ma).mean()) /  df['metric'].rolling(ma).std()

    if z_score == True: 
        if log_scale:
            fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='log'), yaxis2=dict(title='metric', overlaying='y', side='right'))
            fig.add_trace(go.Bar(x=df.index, y=z_scores.rolling(ma+7).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=False)
        if log_scale == False:
            fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='linear'), yaxis2=dict(title='metric', overlaying='y', side='right'))
            fig.add_trace(go.Bar(x=df.index, y=z_scores.rolling(ma+7).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=False)

    else:
        if log_scale == True:
            if log_scale_metric == True :
                fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='log'), yaxis2=dict(title='metric', overlaying='y', side='right'))
                fig.add_trace(go.Line(x=df.index, y=df['metric'].rolling(ma).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=False)
                fig.update_layout(yaxis2=dict(title='metric', type='log', overlaying='y', side='right'))

            elif log_scale_metric == False : 
                fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='log'), yaxis2=dict(title='metric', type='linear', overlaying='y', side='right'))
                fig.add_trace(go.Line(x=df.index, y=df['metric'].rolling(ma).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=False)
        elif log_scale == False : 
            if log_scale_metric == True :
                fig.add_trace(go.Line(x=df.index, y=df['metric'].rolling(ma).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=False)
                fig.update_layout(yaxis2=dict(title='metric', type='log', overlaying='y', side='right'))
            else:
                fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Price', type='linear'), yaxis2=dict(title='metric', type='linear', overlaying='y', side='right'))
                fig.add_trace(go.Line(x=df.index, y=df['metric'].rolling(ma).mean(), name= indic, yaxis='y2'), row=1, col=1, secondary_y=False)
        
    
    title = indic.capitalize()
    fig.update_layout(title=title)
    fig.update_layout(width=600, height=800)
    fig['layout']['xaxis2']['title'] = 'Date'
    fig['layout']['yaxis']['title'] = 'Price'
    fig['layout']['yaxis2']['title'] = 'z-score (mean=1)'
    
    return fig



