
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.renderers.default='browser'
import numpy as np
import json
from datetime import datetime
from scipy import stats


def z_score(df, string = 'metric',  ma):

    zscore_deviation = stats.zscore(df[string])

    df['metric'] = zscore_deviation

    fig_json = e_plot_price_zscore(data)
    
    return json.loads(fig_json)


from scipy import stats

def z_score(df, string,  av = 14):

    rolling_mean = df[::av][string].rolling(window=av, min_periods=1).mean()

    zscore_deviation = stats.zscore(df[::av][string].sub(rolling_mean, fill_value=0))

    df[f'M2_zscore_{av}'] = zscore_deviation

    data = df[::av][['Close',f'M2_zscore_{av}']]

    fig_json = e_plot_price_zscore(data)
    
    return json.loads(fig_json)