
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