import time
from datetime import datetime
#import pylab as plt
try:
    import matplotlib.pyplot as plt
except: %pip install matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
from functions.get_btc import btc
from functions.get_M2 import *


def merged_btc_M2s():
    # importing Bitcoin's price DF 
    sys.path.insert(0, 'C:/Users/axell/Documents/BitcoinML/streamlit')
    df = btc()
    df_btc = pd.DataFrame(df)

    # importing M2_USD price 
    sys.path.insert(0, 'C:/Users/axell/Documents/BitcoinML/streamlit')
    df = M2_usd()
    df_M2_usd = pd.DataFrame(df)

    # importing M2_EUR price 
    df = M2_ecb()
    df_M2_eur = pd.DataFrame(df)

    #Merging all
    df = merging(df_M2_usd,df_M2_eur,df_btc)
    merged = pd.DataFrame(df)
    merged['M2_Fed_and_ECB'] = merged.m2_usd + merged.m2_eur

    return merged