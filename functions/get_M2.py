from full_fred.fred import Fred
import pandas as pd
import numpy as np
from datetime import datetime
import os

def M2_usd():
    Path = os.path.dirname( os.path.abspath(__file__))
    with open('functions/data/apis/API Fred.txt') as api:
        api = api.readlines()[0]
        
    fred = Fred('functions/data/apis/API Fred.txt')

    fred.get_api_key_file()

    fred.set_api_key_file('functions/data/apis/API Fred.txt')

    M2 = fred.get_series_df('WM2NS')

    del M2 ['realtime_start']
    #del M2 ['realtime_end']

    M2.set_index('date', inplace=True)

    M2.index = pd.to_datetime(M2.index)

    M2.sort_index(ascending=False, inplace=True)

    dates = pd.date_range(start="2009-03-08",end=datetime.today())
    dates = pd.to_datetime(dates)
    dates = dates.sort_values(ascending = False)
    dates = pd.to_datetime(dates)
    dates = dates.sort_values(ascending = False)
    # Reindex M2 with the new date range
    M2 = M2.reindex(dates)
    M2 = M2.ffill()
    M2.sort_index(ascending=False, inplace=True)
    M2[M2.realtime_end.isnull() == False].iloc[0,1]
    M2 = M2.value.fillna(M2[M2.realtime_end.isnull() == False].iloc[0,1])
    M2 = pd.DataFrame(M2)
    M2.value = M2.value.astype('float')
    M2.columns = ['total_usd']

    return M2

# only CSV, missing a scraping BS for latest data
def M2_ecb():
    ecb = pd.read_csv('functions/data/ECB_data.csv')
    ecb.columns= ['date','M2_EUR']
    ecb.set_index('date', inplace=True)
    ecb = ecb.iloc[4:,:]
    ecb.index = pd.to_datetime(ecb.index)

    ecb.M2_EUR = ecb.M2_EUR.astype('int')

    fill_dates = pd.date_range(start="2009-03-08",end=datetime.today())
    fill_dates = pd.to_datetime(fill_dates)
    fill_dates = fill_dates.sort_values(ascending = False)

    ecb = ecb.reindex(fill_dates)
    ecb = ecb.ffill()

    ecb.sort_index(ascending=False, inplace=True)

    ecb[ecb.M2_EUR.isnull() == False].iloc[0,0]

    ecb = ecb.M2_EUR.fillna(ecb[ecb.M2_EUR.isnull() == False].iloc[0,0])
    ecb = pd.DataFrame(ecb)
    ecb.M2_EUR = ecb.M2_EUR.astype('float')
    ecb.columns = ['total_eur']
    
    return ecb

def merging(df1,df2, btc):
    Path = os.path.dirname( os.path.abspath(__file__))

    try:
        df = pd.read_csv(Path+'/../data/datos/merged_M2s.csv', index_col= 'timestamp')
       # df.set_index('timestamp', drop = True, inplace=True)
       # df.index = pd.to_datetime(df.index)
        return df

    except:

        combined = df1.merge(df2, left_index = True, right_index = True) 
        combined.iloc[:,1] = combined.iloc[:,1]/1000
        combined.columns = ['m2_usd','m2_eur']
        combined = combined.merge(btc, left_index= True, right_index = True)
        combined.to_csv(Path+'/../data/datos/merged_M2s.csv')

        return combined