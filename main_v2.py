import sys
import pandas as pd
import streamlit as st
from functions.candlestick_v2 import *
from functions.get_btc import btc
import numpy as np
from get_merge_btc_M2s import *
from functions.patterns import *
from functions.on_chain import *
from functions.on_chain_viz import *
from functions.Macro import *
from functions.balances import *
import yfinance as yf
from datetime import datetime
import requests
import cryptocompare as cc
from PIL import Image
from fbprophet import Prophet
import matplotlib.pyplot as plt


df_btc = btc()

st.set_page_config(
    page_title="Bitcoin's inside",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://twitter.com/AxelCryptoytb',
        'Report a bug': "https://twitter.com/AxelCryptoytb",
        'About': "# Bitcoin vu de l'intérieur. Données extraites de mon propre noeud bitcoin. Toutes ces données sont purement informatives et rien n'est à prendre en compte comme un conseil en investissement. *@AxelCrypto*"
    }
)

#Options download
ind = pd.read_csv('data/free_indicators_glassnode.csv')
dic = ind[['category','metric']]
categories = list(ind.category)
categories.append('balances')


## Sidebar options

# titre sidebar
st.sidebar.header('Dashboard')

#st.sidebar.subheader('Données')
categorie = st.sidebar.selectbox("**catégorie**", ('Technique', 'Macro', 'Mining', 'On-Chain'))

if categorie == 'Technique':

    with st.sidebar.form("Indicateurs"):
        indicateur = st.selectbox('Indicateurs techniques ', ('Prix', 'Bull-Market Support Bands', 'EHMA', 'Mayer Multiple', 'Puell Multiple'))

        days_to_plot = st.slider(
            'Nombre de jours',
            min_value=1,
            max_value=len(df_btc),
            value=len(df_btc)
        )

        checkbox_val = st.checkbox("Logarithmic")

        # Every form must have a submit button.
        submitted = st.form_submit_button("**Envoyer**")
 
elif categorie == 'Macro':
    with st.sidebar.form("Macro"):
        indicateur = st.selectbox('Indicateurs macro-économiques', ('Masse Monétaire', 'DXY'))
        days_to_plot = st.slider(
            'Nombre de jours',
            min_value=1,
            max_value=len(df_btc),
            value=len(df_btc)
            )

        checkbox_val = st.checkbox("Logarithmic")
        checkbox_val_metric = st.checkbox("Indicateur Logarithmic")
        ma = st.slider("Moyenne de l'indicateur", min_value=1, max_value=90, value=1)

        #checkbox_zscore = st.checkbox("Activer le Z-Score")

        # Every form must have a submit button.
        submitted = st.form_submit_button("**Envoyer**")

elif categorie == 'Mining':
    with st.sidebar.form("Mining"):
        indicateur = st.selectbox('Mining Indicators', ('Hashrate', 'Total Transaction Fees (BTC)','volume_sum'))
        days_to_plot = st.slider(
            'Nombre de jours',
            min_value=1,
            max_value=len(df_btc),
            value=len(df_btc)
            )

        checkbox_val = st.checkbox("Logarithmic")
        checkbox_val_metric = st.checkbox("Indicateur Logarithmic")
        ma = st.slider("Moyenne de l'indicateur", min_value=1, max_value=90, value=1)

        checkbox_zscore = st.checkbox("Activer le Z-Score")

        submitted = st.form_submit_button("**Envoyer**")

else:
    onchain = st.sidebar.selectbox('**Type**', (sorted(set(categories))))
    with st.sidebar.form("On-Chain"):
        if onchain == 'addresses':
            metrics = st.selectbox("**metrics**", ('active_count','new_non_zero_count','unique-addresses'))
        elif onchain == 'balances':
            type_balance = st.selectbox("**Type of Balances**", ('Adresses Count','Balances'))
            metrics = st.selectbox("**metrics**", ('<0.01','0.01 - 0.1','0.1 - 1', '1 - 10', '10 - 100','100 - 1000', '1k - 10k','10k+'))
        elif onchain == 'transactions':
            metrics = st.selectbox("**metrics**", ('count','size_mean','count'))
        elif onchain == 'indicators':
            metrics = st.selectbox("**metrics**", ('sopr','stock_to_flow_ratio','pi_cycle_top'))
        elif onchain == 'supply':
            metrics = 'active_more_1y_percent'
        elif onchain == 'market':
            metrics = st.selectbox("**metrics**", ('price_drawdown_relative','marketcap_usd'))
        elif onchain == 'institutions':
            metrics = st.selectbox("**metrics**", ('purpose_etf_holdings_sum','purpose_etf_flows_sum','purpose_etf_aum_sum'))

        days_to_plot = st.slider(
        'Nombre de jours',
        min_value=1,
        max_value=len(df_btc),
        value=len(df_btc)
        )

        checkbox_val = st.checkbox("Logarithmic")
        checkbox_val_metric = st.checkbox("Indicateur Logarithmic")
        ma = st.slider("Moyenne de l'indicateur", min_value=1, max_value=90, value=1)

        checkbox_zscore = st.checkbox("Activer le Z-Score")

        # Every form must have a submit button.
        submitted = st.form_submit_button("**Envoyer**")

st.sidebar.markdown('''
---
GM Satoshi 🕵️  
NFA, by [Axel](https://www.youtube.com/c/AxelGirouGarcia).
''')



# Charts selon la selection:

if categorie == 'Technique':
    if indicateur == 'Prix': 
        st.header('Bitcoin `Prix actuel`')
#        st.markdown('### Metrics')
#        col1, col2, col3 = st.columns(3)
        st.write('Current pricing canals :chart_with_upwards_trend: :chart_with_downwards_trend:')

        
#        col2.metric("Wind", "9 mph", "-8%")
#        col3.metric("Humidity", "86%", "4%")


        
           
        c1, c2 = st.columns(2)
        with c1:
            df = df_btc.copy()
            df.reset_index(drop=False, inplace=True)
            df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
            df.Date = pd.to_datetime(df.Date)


            one_year = 300

            # Filter the dates to plot only the region of interest
            df = df[(df['Date'] > max(df['Date']) - one_year * pd.offsets.Day())]
            df = df.reset_index(drop=True)

            frame = 45

            # Get another df of the dates where we draw the support/resistance lines
            df_trend = df[(df.loc[:, 'Date'] > max(df['Date']) - frame * pd.offsets.Day()) & (df.loc[:, 'Date'] < max(df.loc[:, 'Date']))]

            # Apply the smoothing algorithm and get the gradient/intercept terms
            m_res, c_res = find_grad_intercept(case='resistance', 
                x=df_trend.index.values, 
                y=heat_eqn_smooth(df_trend['High'].values.copy()),
            )
            m_supp, c_supp = find_grad_intercept(
                case='support', 
                x=df_trend.index.values, 
                y=heat_eqn_smooth(df_trend['Low'].values.copy()),
            )

            # Get the plotly figure
            layout = go.Layout(
                title=f'Canal de prix des {frame} derniers jours',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Price'},
                legend={'x': 0, 'y': 1.075, 'orientation': 'h'},
                width=600,
                height=500,
            ) 

            fig = go.Figure(
                layout=layout,
                data=[
                    go.Candlestick(
                        x=df['Date'],
                        open=df['Open'], 
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        showlegend=False,
                    ),
                    go.Line(
                        x=df_trend['Date'], 
                        y=m_res*df_trend.index + c_res, 
                        showlegend=False, 
                        line={'color': 'rgba(89, 105, 208, 1)'}, 
                        mode='lines',
                    ),
                    go.Line(
                        x=df_trend['Date'], 
                        y=m_supp*df_trend.index + c_supp, 
                        showlegend=False, 
                        line={'color': 'rgba(89, 105, 208, 1)'}, 
                        mode='lines',
                    ),
                ]
            )

            st.plotly_chart(fig, use_container_width=False)

        with c2:
            #df.reset_index(drop=True, inplace=True)
            #df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

            one_year = 300

            # Filter the dates to plot only the region of interest
            df = df[(df['Date'] > max(df['Date']) - one_year * pd.offsets.Day())]
            df = df.reset_index(drop=True)

            frame = 250

            # Get another df of the dates where we draw the support/resistance lines
            df_trend = df[(df.loc[:, 'Date'] > max(df['Date']) - frame * pd.offsets.Day()) & (df.loc[:, 'Date'] < max(df.loc[:, 'Date']))]

            # Apply the smoothing algorithm and get the gradient/intercept terms
            m_res, c_res = find_grad_intercept(case = 'resistance', 
                x = df_trend.index.values, 
                y = heat_eqn_smooth(df_trend['High'].values.copy()),
            )
            m_supp, c_supp = find_grad_intercept(
                case = 'support', 
                x = df_trend.index.values, 
                y = heat_eqn_smooth(df_trend['Low'].values.copy()),
            )
            
            # Get the plotly figure
            layout = go.Layout(
                title = f'Canal de prix des {frame} derniers jours',
                xaxis = {'title': 'Date'},
                yaxis = {'title': 'Price'},
                legend = {'x': 0, 'y': 1.075, 'orientation': 'h'},
                width = 600,
                height = 500,
            ) 

            fig = go.Figure(
                layout=layout,
                data=[
                    go.Candlestick(
                        x = df['Date'],
                        open = df['Open'], 
                        high = df['High'],
                        low = df['Low'],
                        close = df['Close'],
                        showlegend = False,
                    ),
                    go.Line(
                        x = df_trend['Date'], 
                        y = m_res*df_trend.index + c_res, 
                        showlegend = False, 
                        line = {'color': 'rgba(89, 105, 208, 1)'}, 
                        mode = 'lines',
                    ),
                    go.Line(
                        x = df_trend['Date'], 
                        y = m_supp*df_trend.index + c_supp, 
                        showlegend = False, 
                        line = {'color': 'rgba(89, 105, 208, 1)'}, 
                        mode = 'lines',
                    ),
                ]
            )
            

            st.plotly_chart(fig ,use_container_width=False)


        st.metric("Last  price", f'${df_btc.iloc[-1,0]}', f'{round((df_btc.iloc[-1,0]/df_btc.iloc[-2,0]-1 ) *100,2)}%')
        df = df_btc.copy()
        df = df[-days_to_plot:]
        

        tab1, tab2= st.tabs(["Chart", "Prediction"])

        with tab1:

            st.plotly_chart(get_candlestick_plot(df, checkbox_val, 'btc'),
                use_container_width=True)    
           
        with tab2:
           # try:
            #    forecast = pd.read_csv(f'data/prophet/prophet{indicateur}.csv')

            #except:
            df_fb = df.reset_index(drop= False)
            df_fb = df_fb[['timestamp', 'Close']]
            df_fb.columns=['ds', 'y']
            model=Prophet().fit(df_fb)
            future=model.make_future_dataframe(periods=720, freq='D')
            forecast=model.predict(future)
            forecast=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][forecast.ds> datetime.today() ]

            forecast.columns=['time', 'close', 'lower', 'upper']
            st.table(forecast.head())
            #forecast.to_csv(f'data/prophet/prophet{indicateur}.csv')
            forecast.set_index('time', inplace = True)
            fig = plt(forecast)
            #fig=model.plot(forecast)
            st.pyplot(fig)


            




    elif indicateur == 'Bull-Market Support Bands': 
        st.header('Bitcoin `Bull-Market Support Bands`')


        df = df_btc.copy()
        df['20w_sma'] = df['Close'].rolling(140).mean()
        df['21w_ema'] = df['Close'].ewm(span=21, adjust=False).mean()
        df = df[-days_to_plot:]


        col1, col2, col3 = st.columns(3)
        col1.metric("Last  price", f'${df_btc.iloc[-1,0]}', f'{round((df_btc.iloc[-1,0]/df_btc.iloc[-2,0]-1 ) *100,2)}%')
        col2.metric("Last  price 21w EMA", round(df['21w_ema'][-1],0), round((df['21w_ema'][-1]/df['21w_ema'][-2] - 1)*100,2))
        col3.metric("Last  price 20w SMA", round(df['20w_sma'][-1],0), round((df['20w_sma'][-1]/df['20w_sma'][-2] - 1)*100,2))
        st.write('Bull-Market Support bands are a good support during :green[bullmarkets] and a strong resistence during :red[bearmarkets]')


        if st.button('**Evaluate the current situation**'):

            if df_btc.iloc[-1,0] > min(df['21w_ema'][-1],df['20w_sma'][-1],0):
                st.subheader('We are currently in a :green[bullmarket] :rocket:')

            else :
                st.subheader('We are currently in a :red[bearmarket] :bear:')



        # Determine the y-axis type
        

        tab1, tab2= st.tabs(["Chart", "Prediction"])

        with tab1:
            st.plotly_chart(get_candlestick_plot_ma(df, checkbox_val, 'btc' ),
                use_container_width=True)   
            
        with tab2:
            st.header("ML is comming")   

    elif indicateur == 'EHMA': 
        st.header('Bitcoin `EHMA`')

        df = df_btc.copy()
        df = df[-days_to_plot:]

        # Add EHMA indicator
        period = 180
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
        
        #col1, col2, col3 = st.columns(3)
        #st.metric("Last  price", f'${df_btc.iloc[-1,0]}', f'{round((df_btc.iloc[-1,0]/df_btc.iloc[-2,0]-1 ) *100,2)}%')
        #col2.metric("Last  price 21w EMA", round(df['21w_ema'][-1],0), round((df['21w_ema'][-1]/df['21w_ema'][-2] - 1)*100,2))
        #col3.metric("Last  price 20w SMA", round(df['20w_sma'][-1],0), round((df['20w_sma'][-1]/df['20w_sma'][-2] - 1)*100,2))
        st.write('**Traditional moving averages lag the price activity. But with some clever mathematics the lag can be minimised. If green we are in a :green[bullmarkets], if red we are in a :red[bearmarkets]**')


        if st.button('**Evaluate the current situation**'):

            if ehma[-1] > ehma[-2]:
                st.subheader('We are currently in a :green[bullmarket] :rocket:')

            else :
                st.subheader('We are currently in a :red[bearmarket] :bear:')

        tab1, tab2= st.tabs(["Chart", "Prediction"])

        with tab1:
            # Determine the y-axis type
            if checkbox_val == True:
                st.plotly_chart(
                get_candlestick_plot_EHMA(df, True, 'btc' ),
                use_container_width=True)   
                
            else : 
                st.plotly_chart(get_candlestick_plot_EHMA(df, False, 'btc'),
                    use_container_width=True)     
        
        with tab2:
             st.header('Machine Learning is Coming')

    elif indicateur == 'Mayer Multiple':
        st.header('Bitcoin `Mayer Multiple`')
        df = df_btc.copy()
        df['200d_ma'] = df['Close'].rolling(200).mean()
        df['metric'] = df['Close'] / df['200d_ma']
        df = df[-days_to_plot:]


        col1, col2 = st.columns(2)
        col1.metric("Last  price Bitcoin", round(df['Close'][-1],0), round((df['Close'][-1]/df['Close'][-2] - 1)*100,2))
        col2.metric("Last  Mayer Multiple", round(df['metric'][-1],2), round((df['metric'][-1]/df['metric'][-2] - 1)*100,2))
        st.write('The Mayer Multiple is calculated by dividing the Bitcoin Price by the 200 day moving average of the price.')


        if st.button('**Evaluate the current situation**'):

            if df['metric'][-1] > 2:
                st.subheader('Probably a :red[risk] :small_red_triangle_down:')
            
            elif df['metric'][-1] > 1:
                st.subheader('Neutral risk :bar_chart:')

            else :
                st.subheader('Probably an :green[opportunity] :crossed_flags:')

        tab1, tab2= st.tabs(["Chart", "Prediction"])

        with tab1:
            st.plotly_chart(viz_with_indicator(df, checkbox_val, True, 1, indicateur, False),
                    use_container_width=True)     
            
        with tab2:
            st.header("ML is comming")   


    elif indicateur == 'Puell Multiple':
        st.header('Bitcoin `Puell Multiple`')
        df = df_btc.copy()
        df['365d_ma'] = df['Close'].rolling(365).mean()
        df['metric'] = df['Close'] / df['365d_ma']
        df = df[-days_to_plot:]


        col1, col2 = st.columns(2)
        col1.metric("Last  price Bitcoin", round(df['Close'][-1],0), round((df['Close'][-1]/df['Close'][-2] - 1)*100,2))
        col2.metric("Last  Puell Multiple", round(df['metric'][-1],2), round((df['metric'][-1]/df['metric'][-2] - 1)*100,2))
        st.write('The Puell Multiple is calculated by dividing the Bitcoin Price by the 365 day moving average of the price.')


        if st.button('**Evaluate the current situation**'):

            if df['metric'][-1] > 2:
                st.subheader('Probably a :red[risk] :small_red_triangle_down:')
            
            elif df['metric'][-1] > 1:
                st.subheader('Neutral risk :bar_chart:')

            else :
                st.subheader('Probably an :green[opportunity] :crossed_flags:')

        tab1, tab2= st.tabs(["Chart", "Prediction"])

        with tab1:
            st.plotly_chart(viz_with_indicator(df, checkbox_val, True, 1, indicateur, False),
                    use_container_width=True)     
            
        with tab2:
            st.header("ML is comming")   



elif categorie == 'Macro':
    if indicateur == 'Masse Monétaire': 

        st.header('Bitcoin vs `money printing (ECB+FED)` :money_with_wings:')

        st.write('Traditional currency can be "printed" without limits by central banks.\n\n'
           'The `higher` M2 goes, the `higher` bitcoin price could go. When central banks end easing and start to hike rates, bitcoin price could :red[drop].')

        df_usd = pd.DataFrame(M2_usd())
        df_eur = pd.DataFrame(M2_ecb())
        df_btc.index = pd.to_datetime(df_btc.index)
        df = merging(df_usd,df_eur, df_btc)
        df ['M2_sum'] = df.m2_usd + df.m2_eur

        days = st.number_input(
            'Number of days to compare',
            min_value=31,
            max_value=3650,
            value=31,
            step=1
        )

        if st.button('**Evaluate the current situation**'):

            if df ['M2_sum'][-1] > df ['M2_sum'][-31]:
                st.subheader('We are currently  in a :green[money printing market] :rocket:')

            else :
                st.subheader('We are currently  in a :red[tightning] :bear:')



        st.plotly_chart(macro_zscore(df, checkbox_val, checkbox_val_metric, ma, 
                                    #checkbox_zscore, 
                                    indicateur ),
                        use_container_width=True)   

    elif indicateur == 'DXY': 

        st.header('Bitcoin vs `USD Value Index`')  

        try:
            dxy = pd.read_csv('data/datos/dxy.csv', index_col= 'Date')

        except: 
            dxy = yf.download('DX-Y.NYB', start='2010-01-01', end=datetime.today())
            dxy = pd.DataFrame(dxy)
            #dxy.set_index('Date')
            #dxy.index = pd.to_datetime(df.index)
            dxy.to_csv('data/datos/dxy.csv')

        st.write('The dollar index tracks the relative value of the U.S. dollar against a basket of important world currencies.\n'
                'If the index is :red[rising], it means that the dollar is strengthening against the basket - and vice-versa.\n\n'
                'USD tends to appreciate in a deleveraging and derisking period. :bear:\n\n'
                'USD tends to depreciate in a risk-taking period with low interest rates and quantitative easing. :money_with_wings:')


        days = st.number_input(
            'Number of days to compare',
            min_value=1,
            max_value=3650,
            value=30,
            step=1
        )

        if st.button('**Evaluate the current situation**'):

            


            if dxy ['Close'][-1] < dxy ['Close'][-days-1:].mean():
                st.subheader(f'DXY is currently :green[lower] than the **{days}** average last days. :rocket:')

            else :
                st.subheader(f'DXY is currently :red[higher] than the **{days}** average last days. :bear:')

        st.plotly_chart(macro_dxy(df_btc,dxy,checkbox_val, checkbox_val_metric, ma ),
                    use_container_width=True)   

elif categorie == 'Mining':
    if indicateur == 'Hashrate': 

        try : 
            df = pd.read_csv(f'data/datos/{categorie}_{indicateur}.csv', index_col='timestamp')

        except:
            # Define the API endpoint and parameters
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
        
        st.header(f'You are looking at `{indicateur}` from the category `{categorie}`')
        df = df[-days_to_plot:]
        

        st.write('The hashrate is the computing power miners use to secure the Bitcoin network. \n\n'
                    "When the hashrate :red[drops], it means that miners are closing, maybe selling assets (like bitocin) and the bitcoin's netowrk is less secure. :bear: \n\n"
                    'When the hashrate :green[rises], it :  \n\n'
                    '- increased difficulty which makes it more costly to mine Bitcoin.  \n\n'
                    '- may indicate an increased demand for Bitcoin  \n\n'
                    "- makes the Bitcoin's network more secure :mechanical_arm:" 
        )

        days = st.number_input(
            'Number of days to compare',
            min_value=1,
            max_value=3650,
            value=30,
            step=1
        )

        if st.button('**Evaluate the current situation**'):

            


            if df ['metric'][-1] > df ['metric'][-days-1:].mean():
                st.subheader(f"Hashrate is currently :green[higher] than the **{days}** last days' average. :rocket:")

            else :
                st.subheader(f"Hashrate is currently :red[lower] than the **{days}** last days' average. :bear:")


        st.plotly_chart(viz_with_indicator(df, checkbox_val, checkbox_val_metric, ma, indicateur,checkbox_zscore ),
                            use_container_width=True)   


        url = 'https://www.youtube.com/watch?v=sWrNNh47p3Y'
        st.header('Vidéo explicative de la relation entre Prix du Bitcoin et Hashrate (FR)')
        st.video(url)


    elif indicateur == 'Total Transaction Fees (BTC)': 

        try : 
            df = pd.read_csv(f'data/datos/{categorie}_{indicateur}.csv', index_col='timestamp')

        except:
            # Define the API endpoint and parameters
            url = "https://api.blockchain.info/charts/transaction-fees"
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
        
        st.header(f'You are looking at `{indicateur}` from the category `{categorie}`')
        df = df[-days_to_plot:]
    
        st.write('The total BTC value of all transaction fees paid to miners. \n\n'
                    "Decrease = :red[less] demand :man_with_probing_cane: \n\n"
                    'Increase = :green[higher] demand :man-woman-girl-girl: \n\n'
        )

        days = st.number_input(
            'Number of days to compare',
            min_value=1,
            max_value=3650,
            value=30,
            step=1
        )

        if st.button('**Evaluate the current situation**'):        

            if df ['metric'][-1] > df ['metric'][-days-1:].mean():
                st.subheader(f"{indicateur} is currently :green[higher] than the **{days}** last days' average. :rocket:")

            else :
                st.subheader(f"{indicateur} is currently :red[lower] than the **{days}** last days' average. :bear:")



        st.plotly_chart(viz_with_indicator(df, checkbox_val, checkbox_val_metric, ma, indicateur,checkbox_zscore ),
                        use_container_width=True)   


    elif indicateur == 'volume_sum': 
        df = on_chain_merge('fees', indicateur)
        st.header(f'You are looking at `{indicateur}` from the category `mining`')
        df = df[-days_to_plot:]
        st.write('The total amount of fees paid to miners. Inflation rewards not included. Increasing = :green[Higher Demand] decreasing = :red[Lower Demand]')


        days = st.number_input(
            'Number of days to compare',
            min_value=1,
            max_value=3650,
            value=30,
            step=1
        )

        if st.button('**Evaluate the current situation**'):        

            if df ['metric'][-1] > df ['metric'][-days-1:].mean():
                st.subheader(f"{indicateur} is currently :green[higher] than the **{days}** last days' average. :rocket:")

            else :
                st.subheader(f"{indicateur} is currently :red[lower] than the **{days}** last days' average. :bear:")

        st.plotly_chart(on_chain_viz_zscore(df, checkbox_val, checkbox_val_metric, ma, 'fees', indicateur, checkbox_zscore ),
                        use_container_width=True)   





elif categorie == 'On-Chain':
    
    if onchain == 'balances':
        try: df = pd.read_csv('data/datos/merged_balances.csv', index_col= 'date')

        except:
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


        df = df[-days_to_plot:]




        if type_balance == 'Adresses Count':
            st.header(f'You are looking at `{type_balance}` from the category `{onchain}`')
            st.write('You are looking at the number of adresses per cohort. \n\n'
                    'It is relevant to observe which actors are entering or leaving the market.'
                    "Are there the :blue[whales] which are often instutional actors which can have an impact on the Bitcoin's price. :fish: \n\n"
                    'Or, are there smaller actors which can be more emotional but who decentralise the network. :fishing_pole_and_fish:'
            )


            st.plotly_chart(balances_viz_addresses(df, checkbox_val, checkbox_val_metric, ma,type_balance, metrics, checkbox_zscore),
                use_container_width=True)     
            
        elif type_balance == 'Balances':
            st.header(f'You are looking at `{type_balance}` from the category `{onchain}`')
            st.write('You are looking at the total Bitcoin holdings per cohort. \n\n'
                    ':green[Note that:]  Based on historical data, the smartest cohort are the 10 to 100 btc. \n\n'
                    ':red[Note that:]  Based on historical data, the most emotional cohort are the smallest ones (<1btc).'

            )
            days = st.number_input(
            'Number of days to compare',
            min_value=1,
            max_value=2000,
            value=30,
            step=1
            )
            if st.button('**Evaluate the current situation**'):
                if df['totalVolume_f_10_to_100'].iloc[-1] > df ['totalVolume_f_10_to_100'].iloc[-days]:
                    st.subheader(f"The 10 to 100btc, which are the smartest cohort, are :green[accumulating] since the **{days}** last days. :rocket:")
                else :
                    st.subheader(f"The 10 to 100btc, which are the smartest cohort, are :red[offloading] since the **{days}** last days. :bear:")


            st.plotly_chart(balances_viz_addresses(df, checkbox_val, checkbox_val_metric, ma, type_balance,metrics, checkbox_zscore),
                use_container_width=True)    



    elif metrics == 'unique-addresses':

        try : 
            df = pd.read_csv(f'data/datos/{categorie}_{metrics}.csv', index_col='timestamp')

        except:
            # Define the API endpoint and parameters
            url = "https://api.blockchain.info/charts/n-unique-addresses"
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
            df.to_csv(f'data/datos/{categorie}_{metrics}.csv')
        
        st.header(f'You are looking at `{metrics}` from the category `{categorie}`')

        st.write('Total number of unique addresses used on the blockchain. Increasing = :green[Higher Demand] decreasing = :red[Lower Demand]')

        df = df[-days_to_plot:]
        
        
        days = st.number_input(
            'Number of days to compare',
            min_value=1,
            max_value=2000,
            value=30,
            step=1
            )
                
        if st.button('**Evaluate the current situation**'):
            if df['metric'].iloc[-1] > df ['metric'].iloc[-days:-1].mean():
                st.subheader(f"{metrics} is :green[higher] than compared to the last **{days}** days' average. :rocket:")
            else :
                st.subheader(f"{metrics} is :red[lower] than compared to the last **{days}** days' average. :bear:")
        col1, col2 = st.columns(2)
        col1.metric(f"Last value for {metrics}", df['metric'].iloc[-1] )
        col2.metric(f"Average value of the last **{days}** days of {metrics}", round(df['metric'].iloc[-days:-1].mean(),0) )

        st.plotly_chart(viz_with_indicator(df, checkbox_val, checkbox_val_metric, ma, metrics,checkbox_zscore ),
                        use_container_width=True)   



    else:
        df = on_chain_merge(onchain, metrics)
        st.header(f'You are looking at `{metrics}` from the category `{onchain}`')
        df = df[-days_to_plot:]

        days = st.number_input(
            'Number of days to compare',
            min_value=1,
            max_value=2000,
            value=30,
            step=1
            )
                
        if st.button('**Evaluate the current situation**'):
            if df['metric'].iloc[-1] > df ['metric'].iloc[-days:-1].mean():
                st.subheader(f"{metrics} is :green[higher] than compared to the last **{days}** days' average. :rocket:")
            else :
                st.subheader(f"{metrics} is :red[lower] than compared to the last **{days}** days' average. :bear:")
        col1, col2 = st.columns(2)
        col1.metric(f"Last value for {metrics}", round(df['metric'].iloc[-1],2) )
        col2.metric(f"Average value of the last **{days}** days of {metrics}", round(df['metric'].iloc[-days:-1].mean(),2) )

        st.plotly_chart(on_chain_viz_zscore(df, checkbox_val, checkbox_val_metric, ma, onchain, metrics,checkbox_zscore ),
                        use_container_width=True)   
