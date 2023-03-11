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
import yfinance as yf
from datetime import datetime
import requests



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
addresses = dic.metric[dic.category == 'addresses'].values
transactions = dic.metric[dic.category == 'transactions'].values
indicators = dic.metric[dic.category == 'indicators'].values
supply = dic.metric[dic.category == 'supply'].values
market = dic.metric[dic.category == 'market'].values
institutions = dic.metric[dic.category == 'institutions'].values
categories = list(ind.category)


## Sidebar options

# titre sidebar
st.sidebar.header('Dashboard')

#st.sidebar.subheader('Données')
categorie = st.sidebar.selectbox("**catégorie**", ('Technique', 'Macro', 'Mining', 'On-Chain'))

if categorie == 'Technique':

    with st.sidebar.form("Indicateurs"):
        indicateur = st.selectbox('Indicateurs techniques ', ('Prix', 'Bull-Market Support Bands', 'EHMA'))

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
        st.caption('Current pricing canals :chart_with_upwards_trend: :chart_with_downwards_trend:')

        
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
        

        # Determine the y-axis type
        if checkbox_val == True:
            st.plotly_chart(
            get_candlestick_plot(df, True, 'btc' ),
            use_container_width=True)   
            

        else : 
           st.plotly_chart(get_candlestick_plot(df, False, 'btc'),
            use_container_width=True)    
           


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
        st.caption('Bull-Market Support bands are a good support during :green[bullmarkets] and a strong resistence during :red[bearmarkets]')


        if st.button('**Evaluate the current situation**'):

            if df_btc.iloc[-1,0] > min(df['21w_ema'][-1],df['20w_sma'][-1],0):
                st.subheader('Currently we are in a :green[bullmarket] :rocket:')

            else :
                st.subheader('Currently we are in a :red[bearmarket] :bear:')



        # Determine the y-axis type
        
        if checkbox_val == True:
            st.plotly_chart(
            get_candlestick_plot_ma(df, True, 'btc' ),
            use_container_width=True)   
            
            

        else : 
           st.plotly_chart(get_candlestick_plot_ma(df, False, 'btc'),
            use_container_width=True)            

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
        st.caption('**Traditional moving averages lag the price activity. But with some clever mathematics the lag can be minimised. If green we are in a :green[bullmarkets], if red we are in a :red[bearmarkets]**')


        if st.button('**Evaluate the current situation**'):

            if ehma[-1] > ehma[-2]:
                st.subheader('Currently we are in a :green[bullmarket] :rocket:')

            else :
                st.subheader('Currently we are in a :red[bearmarket] :bear:')

        tab1, tab2= st.tabs(["Chart", "Prediction"])

        with tab1:
            st.header("Chart")
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

elif categorie == 'Macro':
    if indicateur == 'Masse Monétaire': 

        st.header('Bitcoin vs `money printing (ECB+FED)` :money_with_wings:')

        st.caption('Traditional currency can be "printed" without limits by central banks.\n\n'
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
                st.subheader('Currently we are in a :green[money printing market] :rocket:')

            else :
                st.subheader('Currently we are in a :red[tightning] :bear:')



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

        st.caption('The dollar index tracks the relative value of the U.S. dollar against a basket of important world currencies.\n'
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
        

        st.caption('The hashrate is the computing power miners use to secure the Bitcoin network. \n\n'
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
        

        st.plotly_chart(viz_with_indicator(df, checkbox_val, checkbox_val_metric, ma, indicateur,checkbox_zscore ),
                        use_container_width=True)   


    elif indicateur == 'volume_sum': 
        df = on_chain_merge('fees', indicateur)
        st.header(f'You are looking at `{indicateur}` from the category `mining`')
        df = df[-days_to_plot:]
        st.caption('The total amount of fees paid to miners. Inflation rewards not included. Increasing = :green[Higher Demand] decreasing = :red[Lower Demand]')


        st.plotly_chart(on_chain_viz_zscore(df, checkbox_val, checkbox_val_metric, ma, 'fees', indicateur, checkbox_zscore ),
                        use_container_width=True)   





elif categorie == 'On-Chain':
    

    if metrics == 'unique-addresses':

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

        st.caption('Total number of unique addresses used on the blockchain. Increasing = :green[Higher Demand] decreasing = :red[Lower Demand]')

        df = df[-days_to_plot:]
        
        st.plotly_chart(viz_with_indicator(df, checkbox_val, checkbox_val_metric, ma, metrics,checkbox_zscore ),
                        use_container_width=True)   



    else:
        df = on_chain_merge(onchain, metrics)
        st.header(f'You are looking at `{metrics}` from the category `{onchain}`')
        df = df[-days_to_plot:]

        st.plotly_chart(on_chain_viz_zscore(df, checkbox_val, checkbox_val_metric, ma, onchain, metrics,checkbox_zscore ),
                        use_container_width=True)   
