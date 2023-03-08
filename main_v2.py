import pandas as pd
import streamlit as st
from functions.candlestick_v2 import *
from functions.get_btc import btc
import sys
import numpy as np
import sys
from get_merge_btc_M2s import *
from functions.patterns import *
from functions.on_chain import glassnode, on_chain_df




sys.path.insert(0, 'C:/Users/axell/Documents/BitcoinML/streamlit')
btc = btc()
df_btc = pd.DataFrame(btc)


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
mining = dic.metric[dic.category == 'mining'].values
transactions = dic.metric[dic.category == 'transactions'].values
fees = dic.metric[dic.category == 'fees'].values
indicators = dic.metric[dic.category == 'indicators'].values
blockchain = dic.metric[dic.category == 'blockchain'].values
supply = dic.metric[dic.category == 'supply'].values
market = dic.metric[dic.category == 'market'].values
institutions = dic.metric[dic.category == 'institutions'].values
signals = dic.metric[dic.category == 'signals'].values
categories = list(ind.category)

## Sidebar options

# titre sidebar
st.sidebar.header('Dashboard')


#st.sidebar.subheader('Données')
categorie = st.sidebar.selectbox("**catégorie**", ('Technique', 'Macro', 'On-Chain'))

if categorie == 'Technique' or categorie == 'Macro':

    with st.sidebar.form("Indicateurs"):
        #st.write("Selectionnez")
        if categorie == 'Technique':
            indicateur = st.selectbox('Indicateurs techniques ', ('Prix', 'Bull-Market Support Bands', 'EHMA'))
        elif categorie == 'Macro':
            indicateur = st.selectbox('Indicateurs macro-économiques', ('Masse Monétaire', 'DXY'))



        days_to_plot = st.slider(
            'Nombre de jours',
            min_value=1,
            max_value=len(df_btc),
            value=len(df_btc)
        )

        checkbox_val = st.checkbox("Logarithmic")

        # Every form must have a submit button.
        submitted = st.form_submit_button("**Envoyer**")

else:
    onchain = st.sidebar.selectbox('onchain', (sorted(set(categories))))
    with st.sidebar.form("On-Chain"):
        if onchain == 'addresses':
            metrics = st.selectbox("**metrics**", addresses)
        elif onchain == 'blockchain':
            metrics = st.selectbox("**metrics**", blockchain)
        elif onchain == 'mining':
            metrics = st.selectbox("**metrics**", mining)
        elif onchain == 'transactions':
            metrics = st.selectbox("**metrics**", transactions)
        elif onchain == 'fees':
            metrics = st.selectbox("**metrics**", fees)
        elif onchain == 'indicators':
            metrics = st.selectbox("**metrics**", indicators)
        elif onchain == 'supply':
            metrics = st.selectbox("**metrics**", supply)
        elif onchain == 'market':
            metrics = st.selectbox("**metrics**", market)
        elif onchain == 'institutions':
            metrics = st.selectbox("**metrics**", institutions)
        elif onchain == 'signals':
            metrics = st.selectbox("**metrics**", signals)

        days_to_plot = st.slider(
        'Nombre de jours',
        min_value=1,
        max_value=len(df_btc),
        value=len(df_btc)
        )

        checkbox_val = st.checkbox("Logarithmic")

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
           
        c1, c2 = st.columns(2)
        with c1:
            df = df_btc.copy()
            df.reset_index(drop=False, inplace=True)
            df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

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

    elif indicateur == 'Bull-Market Support Bands': 
        st.header('Bitcoin `Bull-Market Support Bands`')


        df = df_btc.copy()
        df['20w_sma'] = df['Close'].rolling(140).mean()
        df['21w_ema'] = df['Close'].ewm(span=21, adjust=False).mean()
        df = df[-days_to_plot:]

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

        # Determine the y-axis type
        if checkbox_val == True:
            st.plotly_chart(
            get_candlestick_plot_EHMA(df, True, 'btc' ),
            use_container_width=True)   
            
        else : 
           st.plotly_chart(get_candlestick_plot_EHMA(df, False, 'btc'),
            use_container_width=True)     

        


elif categorie == 'Macro':
    if indicateur == 'Masse Monétaire': 
        st.header('Bitcoin vs `M2`')
        # importing Merged DF (BTC and M2s) 

        df = merged_btc_M2s()
        merged = pd.DataFrame(df)

        df = df_btc.copy()
        df = df[-days_to_plot:]

        st.plotly_chart(z_score(merged, 'M2_Fed_and_ECB') ,use_container_width=True)



elif categorie == 'On-Chain':
    try:
        df = on_chain_df(onchain, metrics)
        st.header(f'You are looking at `{metrics}` from the category **{onchain}**')
        st.table(df.tail())
    except: 
        st.header(f'`{metrics}`est indisponible pour le moment')



    
