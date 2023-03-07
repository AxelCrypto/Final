import pandas as pd
import streamlit as st
from functions.candlestick_v2 import *
from functions.get_btc import btc
import sys
import numpy as np
import sys
from get_merge_btc_M2s import *
from functions.patterns import *





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


## Sidebar options

# titre sidebar
st.sidebar.header('Dashboard')


#st.sidebar.subheader('Données')
categorie = st.sidebar.selectbox("**catégorie**",('Technique','Macro', 'On-Chain')) 


with st.sidebar.form("Indicateurs"):
   st.write("Selectionnez")
   if categorie == 'Technique':
    indicateur = st.selectbox('Indicateurs techniques ', ('Prix','Bull-Market Support Bands', 'EHMA')) 
   elif categorie == 'Macro':
    indicateur = st.selectbox('Indicateurs macro-économiques', ('Masse Monétaire','DXY')) 
   elif categorie == 'On-Chain':
    indicateur = st.selectbox('Indicateurs On-Chain', ('LT vs ST','MVRV', 'Realised Price')) 


   days_to_plot = st.slider(
    'Nombre de jours', 
    min_value = 1,
    max_value = len(df_btc),
    value = len(df_btc))
   
   checkbox_val = st.checkbox("Logarithmic")

   # Every form must have a submit button.
   submitted = st.form_submit_button("Envoyer")
   if submitted:
       st.write('indicateur', indicateur,"slider", days_to_plot, "checkbox", checkbox_val)


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



if categorie == 'Technique':
    if indicateur == 'Prix':    
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

            frame = 120

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