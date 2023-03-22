import sys
import pandas as pd
import streamlit as st
from functions.candlestick_v2 import *
from functions.get_btc import btc
import numpy as np
from functions.get_merge_btc_M2s import *
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
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objs as go
import json


df_btc = btc()

st.set_page_config(
    page_title="Inside Bitcoin's Price Rabbit hole",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://twitter.com/AxelCryptoytb',
        'Report a bug': "https://twitter.com/AxelCryptoytb",
        'About': "Welcome into bitcoin's price rabbit hole. This is a simple app to help you visualise bitcoin's prices influencers. Nothing can be taken for a financial advise. *@AxelCrypto*"    }
)

#Options download
ind = pd.read_csv('data/indicators_labels.csv')
dic = ind[['category','metric']]
categories = list(ind.category)
categories.append('balances')


## Sidebar options

# titre sidebar
#st.image('images/logo.png', width = 150, clamp =255)

st.sidebar.header(':blue[**DASHBOARD**]')

#st.sidebar.subheader('Données')
categorie = st.sidebar.selectbox("**categories**", ('Technical', 'Macro', 'Mining', 'On-Chain'))

if categorie == 'Technical':

    with st.sidebar.form("Indicators"):
        indicateur = st.selectbox('Technical Indicators', ('Price', 'Price pattern', 'Bull-Market Support Bands', 'EHMA', 'Mayer Multiple', 'Puell Multiple'))

        checkbox_val = st.checkbox("Logarithmic")

        # Every form must have a submit button.
        submitted = st.form_submit_button("**Send**")
 
elif categorie == 'Macro':
    with st.sidebar.form("Macro"):
        indicateur = st.selectbox('Macro Indicators', ('Money Supply', 'DXY'))

        checkbox_val = st.checkbox("Logarithmic")
        checkbox_val_metric = st.checkbox("Logarithmic Indicators")
        ma = st.slider("Indicator's MA", min_value=1, max_value=90, value=1)

        #checkbox_zscore = st.checkbox("Activer le Z-Score")

        # Every form must have a submit button.
        submitted = st.form_submit_button("**Send**")

elif categorie == 'Mining':
    with st.sidebar.form("Mining"):
        indicateur = st.selectbox('Mining Indicators', ('Hashrate', 'Total Transaction Fees (BTC)','volume_sum'))

        checkbox_val = st.checkbox("Logarithmic")
        checkbox_val_metric = st.checkbox("Logarithmic Indicator")
        ma = st.slider("Indicator's MA", min_value=1, max_value=90, value=1)

        checkbox_zscore = st.checkbox("Activate Z-Score")

        submitted = st.form_submit_button("**Send**")

else:
    onchain = st.sidebar.selectbox('**Type**', (sorted(set(categories))))
    with st.sidebar.form("On-Chain"):
        if onchain == 'addresses':
            metrics = st.selectbox("**metrics**", ('active_count','new_non_zero_count','unique-addresses'))
        elif onchain == 'balances':
            type_balance = st.selectbox("**Type of Balances**", ('Adresses Count','Balances'))
            metrics = st.selectbox("**metrics**", ('<0.01','0.01 - 0.1','0.1 - 1', '1 - 10', '10 - 100','100 - 1000', '1k - 10k','10k+'))
        elif onchain == 'transactions':
            metrics = st.selectbox("**metrics**", ('count','size_mean'))
        elif onchain == 'indicators':
            metrics = st.selectbox("**metrics**", ('sopr',
                                                   'stock_to_flow_ratio'
                                                   ,
                                                   #'pi_cycle_top'
                                                   ))
        elif onchain == 'supply':
            metrics = 'active_more_1y_percent'
        elif onchain == 'market':
            metrics = st.selectbox("**metrics**", ('price_drawdown_relative','marketcap_usd'))
        elif onchain == 'institutions':
            metrics = st.selectbox("**metrics**", ('purpose_etf_holdings_sum','purpose_etf_flows_sum','purpose_etf_aum_sum'))

        checkbox_val = st.checkbox("Logarithmic")
        checkbox_val_metric = st.checkbox("Logarithmic Indicator")
        ma = st.slider("Indicator's MA", min_value=1, max_value=90, value=1)

        checkbox_zscore = st.checkbox("Activate Z-Score")

        # Every form must have a submit button.
        submitted = st.form_submit_button("**Send**")

st.sidebar.markdown('''
---
GM Satoshi 🕵️  
By **[Axel](https://twitter.com/AxelCryptoytb)**    
*Data should NOT be used for trading and investing.*
''')

st.sidebar.markdown(f'Last Update: {df_btc.index[-1]}')


# Charts selon la selection:

if categorie == 'Technical':
    if indicateur == 'Price': 
        st.title(":green[GM Satoshi] 🕵️, Welcome to the *Bitcoin's Price Rabbit hole!*")
        st.header('Bitcoin `Actual price`')
        st.metric("Last  price", f'${df_btc.iloc[-1,0]}', f'{round((df_btc.iloc[-1,0]/df_btc.iloc[-2,0]-1 ) *100,2)}%')
        df = df_btc.copy()


        tab1, tab2= st.tabs(["Chart", "Prediction"])

        #display chart:
        with tab1:
            days_to_plot = st.slider(
                'Days to plot',
                min_value=1,
                max_value=len(df_btc),
                value=len(df_btc)
            )
            df = df[-days_to_plot:]
        
            st.plotly_chart(get_candlestick_plot(df, checkbox_val, 'btc'),
                use_container_width=True)    

        #display fbprophet:
        with tab2:
            st.title(':blue[Prediction based on machine learning]')
            st.subheader('Based on historical price only')
            st.markdown("Note that the output can take time.")


            c1, c2 = st.columns(2)
            with c1:
                view = st.radio(
                    ":blue[Select view] :chart_with_upwards_trend:",
                    key="logarithmic",
                    options=["linear", "logarithmic"],
                )

            with c2:
                time = st.radio(
                    ":blue[Select timeframe] :hourglass_flowing_sand:",
                    key="weekly",
                    options=["daily", "weekly", 'monthly'],
                )

            if st.button('**Make Prediction**'):
                #if view == "logarithmic":
                #    st.write('**:red[Attention, logarithmic is not fitted for the diminishing returns]**')
                #else: pass

                df_fb = df.reset_index(drop= False)
                df = df_fb[['timestamp','Close']]
                df.columns = ['ds','y']
                if view == "logarithmic":
                    df['y']= np.log(df.y)
                else: pass

                if time == 'days':
                    pass
                elif time == 'weekly':
                    df = df[::7]
                elif time == 'monthly':
                    df = df[::30]

                model = Prophet(
                    daily_seasonality= True,
                    weekly_seasonality= True,
                    yearly_seasonality = True
                )

                if time == 'daily':
                    model.add_seasonality(name='4yearly', period=1461, fourier_order=10)
                elif time == 'weekly':
                    model.add_seasonality(name='4yearly', period=1461/7, fourier_order =5)
                elif time == 'monthly':
                    model.add_seasonality(name='4yearly', period=1461/7/12, fourier_order =4)

                model.fit(df)
                
                if time == 'daily':
                    days = 60
                    future_dates = model.make_future_dataframe(periods = days, freq='D')
                    st.subheader(f':blue[Prediction at {days} days: ]')

                elif time == 'weekly':
                    weeks = 45
                    future_dates = model.make_future_dataframe(periods = weeks, freq='W')
                    st.subheader(f':blue[Prediction at {weeks} weeks: ]')

                elif time == 'monthly':
                    months = 12
                    future_dates = model.make_future_dataframe(periods = months, freq='M')
                    st.subheader(f':blue[Prediction at {months} months: ]')

                    
                prediction = model.predict(future_dates)


                fig = plot_plotly(model, prediction)

                #st.subheader(f':blue[Prediction at {days} days: ]')
                st.plotly_chart(go.Figure(fig))
            
                fig2 = plot_components_plotly(model, prediction)
                
                st.subheader(':blue[:Historical seasonalities detected by the model: ]')
                st.plotly_chart(fig2)

    elif indicateur == 'Price pattern':

        st.subheader('Current pricing canals :chart_with_upwards_trend: :chart_with_downwards_trend:')      
           
        days_1 = st.slider(
        'Pattern days',
        min_value=4,
        max_value=500,
        value=95
    )
        total_1 = st.slider(
        'Total Frame',
        min_value=1,
        max_value=len(df_btc),
        value=365
    )   

        df = df_btc.copy()
        df.reset_index(drop=False, inplace=True)
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        df.Date = pd.to_datetime(df.Date)

        one_year = total_1

        # Filter the dates to plot only the region of interest
        df = df[(df['Date'] > max(df['Date']) - one_year * pd.offsets.Day())]
        df = df.reset_index(drop=True)

        frame = days_1

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
            title=f'Price patterns of the last {frame} days',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Price'},
            legend={'x': 0, 'y': 1.075, 'orientation': 'h'},
            width=500,
            height=700,
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

        st.plotly_chart(fig, use_container_width=True)

    elif indicateur == 'Bull-Market Support Bands': 
        st.header('Bitcoin `Bull-Market Support Bands`')

        df = df_btc.copy()
        df['20w_sma'] = df['Close'].rolling(140).mean()
        df['21w_ema'] = df['Close'].ewm(span=21, adjust=False).mean()


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



        tab1, tab2= st.tabs(["Chart", "Prediction"])


        with tab1:
            days_to_plot = st.slider(
                'Number of days',
                min_value=1,
                max_value=len(df_btc),
                value=len(df_btc)
            )
            df = df[-days_to_plot:]
            st.plotly_chart(get_candlestick_plot_ma(df, checkbox_val, 'btc' ),
                use_container_width=True)   


        with tab2:
            st.title(':blue[Prediction based on machine learning]')
            st.subheader(f'Based on historical price and its correlation with {indicateur}')
            st.markdown("Note that the output can take time.")
            
 
            c1, c2 = st.columns(2)
            with c1:
                view = st.radio(
                    ":blue[Select view] :chart_with_upwards_trend:",
                    key="linear",
                    options=["linear", "logarithmic"],
                )

            with c2:
                time = st.radio(
                    ":blue[Select timeframe] :hourglass_flowing_sand:",
                    key="weekly",
                    options=["daily", "weekly"],
                )

            if st.button('**Make Prediction**'):
                st.write(df.head())
                df.reset_index(drop = False, inplace = True)
                df['metric'] = df['Close'].rolling(140).mean()
                df = df[['timestamp', 'Close', 'metric']]
                df.dropna(inplace = True)
                df.columns = ['ds', 'y', 'metric']
                
                if view == "logarithmic":
                    df['y']= np.log(df.y)
                else: pass

                if time == 'daily':
                    pass
                
                elif time == 'weekly':
                    df = df.iloc[::7]

            
                # Initialize Prophet model with regressor and fit to data

                model = Prophet()

                if time == 'daily':
                    model.add_seasonality(name='4yearly', period=1461, fourier_order=10)
                elif time == 'weekly':
                    model.add_seasonality(name='4yearly', period=1461/7, fourier_order =8)


                model.add_regressor('metric')
                model.fit(df)


                # Make predictions and plot the results
                if time == 'daily':
                    days = 90
                    future_dates = model.make_future_dataframe(periods = days, freq='D')
                    st.subheader(f':blue[Prediction at {days} days: ]')

                elif time == 'weekly':
                    weeks = 45
                    future_dates = model.make_future_dataframe(periods = weeks, freq='W')
                    st.subheader(f':blue[Prediction at {weeks} weeks: ]')


                
                future_dates['metric'] = df['metric']
                

                prediction = model.predict(future_dates.dropna())

                fig = plot_plotly(model, prediction)

                st.plotly_chart(go.Figure(fig))
            
                fig2 = plot_components_plotly(model, prediction)
                
                st.subheader(':blue[:Historical seasonalities detected by the model: ]')
                st.plotly_chart(fig2)
                
    elif indicateur == 'EHMA': 
        st.header('Bitcoin `EHMA`')

        df = df_btc.copy()

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
        
        st.write('**Traditional moving averages lag the price activity. But with some clever mathematics the lag can be minimised. If green we are in a :green[bullmarkets], if red we are in a :red[bearmarkets]**')


        if st.button('**Evaluate the current situation**'):

            if ehma[-1] > ehma[-2]:
                st.subheader('We are currently in a :green[bullmarket] :rocket:')

            else :
                st.subheader('We are currently in a :red[bearmarket] :bear:')




        tab1, tab2= st.tabs(["Chart", "Prediction"])

        with tab1:
            days_to_plot = st.slider(
                'Days to plot',
                min_value=1,
                max_value=len(df_btc),
                value=len(df_btc)
            )
            df = df[-days_to_plot:]
            # Determine the y-axis type
            if checkbox_val == True:
                st.plotly_chart(
                get_candlestick_plot_EHMA(df, True, 'btc' ),
                use_container_width=True)   
                
            else : 
                st.plotly_chart(get_candlestick_plot_EHMA(df, False, 'btc'),
                    use_container_width=True)     
        

        with tab2:
            st.title(':blue[Prediction based on machine learning]')
            st.subheader(f'Based on historical price and its correlation with {indicateur}')
            st.markdown("Note that the output can take time.")
            
            c1, c2 = st.columns(2)
            with c1:
                view = st.radio(
                    ":blue[Select view] :chart_with_upwards_trend:",
                    key="linear",
                    options=["linear", "logarithmic"],
                )

            with c2:
                time = st.radio(
                    ":blue[Select timeframe] :hourglass_flowing_sand:",
                    key="weekly",
                    options=["daily", "weekly"],
                )

            if st.button('**Make Prediction**'):
                df.reset_index(drop = False, inplace = True)
                df['metric'] = ehma

                df = df[['timestamp', 'Close', 'metric']]
                df.dropna(inplace = True)
                df.columns = ['ds', 'y', 'metric']
                
                if view == "logarithmic":
                    df['y']= np.log(df.y)
                else: pass

                if time == 'daily':
                    pass
                
                elif time == 'weekly':
                    df = df.iloc[::7]

            
                # Initialize Prophet model with regressor and fit to data

                model = Prophet()

                if time == 'daily':
                    model.add_seasonality(name='4yearly', period=1461, fourier_order=10)
                elif time == 'weekly':
                    model.add_seasonality(name='4yearly', period=1461/7, fourier_order =8)


                model.add_regressor('metric')
                model.fit(df)


                # Make predictions and plot the results
                if time == 'daily':
                    days = 90
                    future_dates = model.make_future_dataframe(periods = days, freq='D')
                    st.subheader(f':blue[Prediction at {days} days: ]')

                elif time == 'weekly':
                    weeks = 45
                    future_dates = model.make_future_dataframe(periods = weeks, freq='W')
                    st.subheader(f':blue[Prediction at {weeks} weeks: ]')


                
                future_dates['metric'] = df['metric']
                

                prediction = model.predict(future_dates.dropna())

                fig = plot_plotly(model, prediction)

                st.plotly_chart(go.Figure(fig))
            
                fig2 = plot_components_plotly(model, prediction)
                
                st.subheader(':blue[:Historical seasonalities detected by the model: ]')
                st.plotly_chart(fig2)
                

    elif indicateur == 'Mayer Multiple':
        st.header('Bitcoin `Mayer Multiple`')
        df = df_btc.copy()
        df['200d_ma'] = df['Close'].rolling(200).mean()
        df['metric'] = df['Close'] / df['200d_ma']


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
            days_to_plot = st.slider(
                'Days to plot',
                min_value=1,
                max_value=len(df_btc),
                value=len(df_btc)
            )
            df = df[-days_to_plot:]
            st.plotly_chart(viz_with_indicator(df, checkbox_val, True, 1, indicateur, False),
                    use_container_width=True)     

        with tab2:
            st.title(':blue[Prediction based on machine learning]')
            st.subheader(f'Based on historical price and its correlation with {indicateur}')
            st.markdown("Note that the output can take time.")
            
            c1, c2 = st.columns(2)
            with c1:
                view = st.radio(
                    ":blue[Select view] :chart_with_upwards_trend:",
                    key="linear",
                    options=["linear", "logarithmic"],
                )

            with c2:
                time = st.radio(
                    ":blue[Select timeframe] :hourglass_flowing_sand:",
                    key="weekly",
                    options=["daily", "weekly"],
                )

            if st.button('**Make Prediction**'):
                df.reset_index(drop = False, inplace = True)
                df = df[['timestamp', 'Close', 'metric']]
                df.dropna(inplace = True)
                df.columns = ['ds', 'y', 'metric']
                
                if view == "logarithmic":
                    df['y']= np.log(df.y)
                else: pass

                if time == 'daily':
                    pass
                
                elif time == 'weekly':
                    df = df.iloc[::7]

            
                # Initialize Prophet model with regressor and fit to data

                model = Prophet()

                if time == 'daily':
                    model.add_seasonality(name='4yearly', period=1461, fourier_order=10)
                elif time == 'weekly':
                    model.add_seasonality(name='4yearly', period=1461/7, fourier_order =8)


                model.add_regressor('metric')
                model.fit(df)


                # Make predictions and plot the results
                if time == 'daily':
                    days = 90
                    future_dates = model.make_future_dataframe(periods = days, freq='D')
                    st.subheader(f':blue[Prediction at {days} days: ]')

                elif time == 'weekly':
                    weeks = 45
                    future_dates = model.make_future_dataframe(periods = weeks, freq='W')
                    st.subheader(f':blue[Prediction at {weeks} weeks: ]')


                
                future_dates['metric'] = df['metric']
                

                prediction = model.predict(future_dates.dropna())

                fig = plot_plotly(model, prediction)

                st.plotly_chart(go.Figure(fig))
            
                fig2 = plot_components_plotly(model, prediction)
                
                st.subheader(':blue[:Historical seasonalities detected by the model: ]')
                st.plotly_chart(fig2)
                

    elif indicateur == 'Puell Multiple':
        st.header('Bitcoin `Puell Multiple`')
        df = df_btc.copy()
        df['365d_ma'] = df['Close'].rolling(365).mean()
        df['metric'] = df['Close'] / df['365d_ma']


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
            days_to_plot = st.slider(
                'Days to plot',
                min_value=1,
                max_value=len(df_btc),
                value=len(df_btc)
            )
            df = df[-days_to_plot:]

            st.plotly_chart(viz_with_indicator(df, checkbox_val, True, 1, indicateur, False),
                    use_container_width=True)     
            
        with tab2:
            st.title(':blue[Prediction based on machine learning]')
            st.subheader(f'Based on historical price and its correlation with {indicateur}')
            st.markdown("Note that the output can take time.")
            
            c1, c2 = st.columns(2)
            with c1:
                view = st.radio(
                    ":blue[Select view] :chart_with_upwards_trend:",
                    key="linear",
                    options=["linear", "logarithmic"],
                )

            with c2:
                time = st.radio(
                    ":blue[Select timeframe] :hourglass_flowing_sand:",
                    key="weekly",
                    options=["daily", "weekly"],
                )

            if st.button('**Make Prediction**'):
                df.reset_index(drop = False, inplace = True)
                df = df[['timestamp', 'Close', 'metric']]
                df.dropna(inplace = True)
                df.columns = ['ds', 'y', 'metric']
                
                if view == "logarithmic":
                    df['y']= np.log(df.y)
                else: pass

                if time == 'daily':
                    pass
                
                elif time == 'weekly':
                    df = df.iloc[::7]

            
                # Initialize Prophet model with regressor and fit to data

                model = Prophet()

                if time == 'daily':
                    model.add_seasonality(name='4yearly', period=1461, fourier_order=10)
                elif time == 'weekly':
                    model.add_seasonality(name='4yearly', period=1461/7, fourier_order =8)


                model.add_regressor('metric')
                model.fit(df)


                # Make predictions and plot the results
                if time == 'daily':
                    days = 90
                    future_dates = model.make_future_dataframe(periods = days, freq='D')
                    st.subheader(f':blue[Prediction at {days} days: ]')

                elif time == 'weekly':
                    weeks = 45
                    future_dates = model.make_future_dataframe(periods = weeks, freq='W')
                    st.subheader(f':blue[Prediction at {weeks} weeks: ]')


                
                future_dates['metric'] = df['metric']
                

                prediction = model.predict(future_dates.dropna())

                fig = plot_plotly(model, prediction)

                st.plotly_chart(go.Figure(fig))
            
                fig2 = plot_components_plotly(model, prediction)
                
                st.subheader(':blue[:Historical seasonalities detected by the model: ]')
                st.plotly_chart(fig2)
                

elif categorie == 'Macro':
    if indicateur == 'Money Supply': 

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




        tab1, tab2= st.tabs(["Chart", "Prediction"])

        with tab1:
            days_to_plot = st.slider(
                'Days to plot',
                min_value=1,
                max_value=len(df_btc),
                value=len(df_btc)
            )
            df = df[-days_to_plot:]


            st.plotly_chart(macro_zscore(df, checkbox_val, checkbox_val_metric, ma, 
                                        #checkbox_zscore, 
                                        indicateur ),
                            use_container_width=True)   

        with tab2:
            st.title(':blue[Prediction based on machine learning]')
            st.subheader('Based on historical price and its correlation with M2 (Eur + USD)')
            st.markdown("Note that the output can take time.")


           # c1, c2 = st.columns(2)
            #with c1:
             #   view = st.radio(
              #      ":blue[Select view] :chart_with_upwards_trend:",
               #     key="linear",
                #    options=["linear", "logarithmic"],
                #)

            #with c2:
            time = st.radio(
                ":blue[Select timeframe] :hourglass_flowing_sand:",
                key="weekly",
                options=["daily", "weekly", 'monthly'],
            )

            if st.button('**Make Prediction**'):
                #if view == "logarithmic":
                #    st.write('**:red[Attention, logarithmic is not fitted for the diminishing returns]**')
                #else: pass
                view = "logarithmic"

                try:
                    df.reset_index(drop = False, inplace = True)
                except: pass
                df = df[['index', 'Close', 'M2_sum']]
                df.columns = ['ds', 'y', 'M2_sum']

                if view == "logarithmic":
                    df['y']= np.log(df.y)
                else: pass

                if time == 'daily':
                    pass
                
                elif time == 'weekly':
                    df = df[::7]
                elif time == 'monthly':
                    df = df[::30]


            
                # Initialize Prophet model with regressor and fit to data

                #model = Prophet()
                model = Prophet(
                    daily_seasonality= True,
                    weekly_seasonality= True,
                    yearly_seasonality = True
                )

                if time == 'daily':
                    model.add_seasonality(name='4yearly', period=1461, fourier_order=10)
                elif time == 'weekly':
                    model.add_seasonality(name='4yearly', period=1461/7, fourier_order =8)
                elif time == 'monthly':
                    model.add_seasonality(name='4yearly', period=1461/30, fourier_order =5)

                model.add_regressor('M2_sum')
                model.fit(df)


                # Make predictions and plot the results
                if time == 'daily':
                    days = 60
                    future_dates = model.make_future_dataframe(periods = days, freq='D')
                    st.subheader(f':blue[Prediction at {days} days: ]')

                elif time == 'weekly':
                    weeks = 45
                    future_dates = model.make_future_dataframe(periods = weeks, freq='W')
                    st.subheader(f':blue[Prediction at {weeks} weeks: ]')

                elif time == 'monthly':
                    months = 12
                    future_dates = model.make_future_dataframe(periods = months, freq='M')
                    st.subheader(f':blue[Prediction at {months} months: ]')


                future_dates['M2_sum'] = df['M2_sum']
                prediction = model.predict(future_dates.dropna())

                fig = plot_plotly(model, prediction)

                st.plotly_chart(go.Figure(fig))
            
                fig2 = plot_components_plotly(model, prediction)
                
                st.subheader(':blue[:Historical seasonalities detected by the model: ]')
                st.plotly_chart(fig2)

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
        
                
        f = open('backtesting/correlations_raw.json')
        data = json.load(f)

        st.write("*The correlation coefficient with bitcoin's price is of : *", data['dxy'])


       
        days = st.number_input('Number of days to compare',
                            min_value=1,
                            max_value=3650,
                            value=30,
                            step=1)


        if st.button('**Evaluate the current situation**'):

            


            if dxy ['Close'][-1] < dxy ['Close'][-days-1:].mean():
                st.subheader(f'DXY is currently :green[lower] than the **{days}** average last days. :rocket:')

            else :
                st.subheader(f'DXY is currently :red[higher] than the **{days}** average last days. :bear:')


        

        tab1, tab2= st.tabs(["Chart", "Prediction"])

        with tab1:
            days_to_plot = st.slider(
                'Days to plot dxy',
                min_value=1,
                max_value=len(df_btc),
                value=len(df_btc)
            )
            days_to_plot_btc = st.slider(
                'Days to plot btc',
                min_value=1,
                max_value=len(df_btc),
                value=len(df_btc)
            )
            dxy = dxy[-days_to_plot:]
            df_btc = df_btc[-days_to_plot_btc:]


            st.plotly_chart(macro_dxy(df_btc,dxy,checkbox_val, checkbox_val_metric, ma ),
                        use_container_width=True)   


        with tab2:
            st.title(':blue[Prediction based on machine learning]')
            st.subheader(f'Based on historical price and its correlation with {indicateur}')
            st.markdown("Note that the output can take time.")
            

            
            c1, c2 = st.columns(2)
            with c1:
                view = st.radio(
                    ":blue[Select view] :chart_with_upwards_trend:",
                    key="linear",
                    options=["linear", "logarithmic"],
                )

            with c2:
                time = st.radio(
                    ":blue[Select timeframe] :hourglass_flowing_sand:",
                    key="weekly",
                    options=["daily", "weekly"],
                )

            if st.button('**Make Prediction**'):
                #if view == "logarithmic":
                #    st.write('**:red[Attention, logarithmic is not fitted for the diminishing returns]**')
                #else: pass
                dxy.reset_index(drop=False, inplace = True)
                metr = dxy[['Date','Close']]

                df_btc.reset_index(drop=False, inplace = True)
                df = df_btc[['timestamp','Close']]
                df.columns = ['ds','y']
                metr.columns = ['ds','metric']
                
                df = df.merge(metr,left_on='ds',right_on='ds')

                if view == "logarithmic":
                    df['y']= np.log(df.y)
                else: pass

                if time == 'daily':
                    pass
                
                elif time == 'weekly':
                    df = df.iloc[::7]

            
                # Initialize Prophet model with regressor and fit to data

                model = Prophet()

                if time == 'daily':
                    model.add_seasonality(name='4yearly', period=1461, fourier_order=10)
                elif time == 'weekly':
                    model.add_seasonality(name='4yearly', period=1461/7, fourier_order =8)


                model.add_regressor('metric')
                model.fit(df)


                # Make predictions and plot the results
                if time == 'daily':
                    days = 90
                    future_dates = model.make_future_dataframe(periods = days, freq='D')
                    st.subheader(f':blue[Prediction at {days} days: ]')

                elif time == 'weekly':
                    weeks = 45
                    future_dates = model.make_future_dataframe(periods = weeks, freq='W')
                    st.subheader(f':blue[Prediction at {weeks} weeks: ]')


                
                future_dates['metric'] = df['metric']
                

                prediction = model.predict(future_dates.dropna())

                fig = plot_plotly(model, prediction)

                st.plotly_chart(go.Figure(fig))
            
                fig2 = plot_components_plotly(model, prediction)
                
                st.subheader(':blue[:Historical seasonalities detected by the model: ]')
                st.plotly_chart(fig2)
                

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





        tab1, tab2= st.tabs(["Chart", "Prediction"])

        with tab1:
            
            days_to_plot = st.slider(
                'Days to plot',
                min_value=1,
                max_value=len(df_btc),
                value=len(df_btc)
            )
            df = df[-days_to_plot:]

            st.plotly_chart(viz_with_indicator(df, checkbox_val, checkbox_val_metric, ma, indicateur,checkbox_zscore ),
                                use_container_width=True)   


            url = 'https://www.youtube.com/watch?v=sWrNNh47p3Y'
            st.header('Vidéo explicative de la relation entre Prix du Bitcoin et Hashrate (FR)')
            st.video(url)


        with tab2:
            st.title(':blue[Prediction based on machine learning]')
            st.subheader(f'Based on historical price and its correlation with {indicateur}')
            st.markdown("Note that the output can take time.")


            c1, c2 = st.columns(2)
            with c1:
                view = st.radio(
                    ":blue[Select view] :chart_with_upwards_trend:",
                    key="linear",
                    options=["linear", "logarithmic"],
                )

            with c2:
                time = st.radio(
                    ":blue[Select timeframe] :hourglass_flowing_sand:",
                    key="weekly",
                    options=["daily", "weekly"],
                )

            if st.button('**Make Prediction**'):
                #if view == "logarithmic":
                #    st.write('**:red[Attention, logarithmic is not fitted for the diminishing returns]**')
                #else: pass

                df.reset_index(drop = False, inplace = True)
                df = df[['timestamp', 'Close', 'metric']]
                df.columns = ['ds', 'y', 'metric']
                

                if view == "logarithmic":
                    df['y']= np.log(df.y)
                else: pass

                if time == 'daily':
                    pass
                
                elif time == 'weekly':
                    df = df.iloc[::7]

            
                # Initialize Prophet model with regressor and fit to data

                model = Prophet()

                if time == 'daily':
                    model.add_seasonality(name='4yearly', period=1461, fourier_order=10)
                elif time == 'weekly':
                    model.add_seasonality(name='4yearly', period=1461/7, fourier_order =8)


                model.add_regressor('metric')
                model.fit(df)


                # Make predictions and plot the results
                if time == 'daily':
                    days = 90
                    future_dates = model.make_future_dataframe(periods = days, freq='D')
                    st.subheader(f':blue[Prediction at {days} days: ]')

                elif time == 'weekly':
                    weeks = 45
                    future_dates = model.make_future_dataframe(periods = weeks, freq='W')
                    st.subheader(f':blue[Prediction at {weeks} weeks: ]')


                
                future_dates['metric'] = df['metric']
                

                prediction = model.predict(future_dates.dropna())

                fig = plot_plotly(model, prediction)

                st.plotly_chart(go.Figure(fig))
            
                fig2 = plot_components_plotly(model, prediction)
                
                st.subheader(':blue[:Historical seasonalities detected by the model: ]')
                st.plotly_chart(fig2)


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





        tab1, tab2= st.tabs(["Chart", "Prediction"])

        with tab1:
            days_to_plot = st.slider(
                'Days to plot',
                min_value=1,
                max_value=len(df),
                value=len(df)
            )
            df = df[-days_to_plot:]
            st.plotly_chart(viz_with_indicator(df, checkbox_val, checkbox_val_metric, ma, indicateur,checkbox_zscore ),
                        use_container_width=True)   

        with tab2:
            st.title(':blue[Prediction based on machine learning]')
            st.subheader(f'Based on historical price and its correlation with {indicateur}')
            st.markdown("Note that the output can take time.")


            c1, c2 = st.columns(2)
            with c1:
                view = st.radio(
                    ":blue[Select view] :chart_with_upwards_trend:",
                    key="linear",
                    options=["linear", "logarithmic"],
                )

            with c2:
                time = st.radio(
                    ":blue[Select timeframe] :hourglass_flowing_sand:",
                    key="weekly",
                    options=["daily", "weekly", 'monthly'],
                )

            if st.button('**Make Prediction**'):
                #if view == "logarithmic":
                #    st.write('**:red[Attention, logarithmic is not fitted for the diminishing returns]**')
                #else: pass

                df.reset_index(drop = False, inplace = True)
                df = df[['timestamp', 'Close', 'metric']]
                df.columns = ['ds', 'y', 'metric']

                if view == "logarithmic":
                    df['y']= np.log(df.y)
                else: pass

                if time == 'daily':
                    pass
                
                elif time == 'weekly':
                    df = df[::7]
                elif time == 'monthly':
                    df = df[::30]


            
                # Initialize Prophet model with regressor and fit to data

                #model = Prophet()
                model = Prophet(
                    daily_seasonality= True,
                    weekly_seasonality= True,
                    yearly_seasonality = True
                )

                if time == 'daily':
                    model.add_seasonality(name='4yearly', period=1461, fourier_order=10)
                elif time == 'weekly':
                    model.add_seasonality(name='4yearly', period=1461/7, fourier_order =8)
                elif time == 'monthly':
                    model.add_seasonality(name='4yearly', period=1461/30, fourier_order =5)

                model.add_regressor('metric')
                model.fit(df)


                # Make predictions and plot the results
                if time == 'daily':
                    days = 90
                    future_dates = model.make_future_dataframe(periods = days, freq='D')
                    st.subheader(f':blue[Prediction at {days} days: ]')

                elif time == 'weekly':
                    weeks = 45
                    future_dates = model.make_future_dataframe(periods = weeks, freq='W')
                    st.subheader(f':blue[Prediction at {weeks} weeks: ]')

                elif time == 'monthly':
                    months = 12
                    future_dates = model.make_future_dataframe(periods = months, freq='M')
                    st.subheader(f':blue[Prediction at {months} months: ]')


                future_dates['metric'] = df['metric']
                prediction = model.predict(future_dates.dropna())

                fig = plot_plotly(model, prediction)

                st.plotly_chart(go.Figure(fig))
            
                fig2 = plot_components_plotly(model, prediction)
                
                st.subheader(':blue[:Historical seasonalities detected by the model: ]')
                st.plotly_chart(fig2)

    elif indicateur == 'volume_sum': 
        df = on_chain_merge('fees', indicateur)
        st.header(f'You are looking at `{indicateur}` from the category `mining`')
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


        tab1, tab2= st.tabs(["Chart", "Prediction"])

        with tab1:

            days_to_plot = st.slider(
                'Days to plot',
                min_value=1,
                max_value=len(df),
                value=len(df)
            )
            df = df[-days_to_plot:]

            st.plotly_chart(on_chain_viz_zscore(df, checkbox_val, checkbox_val_metric, ma, 'fees', indicateur, checkbox_zscore ),
                                use_container_width=True)   

        with tab2:
            st.title(':blue[Prediction based on machine learning]')
            st.subheader(f'Based on historical price and its correlation with {indicateur}')
            st.markdown("Note that the output can take time.")


            c1, c2 = st.columns(2)
            with c1:
                view = st.radio(
                    ":blue[Select view] :chart_with_upwards_trend:",
                    key="linear",
                    options=["linear", "logarithmic"],
                )

            with c2:
                time = st.radio(
                    ":blue[Select timeframe] :hourglass_flowing_sand:",
                    key="weekly",
                    options=["daily", "weekly"],
                )

            if st.button('**Make Prediction**'):
                #if view == "logarithmic":
                #    st.write('**:red[Attention, logarithmic is not fitted for the diminishing returns]**')
                #else: pass

                df.reset_index(drop = False, inplace = True)
                df = df[['timestamp', 'Close', 'metric']]
                df.columns = ['ds', 'y', 'metric']

                if view == "logarithmic":
                    df['y']= np.log(df.y)
                else: pass

                if time == 'daily':
                    pass
                
                elif time == 'weekly':
                    df = df[::7]

            
                # Initialize Prophet model with regressor and fit to data

                #model = Prophet()
                model = Prophet(
                    daily_seasonality= True,
                    weekly_seasonality= True,
                    yearly_seasonality = True
                )

                if time == 'daily':
                    model.add_seasonality(name='4yearly', period=1461, fourier_order=10)
                elif time == 'weekly':
                    model.add_seasonality(name='4yearly', period=1461/7, fourier_order =8)


                model.add_regressor('metric')
                model.fit(df)


                # Make predictions and plot the results
                if time == 'daily':
                    days = 90
                    future_dates = model.make_future_dataframe(periods = days, freq='D')
                    st.subheader(f':blue[Prediction at {days} days: ]')

                elif time == 'weekly':
                    weeks = 45
                    future_dates = model.make_future_dataframe(periods = weeks, freq='W')
                    st.subheader(f':blue[Prediction at {weeks} weeks: ]')




                future_dates['metric'] = df['metric']
                prediction = model.predict(future_dates.dropna())

                fig = plot_plotly(model, prediction)

                st.plotly_chart(go.Figure(fig))
            
                fig2 = plot_components_plotly(model, prediction)
                
                st.subheader(':blue[:Historical seasonalities detected by the model: ]')
                st.plotly_chart(fig2)

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


        if type_balance == 'Adresses Count':
            st.header(f'You are looking at `{type_balance}` from the category `{onchain}`')
            st.write('You are looking at the number of adresses per cohort. \n\n'
                    'It is relevant to observe which actors are entering or leaving the market.'
                    "Are there the :blue[whales] which are often instutional actors which can have an impact on the Bitcoin's price. :fish: \n\n"
                    'Or, are there smaller actors which can be more emotional but who decentralise the network. :fishing_pole_and_fish:'
            )

            tab1, tab2= st.tabs(["Chart", "Prediction"])

            with tab1:

                days_to_plot = st.slider(
                    'Days to plot',
                    min_value=1,
                    max_value=len(df),
                    value=len(df)
                )
                df = df[-days_to_plot:]

                st.plotly_chart(balances_viz_addresses(df, checkbox_val, checkbox_val_metric, ma,type_balance, metrics, checkbox_zscore),
                    use_container_width=True)     


            with tab2:
                st.title(':blue[Prediction based on machine learning]')
                st.subheader(f'Based on historical price and its correlation with {metrics}')
                st.subheader(':red[Attention, only 2000 days of data !]')
                st.markdown("Note that the output can take time.")
                

                
                c1, c2 = st.columns(2)
                with c1:
                    view = st.radio(
                        ":blue[Select view] :chart_with_upwards_trend:",
                        key="linear",
                        options=["linear", "logarithmic"],
                    )

                with c2:
                    time = st.radio(
                        ":blue[Select timeframe] :hourglass_flowing_sand:",
                        key="weekly",
                        options=["daily", "weekly"],
                    )

                if st.button('**Make Prediction**'):
                    df.reset_index(drop = False, inplace = True)
                    dic=  {'<0.01':'less_than_0_01', '0.01 - 0.1': 'f_0_01_to_0_1' ,'0.1 - 1': 'f_0_1_to_1', '1 - 10' : 'f_1_to_10', '10 - 100' : 'f_10_to_100','100 - 1000' : 'f_100_to_1000', '1k - 10k' : 'f_1000_to_10000','10k+' : 'f_10000_to_100000', 'all': 'all'}

                    ind = 'addressesCount_' + dic[metrics]
                    df['metric'] = df[ind]

                    df = df[['date','Close','metric']]
                    df.columns = ['ds','y','metric']
                    
                    if view == "logarithmic":
                        df['y']= np.log(df.y)
                    else: pass

                    if time == 'daily':
                        pass
                    
                    elif time == 'weekly':
                        df = df.iloc[::7]

                
                    # Initialize Prophet model with regressor and fit to data

                    model = Prophet()

                    if time == 'daily':
                        model.add_seasonality(name='4yearly', period=1461, fourier_order=10)
                    elif time == 'weekly':
                        model.add_seasonality(name='4yearly', period=1461/7, fourier_order =8)


                    model.add_regressor('metric')
                    model.fit(df)


                    # Make predictions and plot the results
                    if time == 'daily':
                        days = 90
                        future_dates = model.make_future_dataframe(periods = days, freq='D')
                        st.subheader(f':blue[Prediction at {days} days: ]')

                    elif time == 'weekly':
                        weeks = 45
                        future_dates = model.make_future_dataframe(periods = weeks, freq='W')
                        st.subheader(f':blue[Prediction at {weeks} weeks: ]')


                    
                    future_dates['metric'] = df['metric']
                    

                    prediction = model.predict(future_dates.dropna())

                    fig = plot_plotly(model, prediction)

                    st.plotly_chart(go.Figure(fig))
                
                    fig2 = plot_components_plotly(model, prediction)
                    
                    st.subheader(':blue[:Historical seasonalities detected by the model: ]')
                    st.plotly_chart(fig2)
                

            
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

            tab1, tab2= st.tabs(["Chart", "Prediction"])

            with tab1:
                days_to_plot = st.slider(
                    'Days to plot',
                    min_value=1,
                    max_value=len(df),
                    value=len(df)
                )
                df = df[-days_to_plot:]

                st.plotly_chart(balances_viz_addresses(df, checkbox_val, checkbox_val_metric, ma, type_balance,metrics, checkbox_zscore),
                    use_container_width=True)    


            with tab2:
                st.title(':blue[Prediction based on machine learning]')
                st.subheader(f'Based on historical price and its correlation with {metrics}')
                st.subheader(':red[Attention, only 2000 days of data !]')
                st.markdown("Note that the output can take time.")
                

                
                c1, c2 = st.columns(2)
                with c1:
                    view = st.radio(
                        ":blue[Select view] :chart_with_upwards_trend:",
                        key="linear",
                        options=["linear", "logarithmic"],
                    )

                with c2:
                    time = st.radio(
                        ":blue[Select timeframe] :hourglass_flowing_sand:",
                        key="weekly",
                        options=["daily", "weekly"],
                    )

                if st.button('**Make Prediction**'):
                    df.reset_index(drop = False, inplace = True)
                    dic=  {'<0.01':'less_than_0_01', '0.01 - 0.1': 'f_0_01_to_0_1' ,'0.1 - 1': 'f_0_1_to_1', '1 - 10' : 'f_1_to_10', '10 - 100' : 'f_10_to_100','100 - 1000' : 'f_100_to_1000', '1k - 10k' : 'f_1000_to_10000','10k+' : 'f_10000_to_100000', 'all': 'all'}

                    ind = 'addressesCount_' + dic[metrics]
                    df['metric'] = df[ind]

                    df = df[['date','Close','metric']]
                    df.columns = ['ds','y','metric']
                    
                    if view == "logarithmic":
                        df['y']= np.log(df.y)
                    else: pass

                    if time == 'daily':
                        pass
                    
                    elif time == 'weekly':
                        df = df.iloc[::7]

                
                    # Initialize Prophet model with regressor and fit to data

                    model = Prophet()

                    if time == 'daily':
                        model.add_seasonality(name='4yearly', period=1461, fourier_order=10)
                    elif time == 'weekly':
                        model.add_seasonality(name='4yearly', period=1461/7, fourier_order =8)


                    model.add_regressor('metric')
                    model.fit(df)


                    # Make predictions and plot the results
                    if time == 'daily':
                        days = 90
                        future_dates = model.make_future_dataframe(periods = days, freq='D')
                        st.subheader(f':blue[Prediction at {days} days: ]')

                    elif time == 'weekly':
                        weeks = 45
                        future_dates = model.make_future_dataframe(periods = weeks, freq='W')
                        st.subheader(f':blue[Prediction at {weeks} weeks: ]')


                    
                    future_dates['metric'] = df['metric']
                    

                    prediction = model.predict(future_dates.dropna())

                    fig = plot_plotly(model, prediction)

                    st.plotly_chart(go.Figure(fig))
                
                    fig2 = plot_components_plotly(model, prediction)
                    
                    st.subheader(':blue[:Historical seasonalities detected by the model: ]')
                    st.plotly_chart(fig2)
                


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


        tab1, tab2= st.tabs(["Chart", "Prediction"])

        with tab1:
            days_to_plot = st.slider(
                'Days to plot',
                min_value=1,
                max_value=len(df),
                value=len(df)
            )
            df = df[-days_to_plot:]
            st.plotly_chart(viz_with_indicator(df, checkbox_val, checkbox_val_metric, ma, metrics,checkbox_zscore ),
                            use_container_width=True)   


        with tab2:
            st.title(':blue[Prediction based on machine learning]')
            st.subheader(f'Based on historical price and its correlation with {metrics}')
            st.markdown("Note that the output can take time.")
            

            
            c1, c2 = st.columns(2)
            with c1:
                view = st.radio(
                    ":blue[Select view] :chart_with_upwards_trend:",
                    key="linear",
                    options=["linear", "logarithmic"],
                )

            with c2:
                time = st.radio(
                    ":blue[Select timeframe] :hourglass_flowing_sand:",
                    key="weekly",
                    options=["daily", "weekly"],
                )

            if st.button('**Make Prediction**'):
                df.reset_index(drop = False, inplace = True)
                df = df[['timestamp', 'Close', 'metric']]
                df.columns = ['ds', 'y', 'metric']
                
                if view == "logarithmic":
                    df['y']= np.log(df.y)
                else: pass

                if time == 'daily':
                    pass
                
                elif time == 'weekly':
                    df = df.iloc[::7]

            
                # Initialize Prophet model with regressor and fit to data

                model = Prophet()

                if time == 'daily':
                    model.add_seasonality(name='4yearly', period=1461, fourier_order=10)
                elif time == 'weekly':
                    model.add_seasonality(name='4yearly', period=1461/7, fourier_order =8)


                model.add_regressor('metric')
                model.fit(df)


                # Make predictions and plot the results
                if time == 'daily':
                    days = 90
                    future_dates = model.make_future_dataframe(periods = days, freq='D')
                    st.subheader(f':blue[Prediction at {days} days: ]')

                elif time == 'weekly':
                    weeks = 45
                    future_dates = model.make_future_dataframe(periods = weeks, freq='W')
                    st.subheader(f':blue[Prediction at {weeks} weeks: ]')


                
                future_dates['metric'] = df['metric']
                

                prediction = model.predict(future_dates.dropna())

                fig = plot_plotly(model, prediction)

                st.plotly_chart(go.Figure(fig))
            
                fig2 = plot_components_plotly(model, prediction)
                
                st.subheader(':blue[:Historical seasonalities detected by the model: ]')
                st.plotly_chart(fig2)
                

    else:
        df = on_chain_merge(onchain, metrics)
        st.header(f'You are looking at `{metrics}` from the category `{onchain}`')

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


        tab1, tab2= st.tabs(["Chart", "Prediction"])

        with tab1:
            days_to_plot = st.slider(
                'Days to plot',
                min_value=1,
                max_value=len(df),
                value=len(df)
            )
            df = df[-days_to_plot:]

            st.plotly_chart(on_chain_viz_zscore(df, checkbox_val, checkbox_val_metric, ma, onchain, metrics,checkbox_zscore ),
                            use_container_width=True)   


        with tab2:
            st.title(':blue[Prediction based on machine learning]')
            st.subheader(f'Based on historical price and its correlation with {metrics}')
            st.markdown("Note that the output can take time.")
            

            
            c1, c2 = st.columns(2)
            with c1:
                view = st.radio(
                    ":blue[Select view] :chart_with_upwards_trend:",
                    key="linear",
                    options=["linear", "logarithmic"],
                )

            with c2:
                time = st.radio(
                    ":blue[Select timeframe] :hourglass_flowing_sand:",
                    key="weekly",
                    options=["daily", "weekly"],
                )

            if st.button('**Make Prediction**'):
                df.reset_index(drop = False, inplace = True)
                df = df[['timestamp', 'Close', 'metric']]
                df.columns = ['ds', 'y', 'metric']
                
                if view == "logarithmic":
                    df['y']= np.log(df.y)
                else: pass

                if time == 'daily':
                    pass
                
                elif time == 'weekly':
                    df = df.iloc[::7]

            
                # Initialize Prophet model with regressor and fit to data

                model = Prophet()

                if time == 'daily':
                    model.add_seasonality(name='4yearly', period=1461, fourier_order=10)
                elif time == 'weekly':
                    model.add_seasonality(name='4yearly', period=1461/7, fourier_order =8)


                model.add_regressor('metric')
                model.fit(df)


                # Make predictions and plot the results
                if time == 'daily':
                    days = 90
                    future_dates = model.make_future_dataframe(periods = days, freq='D')
                    st.subheader(f':blue[Prediction at {days} days: ]')

                elif time == 'weekly':
                    weeks = 45
                    future_dates = model.make_future_dataframe(periods = weeks, freq='W')
                    st.subheader(f':blue[Prediction at {weeks} weeks: ]')


                
                future_dates['metric'] = df['metric']
                

                prediction = model.predict(future_dates.dropna())

                fig = plot_plotly(model, prediction)

                st.plotly_chart(go.Figure(fig))
            
                fig2 = plot_components_plotly(model, prediction)
                
                st.subheader(':blue[:Historical seasonalities detected by the model: ]')
                st.plotly_chart(fig2)
                


