#Reference: https://blog.streamlit.io/how-to-build-a-real-time-live-dashboard-with-streamlit/

import numpy as np  
import pandas as pd  
import plotly.express as px  # interactive charts
import streamlit as st  #  data web app development
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
import yfinance as yf
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Daily SPY Open Price Predictor",
                 page_icon="🧊",
                 #layout="wide",
                 #initial_sidebar_state="expanded",
                 menu_items={
                     'Get Help': 'https://github.com/valenciavene/spy_daily_predictor',
                     'Report a bug': "https://github.com/valenciavene/spy_daily_predictor/issues",
                     'About': "This app predicts daily SPY price using data from other markets"
                 }
)

st.subheader('Daily SPY Open Price Predictor')

start =  datetime(2017, 1, 1) + timedelta(days=1) #yyyy,mm,dd
end = datetime.today() + timedelta(days=1)   #yfinance time varies from markets timezone, add 1 day to get latest 

alldf =pd.DataFrame()
tickers = ['^AORD','^HSI','^N225','SPY','^GSPC','^IXIC','^DJI','^VIX','^FCHI','^FTSE','^GDAXI']
names = ['aord','hsi','nikkei','spy','sp500','nasdaq','dji','vix','cac40','FTSE 100','daxi']

for ticker, name in zip(tickers, names):
    print(name, ticker)
    df = yf.download(ticker, start=start, end=end) 
    if name in names[4:]:
        alldf[name] = df['Open']-df['Open'].shift(1)
    elif name in names[:3]:
        alldf[name] = df['Close']-df['Open']
    elif name == 'spy':
        alldf[name]=df['Open'].shift(-1)-df['Open']
        alldf['spy_lag1']=alldf[name].shift(1)
        alldf['Price']=df['Open']   

alldf = alldf.fillna(method='ffill')
alldf = alldf.dropna()

features = ['spy_lag1','aord','hsi','nikkei','sp500','nasdaq','dji','vix','cac40','FTSE 100','daxi' ]

# Normalisation
for name in features:
    alldf[name] = (alldf[name] - alldf[name].mean()) / alldf[name].std()
    
X = alldf[features].iloc[:-1]  #When training model, can't use upto today's data 
y = alldf['spy'].iloc[:-1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True, random_state=42)

model = xgb.XGBRegressor(learning_rate=0.05, random_state=5)

# Fit the model with the training data
model.fit(X_train, y_train)

# predict the target on the test dataset
y_predict = model.predict(X_test)

RMSE = mean_squared_error(y_test, y_predict, squared=False)
print('\nRMSE on test dataset: %.4f' % RMSE)

split = int(alldf.shape[0]*3/10)

Train = X.iloc[:-split, :]
Test = X.iloc[-split:, :]

Train_y = y.iloc[:-split]
Test_y = y.iloc[-split:]

Train['PredictedY'] = model.predict(Train)
Test['PredictedY'] = model.predict(Test)

# RMSE - Root Mean Squared Error, Adjusted R^2
def adjustedMetric2(data, model, model_k, yname):
    data['yhat'] = model.predict(data.loc[:, data.columns!=yname])
    SST = ((data[yname] - data[yname].mean())**2).sum()
    SSR = ((data['yhat'] - data[yname].mean())**2).sum()
    SSE = ((data[yname] - data['yhat'])**2).sum()
    r2 = SSR/SST
    adjustR2 = 1 - (1-r2)*(data.shape[0] - 1)/(data.shape[0] -model_k -1)
    RMSE = (SSE/(data.shape[0] -model_k -1))**0.5
    return adjustR2, RMSE

def assessTable2(test, train, model, model_k, yname):
    r2test, RMSEtest = adjustedMetric2(test, model, model_k, yname)
    r2train, RMSEtrain = adjustedMetric2(train, model, model_k, yname)
    assessment = pd.DataFrame(index=['R2', 'RMSE'], columns=['Train', 'Test'])
    assessment['Train'] = [r2train, RMSEtrain]
    assessment['Test'] = [r2test, RMSEtest]
    return assessment

Train_metric = alldf.iloc[:-split, :]
Test_metric = alldf.iloc[-split:, :]

featuresplus = ['spy','spy_lag1','aord','hsi','nikkei','sp500','nasdaq','dji','vix','cac40','FTSE 100','daxi'  ]
metric_df = assessTable2(Test_metric[featuresplus], Train_metric[featuresplus], model, len(features), 'spy')
print('metric_df')
print(metric_df)

job_filter = st.selectbox("Select Test Period", ['1 Year', '1 Month', '1 Week', '1 Day'])

if job_filter == '1 Year':
    test_period = datetime.today() - pd.DateOffset(years=1)
elif job_filter == '1 Month':
    test_period = datetime.today() - pd.DateOffset(months = 1)
elif job_filter == '1 Week':
    test_period = datetime.today() - pd.DateOffset(weeks = 1)
elif job_filter == '1 Day':
    test_period = datetime.today() - pd.DateOffset(days = 1)   

chart_date = test_period.date() 
chart_df = alldf.loc[chart_date:, :]

chart_df['predict_spy'] = model.predict(chart_df.loc[:,features])
chart_df['predict_spy_price'] = chart_df['predict_spy'] + chart_df['Price']
chart_df['predict_price_error'] =  chart_df['Price'] - chart_df['predict_spy_price']

placeholder = st.empty()

with placeholder.container():
        
    col1, col2 = st.columns(2)
    
    col1.metric(
        label= "Date: ",
        value=pd.to_datetime(chart_df['predict_spy_price'].iloc[-1:].index[0]).strftime("%Y-%m-%d")
    )
    
    col2.metric(
        label= "Predicted SPY Open Price: ",
        value= round(float(chart_df['predict_spy_price'].iloc[-1:].values),2)
    )
    
    st.markdown("### SPY PRICE (ACTUAL VS PREDICT)")

    fig = px.line(chart_df, x=chart_df.index, 
                  y=["Price","predict_spy_price"], 
                  labels={"Price":'Actual SPY Price',"predict_spy_price":"Predict SPY Price"})
    series_names=['Actual SPY Price','Predict SPY Price']
    for idx, name in enumerate(series_names):
        fig.data[idx].name = name
    fig.update_traces(
        hovertemplate="<br>".join([
                                    "Date: %{x}",
                                    "Price: %{y:.2f}",
        ])
    )
    fig.update_yaxes(title_text="SPY Price", tickprefix="$")
    fig.update_layout(legend_title="")
    st.write(fig)
    
    st.markdown("### SPY PRICE PREDICTION ERROR (Actual - Predict)")
    fig2 = px.line(chart_df, x=chart_df.index, y='predict_price_error')
    fig2.add_hline(y=0)
    fig2.update_yaxes(title_text="SPY Price Error", tickprefix="$")
    fig2.update_traces(
        hovertemplate="<br>".join([
            "Date: %{x}",
            "Error: $%{y:.2f}",
        ])
    )
    st.write(fig2)
    
    # create 4 columns
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    kpi1.metric(
        label="R2 Train⏳",
        value= round(metric_df.loc['R2','Train'],3),
    )
    
    kpi2.metric(
        label="RMSE Train💍",
        value=round(metric_df.loc['RMSE','Train'],3),
    )

    kpi3.metric(
        label="R2 Test⏳",
        value= round(metric_df.loc['R2','Test'],3),
        delta= round(metric_df.loc['R2','Test'] - metric_df.loc['R2','Train'],3),
    )
    
    kpi4.metric(
        label="RMSE Test💍",
        value=round(metric_df.loc['RMSE','Test'],3),
        delta=round(metric_df.loc['RMSE','Test'] - metric_df.loc['RMSE','Train'],3),
    )
    
st.write("Disclaimers: The information in this website is for information only, the author makes no representation or warranty or assurance as to its adequacy, completeness, accuracy or timeliness for any particular purpose. Opinions and estimates are subject to change without notice. The recommendation does not take into account the specific investment objectives, financial situation or particular needs of any particular person. Any past performance, projection, forecast or simulation of results is not necessarily indicative of the future or likely performance of any investment.")
