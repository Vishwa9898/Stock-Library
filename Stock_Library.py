# # Importing all the libraries
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import date, datetime, timedelta
from plotly.subplots import make_subplots
from sklearn.svm import SVR

# Write the main heading of dashboard
st.write(
    """
    # Stock Market Analysis
    **Visually** show data
    """
)
st.sidebar.header("User Preferences") # Sidebar header
Ticker_Symbol = pd.read_csv("All_tickers.csv") # read data from csv file
Ticker_Symbol_List = list(Ticker_Symbol["Ticker"])  # List of all the ticker Symbols

# here we are using datetime library to get current time
Today_date = date.today()
Current_time = Today_date.strftime("%Y-%m-%d")

# FUNCTIONSs

def get_timelines(): # this function returns the selected timelines
    button2 = st.sidebar.radio("", ("Fixed Timeline", "Manual Timeline"))
    return button2

def get_required_input(feature): # this function collect all the required input from users
    if feature == "Fixed Timeline":
        stock_symbol = st.sidebar.selectbox("Select your company symbol ", Ticker_Symbol_List)
        return stock_symbol
    else:
        start_date = st.sidebar.text_input("Start_Date", "2021-01-02")
        end_date = st.sidebar.text_input("End_Date", "2021-02-05")
        stock_symbol = st.sidebar.selectbox("Select your name ", Ticker_Symbol_List)
        return start_date, end_date, stock_symbol

def get_company_name(symbol): # This function fetch the company name from csv file
    co_data = Ticker_Symbol.loc[Ticker_Symbol["Ticker"]==symbol]
    return co_data["Co_name"].item()

def get_time(): # This function creates the buttons for different time of periods
    c1, c2, c3, c4, c5 = st.sidebar.beta_columns(5)
    c1 = c1.button("10D")
    c2 = c2.button("1M")
    c3 = c3.button("6M")
    c4 = c4.button("1Y")
    c5 = c5.button("5Y")
    return c1,c2,c3,c4,c5

def manual_data(symblo, startdate, enddate): # get data fro manual timeline
    start = pd.to_datetime(startdate)
    end = pd.to_datetime(enddate)
    if symblo in Ticker_Symbol_List:
        data = yf.download(symblo, start=start, end=end)
        data = data.reset_index()
    return data

def Fix_data (T_symbol,c1,c2,c3,c4,c5): # get data for fix timelines
    if c1==True:
        data = yf.download(T_symbol, start="2010-01-02", end=Current_time)
        data = data.reset_index()
        data = data[-10:]
        data = data.reset_index()
        return data
    elif c2==True:
        data = yf.download(T_symbol, start="2010-01-02", end=Current_time)
        data = data.reset_index()
        data = data[-30:]
        data = data.reset_index()
        return data
    elif c3==True:
        data = yf.download(T_symbol, start="2010-01-02", end=Current_time)
        data = data.reset_index()
        data = data[-180:]
        data = data.reset_index()
        return data
    elif c4==True:
        data = yf.download(T_symbol, start="2010-01-02", end=Current_time)
        data = data.reset_index()
        data = data[-365:]
        data = data.reset_index()
        return data
    elif c5==True:
        data = yf.download(T_symbol, start="2010-01-02", end=Current_time)
        data = data.reset_index()
        data = data[-1825:]
        data = data.reset_index()
        return data
    else:
        data = yf.download(T_symbol, start="2010-01-02", end=Current_time)
        data = data.reset_index()
        data = data[-1500:]
        data = data.reset_index()
        return data

def Number_of_MA(): # get the number of moving averages user wants to add
    button = st.sidebar.radio("Number of M.A.",(1,2))
    return button

def Get_MA_Value(button): # with the help of this function, user can set the value for moving averages
    if button == 1: # one moving avg.
        ma = st.sidebar.text_input("Enter Moving Average ", 20)
        ma2 = "22"
        return ma, ma2
    elif button == 2: # two moving avg.
        ma = st.sidebar.text_input("Enter First Moving Average", 20)
        ma2 = st.sidebar.text_input("Enter Second Moving Average", 80)
        return ma,ma2

def Single_MA_data (ma, data): # this function returns the single moving averages.
    ma = int(ma)
    MA_data = pd.DataFrame()
    MA_data["Moving_Avg"]= data["Adj Close"].rolling(window=ma).mean()
    return MA_data

def Single_MA_visualization(data, ma_data): # this function visualize the single moving avg.
    fig = px.line(data,
                  x="Date",
                  y="Adj Close",
                  template="none",
                  height=500,
                  width=800)
    fig.add_scatter(
        x=data["Date"],
        y=ma_data["Moving_Avg"],
        mode="lines")
    st.plotly_chart(fig)

def Two_MA_data(m1, m2, data): # this function returns the double moving averages.
    m1 = int(m1)
    m2 = int(m2)
    MA1_data = pd.DataFrame()
    MA1_data["Moving_Avg"] = data["Adj Close"].rolling(window=m1).mean()
    MA1_data["Moving_Avg2"]= data["Adj Close"].rolling(window=m2).mean()
    return  MA1_data

def buy_sell(data):
    sigPriceBuy = []
    sigPriceSell = []
    diff = []
    skip = 0
    buy_price = 0
    support = 0
    flag = 0
    for i in range(len(data)):
        if data["Moving_Avg"][i] > data["Moving_Avg2"][i] and skip == 0:
            if flag != 1 and skip == 0:
                sigPriceBuy.append(data["Adj Close"][i])
                buy_price = data["Adj Close"][i]
                support = buy_price - (buy_price * 0.30)
                print(f"buying price{buy_price}")
                print(f"buysupport{support}")
                resistance = buy_price + (buy_price * 0.7)
                sigPriceSell.append(np.nan)
                flag = 1
                skip = 1
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
        elif data["Adj Close"][i] < support:
            pp = data["Adj Close"][i]
            print(f"closing_price{pp}")
            diff.append(data["Adj Close"][i] - buy_price)
            sigPriceBuy.append(np.nan)
            sigPriceSell.append(data["Adj Close"][i])
            skip = 1
            flag = 0
            support = 0

        elif data["Moving_Avg"][i] < data["Moving_Avg2"][i]:
            skip = 0
            if flag != 0:
                print(f"sell{support}")
                if buy_price != 0:
                    diff.append(data["Adj Close"][i] - buy_price)
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(data["Adj Close"][i])
                skip = 0
                flag = 0
                support = 0
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
        else:
            sigPriceBuy.append(np.nan)
            sigPriceSell.append(np.nan)
    return (sigPriceBuy, sigPriceSell)

def Double_MA_visualization(data, m1_data): # this function visualize the single moving avg.
    fig = px.line(data,
                  x="Date",
                  y="Adj Close",
                  template="none")
    fig.add_scatter(
        x=data["Date"],
        y=m1_data["Moving_Avg"],
        mode="lines")
    fig.add_scatter(
        x=data["Date"],
        y=m1_data["Moving_Avg2"],
        mode="lines")
    fig.add_scatter(x=data.Date,
                    y=m1_data["Buy_Signal_Price"],
                    name="Buy Signal",
                    mode="markers")
    fig.add_scatter(x=data.Date,
                    y=m1_data["Sell_Signal_Price"],
                    name="Sell Signal",
                    mode="markers",
                    )
    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=3,
                                            color='white')),
                      selector=dict(mode='markers'))
    st.plotly_chart(fig)
# EXECUTING THE FUNCTIONALITY

Feature = get_timelines() # this variable stores the value of fix timelines or manual timelines

if Feature =="Fixed Timeline": # if user select fix timeline
    T_symbol = get_required_input(Feature)  # Store Ticker symbol
    company_name = get_company_name(T_symbol.upper()) # Store name of company
    st.sidebar.subheader("""Time Frame""")
    c1, c2, c3, c4, c5 = get_time() # get the timeline
    company_data = Fix_data(T_symbol,c1,c2,c3,c4,c5)

elif Feature=="Manual Timeline": # if user select manual timeline
    S_date, e_date, T_symbol = get_required_input(Feature) # collect start and end date and Ticker symbol
    company_name = get_company_name(T_symbol.upper())
    company_data = manual_data(T_symbol, S_date, e_date)

year = [] # creating empty list
for date in company_data["Date"]:
    y = date.year # fetch year from timestamp
    year.append(y)
company_data["Year"] = year # add year column in company_data dataframe

# VISUALIZATIONS

st.header(company_name + " Data")
st.write(company_data) # show the data of company
st.header(company_name + " Data Statistic")
st.write(
    company_data.describe() # Statistical analysis of company data
)

# visualize the closing price of company
st.header(company_name+" Close Price by each years\n")
fig1 = px.line(company_data,
               x = company_data.Date,
               y= company_data.Close,
               template="none",
               color="Year",
               height=500,
               width=800)
st.plotly_chart(fig1)

# this bar chart represents the day to day volume of company
st.header(company_name+" Volume Bars\n")
fig_volume_bar = px.bar(company_data,
                        x="Date",
                        y = "Volume",
                        height = 500,
                        width = 800,
                        color="Year",
                        template = "none"
                        )
st.plotly_chart(fig_volume_bar)

# candlestick chart for data analysis
st.header(company_name + " Candles for previous prices")
fig = go.Figure(data=[go.Candlestick(x=company_data['Date'],
                                   open=company_data['Open'],
high=company_data['High'],
low=company_data['Low'],
close=company_data['Close'])])
fig.update_layout(height = 500, width = 800,template = "none")
st.plotly_chart(fig)

# showcase the everyday changes in closing price of company
st.header(company_name+" Everyday changes in stock price ")
fig_EDC = px.bar(company_data,
                 x = "Date",
                 y = company_data.Open - company_data.Close,
                 color="Year",
                 width = 800,
                 height=500,
                 template="none")
st.plotly_chart(fig_EDC)

# Stock Indicators
st.sidebar.header("""Stock Indicator""") # create the sidebar header
st.header(company_name+"'s closing price with moving averages ")
button = Number_of_MA() # store the return value of function
ma1, ma2 = Get_MA_Value(button) # store the value of moving averages.
if button == 1:
    Single_MA_Data = Single_MA_data(ma1, company_data)
    Single_MA_visualization(company_data,Single_MA_Data)

elif button == 2:
    Double_MA_Data= Two_MA_data(ma1, ma2, company_data)
    Double_MA_Data["Adj Close"] = company_data["Adj Close"]
    Double_MA_Data = Double_MA_Data.reset_index()
    buy_sell = buy_sell(Double_MA_Data)
    Double_MA_Data["Buy_Signal_Price"] = buy_sell[0]
    Double_MA_Data["Sell_Signal_Price"] = buy_sell[1]
    Double_MA_visualization(company_data, Double_MA_Data)




# part 2

def get_data_for_strategies():
    Today = datetime.now()
    current_date = Today.strftime("%d-%m-%y")
    current_date = pd.to_datetime(current_date)
    delta_date = current_date - timedelta(days=500)
    data = yf.download(T_symbol, start="2018-10-01", end=current_date)
    company_name = get_company_name(T_symbol)
    data = data[-339:]
    data = data.reset_index()
    return data, company_name
company_data, company_name = get_data_for_strategies()
# st.write(company_data.shape)
# st.write(company_name)
# st.write(company_data)
def get_MACD (Price, SlowMA, FastMA, Smooth):
    Slowline = Price.ewm(span = SlowMA, adjust = False).mean()
    Fastline = Price.ewm(span = FastMA, adjust = False).mean()
    macd = pd.DataFrame(Fastline - Slowline).rename(columns= {"Close":"MACD"})
    signal = pd.DataFrame(macd.ewm(span = Smooth, adjust = False).mean()).rename(columns = {'MACD':'Signal'})
    hist = pd.DataFrame(macd['MACD'] - signal['Signal']).rename(columns = {0:'Hist'})
    frames =  [macd, signal, hist]
    df = pd.concat(frames, join = 'inner', axis = 1)
    return df
macd = get_MACD(company_data["Close"], 26, 12, 9)
macd = macd.reset_index()
macd["Date"] = company_data["Date"]

def plot_macd(company_data, macd):
    fig = make_subplots(vertical_spacing = 0.02, rows=3, cols=1, row_heights=[0.6, 0.2, 0.2])
    fig.add_trace(go.Candlestick(x = company_data["Date"],
                                open = company_data["Open"],
                                high = company_data.High,
                                low = company_data.Low,
                                close = company_data.Close))
    fig.add_trace(go.Scatter(x=company_data['Date'], y = macd.MACD),row=2, col=1)
    fig.add_trace(go.Scatter(x=company_data['Date'], y = macd.Signal), row=2, col=1)
    fig.add_trace(go.Bar(x=company_data['Date'], y = macd.Hist), row=2, col=1)
    fig.add_trace(go.Bar(x=company_data['Date'], y = company_data.Volume), row=3, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False,
                      xaxis=dict(zerolinecolor='black', showticklabels=False),
                      xaxis2=dict(showticklabels=False))
    fig.update_layout(template = "none")

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=False)
    st.plotly_chart(fig)
st.header(company_name + " MACD Chart")
plot_macd(company_data, macd)

# if st.sidebar.button("Get strategies"):
#     get_data_for_strategies()

def get_RSI(Price, MA):
    difference = Price.diff()
    up = []
    down = []
    for i in range(len(difference)):
        if difference[i]<0:
            up.append(0)
            down.append(difference[i])
        else:
            up.append(difference[i])
            down.append(0)
    up_series = pd.Series(up)
    down_series = pd.Series(down).abs()
    up_ewm = up_series.ewm(com = MA-1, adjust = False).mean()
    down_ewm = down_series.ewm(com =MA-1, adjust = False).mean()
    rs = up_ewm/down_ewm
    rsi = 100 - (100 / (1 + rs))
    rsi_df = pd.DataFrame(rsi).rename(columns = {0:'rsi'}).set_index(Price.index)
    rsi_df = rsi_df.dropna()
    return rsi_df[3:]
rsi = get_RSI(company_data.Close, 14)

def plot_rsi(rsi):
    fig2 = px.line(rsi ,y = "rsi")
    fig2.update_layout(yaxis_range = [10,90])
    fig2.add_hline(y = 50, opacity = 1, annotation_text = "Base line",
                 annotation_position = "bottom right",
                   line_color = "white", line_width = 1)
    fig2.add_hline(y = 70, opacity = 1,line_color = "white", line_width = 1)
    fig2.add_hline(y = 30, opacity = 1,line_color = "white", line_width = 1)
    fig2.add_hrect(y0 = 70, y1 = 100, annotation_text="over bought", annotation_position="top left",
                  fillcolor="red", opacity=0.25, line_width=0 )
    fig2.add_hrect(y0 = 0, y1 = 30, annotation_text="over sold", annotation_position="top left",
                  fillcolor="green", opacity=0.25, line_width=0 )
    fig2.update_xaxes(showgrid = False)
    fig2.update_yaxes(showgrid = False)
    st.plotly_chart(fig2)
st.header(company_name + " RSI Chart")
plot_rsi(rsi)

def get_atr(data):
    high_low = data["High"] - data["Low"]
    high_close = np.abs(data["High"] - data["Close"].shift())
    low_close = np.abs(data["Low"] - data["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_ranges = np.max(ranges, axis=1)
    atr = true_ranges.rolling(14).sum() / 14
    atr_data = pd.DataFrame()
    atr_data["Date"] = data["Date"]
    atr_data["Atr"] = atr
    return  atr_data

atr_data = get_atr(company_data)

def plot_atr(data, atr_data):
    fig_atr = make_subplots(vertical_spacing=0.02, rows=2, cols=1, row_heights=[0.5, 0.5])
    fig_atr.add_trace(go.Scatter(x=atr_data['Date'], y=atr_data["Atr"]), row=1, col=1)
    fig_atr.add_trace(go.Scatter(x=data['Date'], y=data["Close"]), row=2, col=1)
    fig_atr.update_layout(xaxis_rangeslider_visible=False,
                          xaxis=dict(zerolinecolor='black', showticklabels=False),
                          xaxis2=dict(showticklabels=False))
    fig_atr.update_layout(template = "none")
    st.plotly_chart(fig_atr)
st.header(company_name + " Average True Range chart")
plot_atr(company_data, atr_data)
def get_svr(data):
    data_except_last = data.head(len(data)-1)

    days = list()
    adj_close = list()
    df_days = data_except_last.loc[:, "Date"]
    df_adjclose = data_except_last.loc[:, "Adj Close"]

    for price in df_adjclose:
        adj_close.append(float(price))
    days = np.arange(1, 339)
    dayl = list()
    for i in days:
        dayl.append([int(i)])
    rbf_svr = SVR(kernel='rbf', C=1000.0, gamma=0.15)
    rbf_svr.fit(dayl, adj_close)
    dayl2 = [[339], [340], [341], [342], [343], [344], [345], [346], [347], [348], [349], [350]]
    return rbf_svr.predict(dayl2)
predicted_data = pd.DataFrame()
predicted_data["predicted_val"] = get_svr(company_data)
st.header(company_name + " Predicted values of ten days from SVR Algorithm")
st.write(predicted_data)


ex_data = company_data.tail(14).mean()
# st.write(ex_data)
examine_pre_data = predicted_data.loc[:3,:].mean()
# st.write(predicted_data.loc[:2,:])
examine_macd = macd.tail(1)
examine_rsi = rsi.tail(1)
st.header("Stock's selling and buying suggestions based on our study.")
val = 0

if examine_macd["MACD"].item() > examine_macd["Signal"].item():
    if examine_macd["Hist"].item()> 0:
        val = val + 25
    else:
        val = val + 20
else:
    st.write("Good time to sell stock as par MACD")

if examine_rsi["rsi"].item()>50:
    val = val + 30
else:
    st.write("Good time to sell stock as par RSI")
examine_dma = Double_MA_Data.tail(1)
# st.write(examine_dma)
if examine_dma["Moving_Avg"].item() > examine_dma["Moving_Avg2"].item():
    val = val + 20
else:
    st.write("Good time to sell stock as par Double moving Avg")
if examine_dma["Adj Close"].item() < examine_pre_data.item():
    val = val + 25
else:
    if ex_data["Adj Close"].item() < examine_dma["Adj Close"].item():
        val = val + 10
    else:
        pass
    st.write("Good time to sell stock as par SVR")


fig_guage = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = val,
    mode = "gauge+number+delta",
    title = {'text': "Possibility of buying the stock"},
    delta = {'reference': 85},
    gauge = {'axis': {'range': [None, 100]},
             'steps' : [
                 {'range': [0, 55], 'color': "lightgray"},
                 {'range': [55, 80], 'color': "gray"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 85}}))

st.plotly_chart(fig_guage)

