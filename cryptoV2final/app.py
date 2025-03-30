import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
import datetime as dt
import yfinance as yf
import pandas_ta as ta
from plotly.subplots import make_subplots
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title='CryptoPredict 2.0', page_icon=':chart_with_upwards_trend:')
st.title('CryptoCurrency Price Prediction')


stocks = [ 'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD', 'DOT-USD', 'DOGE-USD',
          'AVAX-USD', 'LTC-USD', 'MATIC-USD', 'SHIB-USD']



st.markdown('#')

with st.expander(""):
    col1, col2, col3 = st.columns([1, 1, 1])



    col2.markdown("###### CRYPTO CURRENCIES")
    col2.markdown("""
    | Cryptocurrency | Ticker Symbol |
    | --- | --- |
    | Bitcoin | BTC-USD |
    | Ethereum | ETH-USD |
    | Binance Coin | BNB-USD |
    | Solana | SOL-USD |
    | Cardano | ADA-USD |
    | XRP | XRP-USD |
    | Polkadot | DOT-USD |
    | Dogecoin | DOGE-USD |
    | Avalanche | AVAX-USD |
    | Litecoin | LTC-USD |
    | Polygon | MATIC-USD |
    | Shiba Inu | SHIB-USD |
    """)




user_input = st.selectbox('Enter Stock Ticker', stocks)

st.markdown('# ')

st.markdown('##### Select The Date Range For Technical Analysis')

START = st.date_input('START:', value=pd.to_datetime("2017-01-01"))
TODAY = st.date_input('END (Today):', value=pd.to_datetime("today"))

stock_info = yf.Ticker(user_input).fast_info

# stock_info.keys() for other properties you can explore


st.subheader(user_input)


def load_data(user_input):
    yf.pdr_override()
    daata = data.get_data_yahoo(user_input, start=START, end=TODAY)
    daata.reset_index(inplace=True)
    return daata


df = load_data(user_input)

# describing data

st.subheader('Data Range 2017-Today')
# df= df.reset_index()

st.write(df.tail(10))
st.write(df.describe())
# Force lowercase (optional)
df.columns = [x.lower() for x in df.columns]



st.subheader("Prediction of Stock Price")

# train test split
data_training = pd.DataFrame(df['close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['close'][int(len(df) * 0.70): int(len(df))])

st.write("training data: ", data_training.shape)
st.write("testing data: ", data_testing.shape)

# scaling of data using min max scaler (0,1)


scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

# Load model
model = load_model("lstm_model_2.h5")

# testing part
past_100_days = data_training.tail(30)

final_df = past_100_days.append(data_testing, ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1 / scaler[0]

y_predicted = y_predicted * scale_factor

y_test = y_test * scale_factor



st.subheader('Stock Price Prediction by Date')

df1 = df.reset_index()['close']
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

# datemax="24/06/2022"
datemax = dt.datetime.strftime(dt.datetime.now() - timedelta(1), "%d/%m/%Y")
datemax = dt.datetime.strptime(datemax, "%d/%m/%Y")
x_input = df1[:].reshape(1, -1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

date1 = st.date_input("Enter Date in this format yyyy-mm-dd")

result = st.button("Predict")
# st.write(result)
if result:
    from datetime import datetime

    my_time = datetime.min.time()
    date1 = datetime.combine(date1, my_time)
    # date1=str(date1)
    # date1=dt.datetime.pastime(time_str,"%Y-%m-%d")

    nDay = date1 - datemax
    nDay = nDay.days

    date_rng = pd.date_range(start=datemax, end=date1, freq='D')
    date_rng = date_rng[1:date_rng.size]
    lst_output = []
    n_steps = x_input.shape[1]
    i = 0

    while i <= nDay:

        if len(temp_input) > n_steps:
            # print(temp_input)
            x_input = np.array(temp_input[1:])
            print("{} day input {}".format(i, x_input))
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            # print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i, yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            # print(temp_input)
            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i = i + 1
    res = scaler.inverse_transform(lst_output)
    # output = res[nDay-1]

    output = res[nDay]

    st.write("*Predicted Price for Date :*", date1, "*is*", np.round(output[0], 2))
    st.success('The Price is {}'.format(np.round(output[0], 2)))

    # st.write("predicted price : ",output)

    predictions = res[res.size - nDay:res.size]
    print(predictions.shape)
    predictions = predictions.ravel()
    print(type(predictions))
    print(date_rng)
    print(predictions)
    print(date_rng.shape)


    @st.cache_data
    def convert_df(df):
        return df.to_csv().encode('utf-8')


    df = pd.DataFrame(data=date_rng)
    df['Predictions'] = predictions.tolist()
    df.columns = ['Date', 'Price']
    st.write(df)
    csv = convert_df(df)
    st.download_button(
        "Press to Download",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
    )
    # visualization

    fig = plt.figure(figsize=(10, 6))
    xpoints = date_rng
    ypoints = predictions

    plt.plot(xpoints, ypoints, color='blue', marker='o', linestyle='-', linewidth=2,
             markersize=5)  # Customize line style and marker
    plt.xticks(rotation=45, fontsize=10)  # Rotate x-axis labels and adjust fontsize
    plt.yticks(fontsize=10)  # Adjust fontsize of y-axis labels
    plt.xlabel('Date', fontsize=12)  # Set x-axis label and adjust fontsize
    plt.ylabel('Price', fontsize=12)  # Set y-axis label and adjust fontsize
    plt.title('Cryptocurrency Price Prediction', fontsize=14)  # Set plot title and adjust fontsize
    plt.grid(True, linestyle='--', alpha=0.5)  # Add grid lines with linestyle and transparency
    plt.tight_layout()  # Adjust layout to prevent clipping of labels

    # Display the plot in Streamlit
    st.pyplot(fig)
