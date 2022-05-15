
'''
Moving Average Crossover Strategy
There are several ways in which stock market analysts and investors can use moving averages to analyse price trends and predict upcoming change of trends. 
There are vast varieties of the moving average strategies that can be developed using different types of moving averages. In this article, 
I’ve tried to demonstrate well-known simplistic yet effective momentum strategies — Simple Moving Average Crossover strategy and Exponential Moving Average Crossover strategy.

In the statistics of time-series, and in particular the Stock market technical analysis, a moving-average crossover occurs when on plotting, 
the two moving averages each based on different time-periods tend to cross. 
This indicator uses two (or more) moving averages — a faster moving average(short-term) and a slower(long-term) moving average. 
The faster moving average may be 5-, 10- or 25-day period while the slower moving average can be 50-, 100- or 200-day period.
A short term moving average is faster because it only considers prices over short period of time and is thus more reactive to daily price changes. 
On the other hand, a long-term moving average is deemed slower as it encapsulates prices over a longer period and is more lethargic.

------------------------------------------------------------------------------------------------------------------------------------------------------------
Generating Trade signals from crossovers
A moving average, as a line by itself, is often overlaid in price charts to indicate price trends. 
A crossover occurs when a faster moving average (i.e. a shorter period moving average) crosses a slower moving average 
(i.e. a longer period moving average). In stock trading, this meeting point can be used as a potential indicator to buy or sell an asset.

When the short term moving average crosses above the long term moving average, this indicates a buy signal.
Contrary, when the short term moving average crosses below the long term moving average, it may be a good moment to sell.
Having equipped with the necessary theory, now let’s continue our Python implementation wherein we’ll try to incorporate this strategy.

In our existing pandas dataframe, create a new column ‘Signal’ such that if 20-day SMA is greater than 50-day SMA 
then set Signal value as 1 else when 50-day SMA is greater than 20-day SMA then set it’s value as 0.
'''
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import datetime
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')
from binance.client import Client
import pandas_datareader.data as web
import pylab as p

matplotlib.use("TkAgg")

# client configuration
api_key = 'zfGBtmzyRrRuxkjYDU8bwh8e8CeV0dLLRc05jWCdJY6q2eLUtQMBiw1pRJsQELeu' 
api_secret = 'IKzXhtUT5XMIQDQv3KHwsRHg41UkeVRqcDkcsFUcn5TRaVOPTZQx7dZSA9Dgv0o2'
symbol = "BTCUSDT"

def getDataset():
    client = Client(api_key, api_secret)
    # print (stock_df[1])

    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY , "1 Jan, 2017" )
    data = pd.DataFrame(klines)
    # create colums name
    data.columns = ['open_time','open', 'high', 'low', 'close', 'volume','close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']
                
    # change the timestamp
    data.index = [datetime.datetime.fromtimestamp(x/1000.0) for x in data.close_time]
    data['datetime'] = [datetime.datetime.fromtimestamp(x/1000.0) for x in data.close_time]
    
    # data.to_csv(symbol+'.csv', index = None, header=True)

    # print (data)
    # print (data['close'])
    return data

def MovingAverageCrossStrategy(stock_symbol = 'BTC/USDT', start_date = '2018-01-01', end_date = '2020-01-01', 
                               short_window = 20, long_window = 50, moving_avg = 'SMA', display_table = True):
    '''
    The function takes the stock symbol, time-duration of analysis, 
    look-back periods and the moving-average type(SMA or EMA) as input 
    and returns the respective MA Crossover chart along with the buy/sell signals for the given period.
    '''
    # stock_symbol - (str)stock ticker as on Yahoo finance. Eg: 'ULTRACEMCO.NS' 
    # start_date - (str)start analysis from this date (format: 'YYYY-MM-DD') Eg: '2018-01-01'
    # end_date - (str)end analysis on this date (format: 'YYYY-MM-DD') Eg: '2020-01-01'
    # short_window - (int)lookback period for short-term moving average. Eg: 5, 10, 20 
    # long_window - (int)lookback period for long-term moving average. Eg: 50, 100, 200
    # moving_avg - (str)the type of moving average to use ('SMA' or 'EMA')
    # display_table - (bool)whether to display the date and price table at buy/sell positions(True/False)

    # import the closing price data of the stock for the aforementioned period of time in Pandas dataframe
    start = datetime.datetime(*map(int, start_date.split('-')))
    end = datetime.datetime(*map(int, end_date.split('-'))) 
    # stock_df = web.DataReader(stock_symbol, 'yahoo', start = start, end = end)['Close']
    # print (stock_df.index)
    # print (stock_df.columes)
    # print (stock_df)
    # stock_df = pd.DataFrame(stock_df) # convert Series object to dataframe 
    stock_df = getDataset()
    # stock_df.index = stock_df.index.strftime('%y-%m-%d')
    # stock_df=stock_df.rename(index={1: 'Date'})
    # print (stock_df_2)
    # convert just columns "a" and "b"
    stock_df[["close"]] = stock_df[["close"]].apply(pd.to_numeric)
    # print (stock_df.index)
    # stock_df.columns = {'Close Price'} # assign new colun name
    # stock_df['Close Price'] = stock_df['close']
    # print (stock_df['Close Price'] )
    stock_df['Close Price'] = stock_df['close']

    stock_df.dropna(axis = 0, inplace = True) # remove any null rows 
    # column names for long and short moving average columns
    short_window_col = str(short_window) + '_' + moving_avg
    long_window_col = str(long_window) + '_' + moving_avg  
  
    if moving_avg == 'SMA':
        # Create a short simple moving average column
        stock_df[short_window_col] = stock_df['Close Price'].rolling(window = short_window, min_periods = 1).mean()

        # Create a long simple moving average column
        stock_df[long_window_col] = stock_df['Close Price'].rolling(window = long_window, min_periods = 1).mean()

    elif moving_avg == 'EMA':
        # Create short exponential moving average column
        stock_df[short_window_col] = stock_df['Close Price'].ewm(span = short_window, adjust = False).mean()

        # Create a long exponential moving average column
        stock_df[long_window_col] = stock_df['Close Price'].ewm(span = long_window, adjust = False).mean()

    # create a new column 'Signal' such that if faster moving average is greater than slower moving average 
    # then set Signal as 1 else 0.
    stock_df['Signal'] = 0.0  
    stock_df['Signal'] = np.where(stock_df[short_window_col] > stock_df[long_window_col], 1.0, 0.0) 

    # create a new column 'Position' which is a day-to-day difference of the 'Signal' column. 
    stock_df['Position'] = stock_df['Signal'].diff()

    # plot close price, short-term and long-term moving averages
    plt.figure(figsize = (20,10))
    plt.tick_params(axis = 'both', labelsize = 14)
    stock_df['Close Price'].plot(color = 'k', lw = 1, label = 'Close Price')  
    stock_df[short_window_col].plot(color = 'b', lw = 1, label = short_window_col)
    stock_df[long_window_col].plot(color = 'g', lw = 1, label = long_window_col) 

    print(stock_df.index)
    print(stock_df)
    stock_df.to_csv(symbol+'.csv', index = None, header=True)

    # plot 'buy' signals
    plt.plot(stock_df[stock_df['Position'] == 1].index, 
            stock_df[short_window_col][stock_df['Position'] == 1], 
            '^', markersize = 15, color = 'g', alpha = 0.7, label = 'buy')

    # plot 'sell' signals
    plt.plot(stock_df[stock_df['Position'] == -1].index, 
            stock_df[short_window_col][stock_df['Position'] == -1], 
            'v', markersize = 15, color = 'r', alpha = 0.7, label = 'sell')
    plt.ylabel('Price in ₹', fontsize = 16 )
    plt.xlabel('Date', fontsize = 16 )
    plt.title(str(stock_symbol) + ' - ' + str(moving_avg) + ' Crossover', fontsize = 20)
    plt.legend()
    plt.grid()
    plt.show()
    # plt.ion()
    if display_table == True:
        df_pos = stock_df[(stock_df['Position'] == 1) | (stock_df['Position'] == -1)]
        df_pos['Position'] = df_pos['Position'].apply(lambda x: 'Buy' if x == 1 else 'Sell')
        print(tabulate(df_pos, headers = 'keys', tablefmt = 'psql'))



MovingAverageCrossStrategy()