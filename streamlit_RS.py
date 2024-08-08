import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set the page configuration
st.set_page_config(page_title="Relative Strength Dashboard", layout="wide")

# Define the US symbols
us_symbols = [
    '^NDX', '^GSPC', 'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOG', 'META', 'TSLA', 'JPM',
    'V', 'UNH', 'LLY', 'JNJ', 'XOM', 'WMT', 'MA', 'PG', 'KO', 'HD',
    'AVGO', 'CVX', 'MRK', 'PE', 'GS', 'ABBV', 'COST', 'TSM', 'VZ', 'PFE',
    'NFLX', 'ADBE', 'ASML', 'CRM', 'ACN', 'TRV', 'BA', 'TXN', 'IBM', 'DIS',
    'UPS', 'SPGI', 'INTC', 'AMD', 'QCOM', 'AMT', 'CHTR', 'SBUX', 'MS', 'BLK',
    'GE', 'MMM', 'GILD', 'CAT', 'INTU', 'ISRG', 'AMGN', 'CVS', 'DE', 'EQIX',
    'TJX', 'PGR', 'BKNG', 'MU', 'LRCX', 'REGN', 'ARM', 'PLTR', 'SNOW', 'PANW',
    'CRWD', 'ZS', 'ABNB', 'CDNS', 'DDOG', 'ICE', 'TTD', 'TEAM', 'CEG', 'VST',
    'NRG', 'NEE', 'PYPL', 'FTNT', 'IDXX', 'SMH', 'XLU', 'XLP', 'XLE', 'XLK',
    'XLY', 'XLI', 'XLB', 'XLRE', 'XLF', 'XLV', 'OXY', 'NVO', 'CCL', 'LEN'
]

# Define the HK symbols
hk_symbols = [
    '^HSI', '2007.HK', '0017.HK', '0241.HK', '0066.HK', '1038.HK', '0006.HK', '0011.HK', '0012.HK', '0857.HK',
    '3988.HK', '1044.HK', '0386.HK', '2388.HK', '1113.HK', '0941.HK', '1997.HK', '0001.HK', '1093.HK', '1109.HK',
    '1177.HK', '1211.HK', '1299.HK', '1398.HK', '0016.HK', '0175.HK', '1810.HK', '1876.HK', '1928.HK', '2007.HK',
    '2018.HK', '2269.HK', '2313.HK', '2318.HK', '2319.HK', '2331.HK', '2382.HK', '2628.HK', '0267.HK', '0027.HK',
    '0288.HK', '0003.HK', '3690.HK', '0388.HK', '3968.HK', '0005.HK', '6098.HK', '0669.HK', '6862.HK', '0688.HK',
    '0700.HK', '0762.HK', '0823.HK', '0868.HK', '0883.HK', '0939.HK', '0960.HK', '0968.HK', '9988.HK', '1024.HK',
    '1347.HK', '1833.HK', '2013.HK', '2518.HK', '0268.HK', '0285.HK', '3888.HK', '0522.HK', '6060.HK', '6618.HK',
    '6690.HK', '0772.HK', '0909.HK', '9618.HK', '9626.HK', '9698.HK', '0981.HK', '9888.HK', '0992.HK', '9961.HK',
    '9999.HK', '2015.HK', '0291.HK', '0293.HK', '0358.HK', '1772.HK', '1776.HK', '1787.HK', '1801.HK', '1818.HK',
    '1898.HK', '0019.HK', '1929.HK', '0799.HK', '0836.HK', '0853.HK', '0914.HK', '0916.HK', '6078.HK', '2333.HK'
]

# Sidebar for user input
st.sidebar.title("Relative Strength Dashboard by Jason")
st.sidebar.header('User Input')
market = st.sidebar.radio('Select Market', ['US Stock', 'HK Stock'])
window = st.sidebar.number_input('Moving Average Window for Relative Strength', min_value=1, max_value=365, value=200)
compare_days = st.sidebar.number_input('Compare N days ago', min_value=1, max_value=365, value=1)

# Add a refresh button
if st.sidebar.button('Refresh Data'):
    st.cache_data.clear()

# Select the appropriate symbol list and benchmarks based on the user's choice
if market == 'US Stock':
    symbols = us_symbols
    benchmarks = ['^NDX', '^GSPC']
else:  # HK Stock
    symbols = hk_symbols
    benchmarks = ['^HSI']

# Download data
@st.cache_data
def download_data(symbols):
    hk_tz = pytz.timezone('Asia/Hong_Kong')
    end_date = datetime.now(hk_tz)
    start_date = end_date - timedelta(days=365)
    end_date += timedelta(days=1)
    data = yf.download(symbols, start=start_date, end=end_date, progress=False)
    return data

data = download_data(symbols)

# Print the last available date for debugging
st.write(f"Last available date in the data: {data.index[-1]}")

def calculate_relative_strength(data, window=200, date=None):
    if date is None:
        date = data.index[-1]
    rs_scores = pd.DataFrame(index=data.index, columns=data.columns)
    for symbol in data.columns:
        symbol_data = data[symbol]
        other_data = data.drop(columns=[symbol])
        pct_change = symbol_data.pct_change(periods=window).loc[:date]
        other_pct_change = other_data.pct_change(periods=window).loc[:date]
        outperformance = (pct_change.iloc[-1] > other_pct_change.iloc[-1]).sum()
        underperformance = (pct_change.iloc[-1] < other_pct_change.iloc[-1]).sum()
        rs_scores[symbol].loc[date] = outperformance - underperformance
    return rs_scores

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=window-1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window-1, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_dashboard(data, rs_scores, date, benchmarks):
    rsi_values = {}
    for symbol in data.columns.levels[1]:
        symbol_data = data['Close'][symbol].loc[:date].dropna()
        if len(symbol_data) >= 14:
            rsi_series = calculate_rsi(symbol_data)
            rsi_values[symbol] = rsi_series.iloc[-1]
        else:
            rsi_values[symbol] = np.nan

    latest_scores = rs_scores.loc[date].sort_values(ascending=False)
    dashboard_data = pd.DataFrame({
        'Symbol': latest_scores.index,
        'Score': latest_scores.values,
        'RSI': [rsi_values[symbol] for symbol in latest_scores.index]
    })

    dashboard_data = dashboard_data.reset_index(drop=True)
    dashboard_data['Rank'] = dashboard_data.index + 1

    benchmark_scores = [dashboard_data.loc[dashboard_data['Symbol'] == b, 'Score'].values[0] for b in benchmarks]
    benchmark_score = max(benchmark_scores)

    st.write(f"Relative Strength Dashboard ({date.strftime('%Y-%m-%d')})")
    
    cols = st.columns(5)
    for idx, row in dashboard_data.iterrows():
        col = cols[idx % 5]
        symbol = row['Symbol']
        score = int(row['Score'])
        rsi_value = row['RSI']
        rsi_str = 'N/A' if np.isnan(rsi_value) or np.isinf(rsi_value) else f"{rsi_value:.1f}"
        
        color = 'yellow' if symbol in benchmarks else \
                'lightgreen' if score > 10 and score > benchmark_score else \
                'lightgray' if score > 0 and score <= benchmark_score else \
                'lightcoral' if score <= 0 and score <= benchmark_score else \
                'white'
        
        text_color = 'red' if not np.isnan(rsi_value) and not np.isinf(rsi_value) and rsi_value > 70 else \
                     'blue' if not np.isnan(rsi_value) and not np.isinf(rsi_value) and rsi_value < 30 else \
                     'black'
        
        if col.button(f"{symbol}: {score}\nRSI: {rsi_str}", key=f"{symbol}_{date}"):
            st.session_state.selected_symbol = symbol

def create_candlestick_chart(symbol_data, symbol, days=100):
    df = symbol_data.tail(days)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(symbol, 'Volume'),
                        row_width=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='Price'),
                  row=1, col=1)
    
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)
    
    fig.update_layout(height=600, width=800, title_text=f"{symbol} - Last {days} Days",
                      showlegend=False, xaxis_rangeslider_visible=False)
    
    return fig

def get_previous_trading_day(data, current_date, days_ago):
    date_index = data.index.get_loc(current_date)
    if days_ago <= date_index:
        return data.index[date_index - days_ago]
    else:
        st.error(f"Not enough historical data to go back {days_ago} trading days.")
        return None

# Get the current date and the date to compare against
current_date = data.index[-1]
previous_date = get_previous_trading_day(data['Close'], current_date, compare_days)

if previous_date is not None:
    rs_scores_current = calculate_relative_strength(data['Close'], window=window, date=current_date)
    rs_scores_previous = calculate_relative_strength(data['Close'], window=window, date=previous_date)

    col1, col2 = st.columns(2)

    with col1:
        create_dashboard(data, rs_scores_current, current_date, benchmarks)

    with col2:
        create_dashboard(data, rs_scores_previous, previous_date, benchmarks)

    if 'selected_symbol' in st.session_state and st.session_state.selected_symbol:
        symbol_data = data[st.session_state.selected_symbol]
        candlestick_chart = create_candlestick_chart(symbol_data, st.session_state.selected_symbol)
        st.plotly_chart(candlestick_chart)

else:
    st.error("Unable to create comparison dashboard due to insufficient historical data.")

