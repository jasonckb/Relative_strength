import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz

@st.cache_data
def download_data(symbols):
    hk_tz = pytz.timezone('Asia/Hong_Kong')
    end_date = datetime.now(hk_tz)
    start_date = end_date - timedelta(days=365)
    end_date += timedelta(days=1)
    
    all_data = {}
    for symbol in symbols:
        try:
            symbol_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not symbol_data.empty:
                all_data[symbol] = symbol_data
            else:
                st.warning(f"No data available for {symbol}")
        except Exception as e:
            st.warning(f"Error downloading data for {symbol}: {str(e)}")
    
    if not all_data:
        st.error("Unable to download data for any symbols")
        return None
    
    combined_data = pd.concat(all_data.values(), keys=all_data.keys(), axis=1)
    combined_data.columns.names = ['Symbol', 'Data']
    return combined_data

# Use the function
symbols = us_symbols if market == 'US Stock' else hk_symbols
data = download_data(symbols)

if data is not None:
    st.write(f"Last available date in the data: {data.index[-1]}")
else:
    st.error("No data available. Please check your internet connection and try again.")









