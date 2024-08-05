import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the page configuration
st.set_page_config(page_title="Relative Strength Dashboard", layout="centered")

# Define the 100 symbols
symbols = [
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

# Sidebar for user input
st.sidebar.header('User Input')
window = st.sidebar.number_input('Moving Average Window for Relative Strength', min_value=1, max_value=365, value=200)

# Download data
@st.cache_data
def download_data(symbols):
    data = yf.download(symbols, period="1y")['Close']
    return data

data = download_data(symbols)

# Calculate relative strength
def calculate_relative_strength(data, window=200):
    rs_scores = pd.DataFrame(index=data.index, columns=data.columns)
    for symbol in data.columns:
        symbol_data = data[symbol]
        other_data = data.drop(columns=[symbol])
        
        pct_change = symbol_data.pct_change(periods=window)
        other_pct_change = other_data.pct_change(periods=window)
        
        outperformance = (pct_change.iloc[-1] > other_pct_change.iloc[-1]).sum()
        underperformance = (pct_change.iloc[-1] < other_pct_change.iloc[-1]).sum()
        
        rs_scores[symbol] = outperformance - underperformance
    
    return rs_scores

# RSI calculation
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(com=window-1, adjust=False).mean()
    avg_loss = loss.ewm(com=window-1, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

rs_scores = calculate_relative_strength(data, window=window)
rsi = calculate_rsi(data)

# Get the latest scores and RSI values
latest_scores = rs_scores.iloc[-1].sort_values(ascending=False)
latest_rsi = rsi.iloc[-1]

# Create a DataFrame with scores and RSI
dashboard_data = pd.DataFrame({
    'Symbol': latest_scores.index,
    'Score': latest_scores.values,
    'RSI': latest_rsi.values
})

# Sort by Score and add Rank
dashboard_data = dashboard_data.sort_values('Score', ascending=False).reset_index(drop=True)
dashboard_data['Rank'] = dashboard_data.index + 1

# Get benchmark scores
ndx_score = dashboard_data.loc[dashboard_data['Symbol'] == '^NDX', 'Score'].values[0]
gspc_score = dashboard_data.loc[dashboard_data['Symbol'] == '^GSPC', 'Score'].values[0]
benchmark_score = max(ndx_score, gspc_score)

# Create the dashboard
fig, ax = plt.subplots(figsize=(20, 12))
ax.axis('off')

# Add title
ax.text(0.5, 1.05, "Relative Strength Dashboard", fontsize=24, fontweight='bold', ha='center', va='bottom', transform=ax.transAxes)

# Prepare data for the table
num_symbols = len(dashboard_data)
rows = 10
columns = 10

table_data = [[''] * columns for _ in range(rows)]

for idx, row in dashboard_data.iterrows():
    col = idx % columns
    row_idx = idx // columns
    symbol = row['Symbol']
    score = int(row['Score'])
    rsi = row['RSI']
    table_data[row_idx][col] = f"{symbol}: {score}\nRSI: {rsi:.1f}"

# Create the table
table = ax.table(cellText=table_data, cellLoc='center', loc='center')

# Style the table
table.auto_set_font_size(True)
#table.set_fontsize(11)
table.scale(2, 3)

# Color coding for cells
for (row, col), cell in table.get_celld().items():
    idx = row * columns + col
    if idx < num_symbols:
        symbol = dashboard_data.iloc[idx]['Symbol']
        score = int(dashboard_data.iloc[idx]['Score'])
        rsi = dashboard_data.iloc[idx]['RSI']
        
        # Set cell color based on new rules
        if symbol in ['^NDX', '^GSPC']:
            cell.set_facecolor('yellow')
        elif score > 10 and score > benchmark_score:
            cell.set_facecolor('lightgreen')
        elif score > 0 and score <= benchmark_score:
            cell.set_facecolor('lightgray')
        elif score <= 0 and score <= benchmark_score:
            cell.set_facecolor('lightcoral')
        else:
            cell.set_facecolor('white')  # For any other case
        
        # Set text and RSI color
        text_obj = cell.get_text()
        text_obj.set_text(f"{symbol}: {score}\nRSI: {rsi:.1f}")
        if rsi > 75:
            text_obj.set_color('red')
        elif rsi < 25:
            text_obj.set_color('blue')
        else:
            text_obj.set_color('black')

plt.tight_layout()
plt.subplots_adjust(top=0.85)  # Adjust top margin for title

# Display the dashboard in Streamlit
st.pyplot(fig)







