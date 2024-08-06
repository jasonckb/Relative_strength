import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set the page configuration
st.set_page_config(page_title="Relative Strength Dashboard", layout="wide")

# Define the US symbols
us_symbols = [
    '^NDX', '^GSPC', 'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOG', 'META', 'TSLA', 'JPM',
    # ... (rest of the US symbols)
]

# Define the HK symbols
hk_symbols = [
    '^HSI', '^HSCEI', '0017.HK', '0241.HK', '0066.HK', '1038.HK', '0006.HK', '0011.HK', '0012.HK', '0857.HK',
    # ... (rest of the HK symbols)
]

# Sidebar for user input
st.sidebar.title("Relative Strength Dashboard by Jason")
st.sidebar.header('User Input')
market = st.sidebar.radio('Select Market', ['US Stock', 'HK Stock'])
window = st.sidebar.number_input('Moving Average Window for Relative Strength', min_value=1, max_value=365, value=200)
compare_days = st.sidebar.number_input('Compare N days ago', min_value=1, max_value=365, value=1)

# Select the appropriate symbol list and benchmarks based on the user's choice
if market == 'US Stock':
    symbols = us_symbols
    benchmarks = ['^NDX', '^GSPC']
else:  # HK Stock
    symbols = hk_symbols
    benchmarks = ['^HSI', '^HSCEI']

# Add a refresh button
if st.sidebar.button('Refresh Data'):
    st.cache_data.clear()

# Download data
@st.cache_data
def download_data(symbols):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    data = yf.download(symbols, start=start_date, end=end_date)['Close']
    return data

data = download_data(symbols)

# Calculate relative strength
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

# Updated RSI calculation
def calculate_rsi(data, window=14):
    delta = data.diff()
    
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    ma_up = up.ewm(com=window-1, adjust=False).mean()
    ma_down = down.ewm(com=window-1, adjust=False).mean()
    
    rs = ma_up / ma_down
    
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_dashboard(data, rs_scores, rsi, date, benchmarks):
    latest_scores = rs_scores.loc[date].sort_values(ascending=False)
    latest_rsi = rsi.loc[date]

    dashboard_data = pd.DataFrame({
        'Symbol': latest_scores.index,
        'Score': latest_scores.values,
        'RSI': latest_rsi.values
    })

    dashboard_data = dashboard_data.sort_values('Score', ascending=False).reset_index(drop=True)
    dashboard_data['Rank'] = dashboard_data.index + 1

    benchmark_scores = [dashboard_data.loc[dashboard_data['Symbol'] == b, 'Score'].values[0] for b in benchmarks]
    benchmark_score = max(benchmark_scores)

    fig, ax = plt.subplots(figsize=(12, 17))
    ax.axis('off')

    ax.text(0.5, 0.98, f"Relative Strength Dashboard ({date.strftime('%Y-%m-%d')})", 
            fontsize=16, fontweight='bold', ha='center', va='bottom', transform=ax.transAxes)

    num_symbols = len(dashboard_data)
    rows, columns = 20, 5
    table_data = [[''] * columns for _ in range(rows)]

    for idx, row in dashboard_data.iterrows():
        col = idx % columns
        row_idx = idx // columns
        symbol = row['Symbol']
        score = int(row['Score'])
        rsi_value = row['RSI']
        rsi_str = 'N/A' if np.isnan(rsi_value) or np.isinf(rsi_value) else f"{rsi_value:.1f}"
        table_data[row_idx][col] = f"{symbol}: {score}\nRSI: {rsi_str}"

    table = ax.table(cellText=table_data, cellLoc='center', loc='center', bbox=[0, 0.02, 1, 0.95])
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        idx = row * columns + col
        if idx < num_symbols:
            symbol = dashboard_data.iloc[idx]['Symbol']
            score = int(dashboard_data.iloc[idx]['Score'])
            rsi_value = dashboard_data.iloc[idx]['RSI']
            
            if symbol in benchmarks:
                cell.set_facecolor('yellow')
            elif score > 10 and score > benchmark_score:
                cell.set_facecolor('lightgreen')
            elif score > 0 and score <= benchmark_score:
                cell.set_facecolor('lightgray')
            elif score <= 0 and score <= benchmark_score:
                cell.set_facecolor('lightcoral')
            else:
                cell.set_facecolor('white')
            
            text_obj = cell.get_text()
            rsi_str = 'N/A' if np.isnan(rsi_value) or np.isinf(rsi_value) else f"{rsi_value:.1f}"
            text_obj.set_text(f"{symbol}: {score}\nRSI: {rsi_str}")
            
            if not np.isnan(rsi_value) and not np.isinf(rsi_value):
                if rsi_value > 70:
                    text_obj.set_color('red')
                elif rsi_value < 30:
                    text_obj.set_color('blue')
                else:
                    text_obj.set_color('black')
            else:
                text_obj.set_color('black')

    plt.tight_layout(pad=1.0)
    return fig

rsi = calculate_rsi(data)

# Get the current date and the date to compare against
current_date = data.index[-1]
previous_date = data.index[-compare_days]

# Calculate relative strength scores for the current date and the comparison date
rs_scores_current = calculate_relative_strength(data, window=window, date=current_date)
rs_scores_previous = calculate_relative_strength(data, window=window, date=previous_date)

# Create two columns for side-by-side display
col1, col2 = st.columns(2)

# Display the current day dashboard in the first column
with col1:
    current_dashboard = create_dashboard(data, rs_scores_current, rsi, current_date, benchmarks)
    st.pyplot(current_dashboard)

# Display the previous trading day dashboard in the second column
with col2:
    previous_dashboard = create_dashboard(data, rs_scores_previous, rsi, previous_date, benchmarks)
    st.pyplot(previous_dashboard)









