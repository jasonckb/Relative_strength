import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the page configuration
st.set_page_config(page_title="Relative Strength Dashboard", layout="wide")

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
compare_days = st.sidebar.number_input('Compare N days ago', min_value=1, max_value=365, value=1)

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

def create_dashboard(data, rs_scores, rsi, date):
    latest_scores = rs_scores.loc[date].sort_values(ascending=False)
    latest_rsi = rsi.loc[date]

    dashboard_data = pd.DataFrame({
        'Symbol': latest_scores.index,
        'Score': latest_scores.values,
        'RSI': latest_rsi.values
    })

    dashboard_data = dashboard_data.sort_values('Score', ascending=False).reset_index(drop=True)
    dashboard_data['Rank'] = dashboard_data.index + 1

    ndx_score = dashboard_data.loc[dashboard_data['Symbol'] == '^NDX', 'Score'].values[0]
    gspc_score = dashboard_data.loc[dashboard_data['Symbol'] == '^GSPC', 'Score'].values[0]
    benchmark_score = max(ndx_score, gspc_score)

    fig, ax = plt.subplots(figsize=(12, 17))  # Slightly reduced figure height
    ax.axis('off')

    # Add date information closer to the table
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
        rsi_str = 'N/A' if np.isnan(rsi_value) or np.isinf(rsi_value) else f"{int(np.round(rsi_value))}"
        table_data[row_idx][col] = f"{symbol}: {score}\nRSI: {rsi_str}"

    # Adjust table position and properties
    table = ax.table(cellText=table_data, cellLoc='center', loc='center', bbox=[0, 0.02, 1, 0.95])
    table.auto_set_font_size(False)
    table.set_fontsize(16)  # Increased font size
    table.scale(1, 1.5)  # Reduced row height

    for (row, col), cell in table.get_celld().items():
        idx = row * columns + col
        if idx < num_symbols:
            symbol = dashboard_data.iloc[idx]['Symbol']
            score = int(dashboard_data.iloc[idx]['Score'])
            rsi_value = dashboard_data.iloc[idx]['RSI']
            
            if symbol in ['^NDX', '^GSPC']:
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
            rsi_str = 'N/A' if np.isnan(rsi_value) or np.isinf(rsi_value) else f"{int(np.round(rsi_value))}"
            text_obj.set_text(f"{symbol}: {score}\nRSI: {rsi_str}")
            
            if not np.isnan(rsi_value) and not np.isinf(rsi_value):
                if rsi_value > 75:
                    text_obj.set_color('red')
                elif rsi_value < 25:
                    text_obj.set_color('blue')
                else:
                    text_obj.set_color('black')
            else:
                text_obj.set_color('black')

    plt.tight_layout(pad=1.0)  # Reduce padding
    return fig

rs_scores = calculate_relative_strength(data, window=window)
rsi = calculate_rsi(data)

# Create dashboards for current and previous trading days
current_date = data.index[-1]
previous_date = data.index[-compare_days]  # Use the user input for previous days

# Create two columns for side-by-side display
col1, col2 = st.columns(2)

# Display the current day dashboard in the first column
with col1:
    current_dashboard = create_dashboard(data, rs_scores, rsi, current_date)
    st.pyplot(current_dashboard)

# Display the previous trading day dashboard in the second column
with col2:
    previous_dashboard = create_dashboard(data, rs_scores, rsi, previous_date)
    st.pyplot(previous_dashboard)








