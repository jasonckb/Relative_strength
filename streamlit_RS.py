import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz

# Set the page configuration
st.set_page_config(page_title="Relative Strength Dashboard", layout="wide")

# (Keep the symbol lists and user input code as before)

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

# (Keep the calculate_relative_strength and calculate_rsi functions as before)

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

    fig = go.Figure(data=[go.Table(
        header=dict(values=['Rank', 'Symbol', 'Score', 'RSI'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[dashboard_data['Rank'], dashboard_data['Symbol'], 
                           dashboard_data['Score'], dashboard_data['RSI'].round(1)],
                   fill_color=[['lightgreen' if (score > 10 and score > benchmark_score) else
                                'lightgray' if (score > 0 and score <= benchmark_score) else
                                'lightcoral' if (score <= 0 and score <= benchmark_score) else
                                'white' for score in dashboard_data['Score']]],
                   align='left'))
    ])

    fig.update_layout(title=f"Relative Strength Dashboard ({date.strftime('%Y-%m-%d')})",
                      height=800, width=600)

    return fig

# (Keep the get_previous_trading_day function as before)

# Main execution
current_date = data.index[-1]
previous_date = get_previous_trading_day(data['Close'], current_date, compare_days)

if previous_date is not None:
    rs_scores_current = calculate_relative_strength(data['Close'], window=window, date=current_date)
    rs_scores_previous = calculate_relative_strength(data['Close'], window=window, date=previous_date)

    col1, col2 = st.columns(2)

    with col1:
        current_dashboard = create_dashboard(data, rs_scores_current, current_date, benchmarks)
        st.plotly_chart(current_dashboard)

    with col2:
        previous_dashboard = create_dashboard(data, rs_scores_previous, previous_date, benchmarks)
        st.plotly_chart(previous_dashboard)

    # Add interactivity for candlestick charts
    st.write("Hover over a symbol in the dashboard and click to see its candlestick chart.")
    selected_symbol = st.selectbox("Or select a symbol here:", data.columns.levels[1])
    if selected_symbol:
        candlestick_chart = create_candlestick_chart(data[selected_symbol], selected_symbol)
        st.plotly_chart(candlestick_chart)
else:
    st.error("Unable to create comparison dashboard due to insufficient historical data.")










