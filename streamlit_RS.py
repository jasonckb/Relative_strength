def create_dashboard(data, rs_scores, date, benchmarks):
    # Calculate RSI for each symbol
    rsi_values = {}
    for symbol in data.columns:
        symbol_data = data[symbol].loc[:date].dropna()
        if len(symbol_data) >= 14:
            rsi_series = calculate_rsi(symbol_data)
            rsi_values[symbol] = rsi_series.iloc[-1] if not rsi_series.empty else np.nan
        else:
            rsi_values[symbol] = np.nan

    # Sort symbols based on relative strength scores
    latest_scores = rs_scores.loc[date].sort_values(ascending=False)

    # Create dashboard data
    dashboard_data = pd.DataFrame({
        'Symbol': latest_scores.index,
        'Score': latest_scores.values,
        'RSI': [rsi_values[symbol] for symbol in latest_scores.index]
    })

    dashboard_data = dashboard_data.reset_index(drop=True)
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











