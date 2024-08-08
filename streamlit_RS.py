def compare_top_stocks(current_data, previous_data, n=10):
    # Ensure 'Score' column is numeric
    current_data['Score'] = pd.to_numeric(current_data['Score'], errors='coerce')
    previous_data['Score'] = pd.to_numeric(previous_data['Score'], errors='coerce')
    
    current_top = current_data.nlargest(n, 'Score')
    previous_top = previous_data.nlargest(n, 'Score')
    
    maintained = set(current_top['Symbol']) & set(previous_top['Symbol'])
    new_entries = set(current_top['Symbol']) - set(previous_top['Symbol'])
    dropped_out = set(previous_top['Symbol']) - set(current_top['Symbol'])
    
    return maintained, new_entries, dropped_out

# Modify the main execution part of the script

# Main execution
current_date = data.index[-1]
previous_date = get_previous_trading_day(data, current_date, compare_days)

if previous_date is not None:
    rs_scores_current = calculate_relative_strength(data, window=window, date=current_date)
    rs_scores_previous = calculate_relative_strength(data, window=window, date=previous_date)

    # Create dashboard data for comparison
    dashboard_data_current = pd.DataFrame({
        'Symbol': rs_scores_current.loc[current_date].index,
        'Score': rs_scores_current.loc[current_date].values
    }).sort_values('Score', ascending=False).reset_index(drop=True)

    dashboard_data_previous = pd.DataFrame({
        'Symbol': rs_scores_previous.loc[previous_date].index,
        'Score': rs_scores_previous.loc[previous_date].values
    }).sort_values('Score', ascending=False).reset_index(drop=True)

    # Ensure 'Score' column is numeric
    dashboard_data_current['Score'] = pd.to_numeric(dashboard_data_current['Score'], errors='coerce')
    dashboard_data_previous['Score'] = pd.to_numeric(dashboard_data_previous['Score'], errors='coerce')

    # Compare top stocks
    maintained, new_entries, dropped_out = compare_top_stocks(dashboard_data_current, dashboard_data_previous)

    # Display top stocks comparison
    st.header("Top Stocks")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Maintained in Top 10")
        st.write(", ".join(maintained))
    with col2:
        st.subheader("New in Top 10")
        st.write(", ".join(new_entries))
    with col3:
        st.subheader("Dropped from Top 10")
        st.write(", ".join(dropped_out))

    # Create two columns for side-by-side display
    dashboard_col1, dashboard_col2 = st.columns(2)

    # Display the current day dashboard in the first column
    with dashboard_col1:
        current_dashboard = create_dashboard(data, rs_scores_current, current_date, benchmarks)
        st.pyplot(current_dashboard)

    # Display the previous trading day dashboard in the second column
    with dashboard_col2:
        previous_dashboard = create_dashboard(data, rs_scores_previous, previous_date, benchmarks)
        st.pyplot(previous_dashboard)
else:
    st.error("Unable to create comparison dashboard due to insufficient historical data.")

