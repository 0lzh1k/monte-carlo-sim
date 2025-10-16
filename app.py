import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.express as px
from simulation import run_monte_carlo_simulation, calculate_risk_metrics, get_simulation_summary

st.title("Monte Carlo Simulation for Financial Risk Assessment")

st.header("Project Overview")
st.write("""
This application performs a Monte Carlo simulation to model financial risk and return. 
Users can input financial parameters and visualize the potential outcomes of their investments.
""")

st.sidebar.header("User Input Parameters")

stock_ticker = st.sidebar.text_input("Stock Ticker (optional - for real data)", value="AAPL", help="Enter a stock ticker to use real historical data")
use_real_data = st.sidebar.checkbox("Use Real Market Data", value=False, help="Fetch real data to calculate mu and sigma")

def get_stock_data(ticker, period="2y"):
    """Fetch stock data and calculate returns statistics"""
    try:
        hist = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        
        if hist.empty:
            return None, None, None
        
        if 'Adj Close' in hist.columns:
            close_prices = hist['Adj Close']
        elif ('Adj Close', ticker) in hist.columns:
            close_prices = hist[('Adj Close', ticker)]
        elif 'Close' in hist.columns:
            close_prices = hist['Close']
        else:
            if hist.columns.nlevels > 1:
                close_prices = hist.xs('Adj Close', level=0, axis=1).iloc[:, 0]
            else:
                st.error(f"Could not find price data for {ticker}")
                return None, None, None
            
        daily_returns = close_prices.pct_change().dropna()
        
        mu = daily_returns.mean() * 252
        sigma = daily_returns.std() * np.sqrt(252)
        
        return mu, sigma, hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, None, None

def user_input_features():
    initial_investment = st.sidebar.number_input("Initial Investment (USD)", min_value=1000, max_value=1000000, value=10000, step=1000)
    
    if use_real_data and stock_ticker:
        mu_real, sigma_real, stock_hist = get_stock_data(stock_ticker)
        if mu_real is not None:
            st.sidebar.success(f"Using real data from {stock_ticker}")
            mu_val = float(mu_real)
            sigma_val = float(sigma_real)
            st.sidebar.write(f"Calculated μ: {mu_val:.4f} ({mu_val*100:.2f}%)")
            st.sidebar.write(f"Calculated σ: {sigma_val:.4f} ({sigma_val*100:.2f}%)")
            mu_default = mu_val
            sigma_default = sigma_val
        else:
            st.sidebar.error("Failed to fetch real data, using manual inputs")
            mu_default = 0.08
            sigma_default = 0.15
    else:
        mu_default = 0.08
        sigma_default = 0.15
        stock_hist = None
    
    mu = st.sidebar.slider('Expected Annual Return (μ)', 0.0, 0.50, mu_default, format="%.4f")
    sigma = st.sidebar.slider('Annual Volatility (σ)', 0.0, 1.0, sigma_default, format="%.4f")
    time_horizon = st.sidebar.slider('Time Horizon (Years)', 1, 30, 10)
    num_simulations = st.sidebar.slider('Number of Simulations', 100, 10000, 5000)
    
    data = {'initial_investment': initial_investment,
            'mu': mu,
            'sigma': sigma,
            'time_horizon': time_horizon,
            'num_simulations': num_simulations,
            'stock_hist': stock_hist,
            'ticker': stock_ticker if use_real_data else None}
    
    return data

params = user_input_features()

def get_price_column(hist, ticker, column_name='Adj Close'):
    """Helper function to get the right price column from yfinance data"""
    if column_name in hist.columns:
        return hist[column_name]
    elif (column_name, ticker) in hist.columns:
        return hist[(column_name, ticker)]
    elif column_name == 'Adj Close' and 'Close' in hist.columns:
        return hist['Close']
    elif hist.columns.nlevels > 1:
        try:
            col_data = hist.xs(column_name, level=0, axis=1)
            if isinstance(col_data, pd.DataFrame):
                return col_data.iloc[:, 0]
            return col_data
        except:
            col_data = hist.xs('Close', level=0, axis=1)
            if isinstance(col_data, pd.DataFrame):
                return col_data.iloc[:, 0]
            return col_data
    else:
        col_data = hist.iloc[:, -1]
        if isinstance(col_data, pd.DataFrame):
            return col_data.iloc[:, 0]
        return col_data

if params['stock_hist'] is not None and params['ticker']:
    st.subheader(f"Historical Data for {params['ticker']}")
    
    adj_close_prices = get_price_column(params['stock_hist'], params['ticker'], 'Adj Close')
    
    if hasattr(adj_close_prices, 'values'):
        price_values = adj_close_prices.values.flatten()
    else:
        price_values = np.array(adj_close_prices).flatten()
    
    plot_data = pd.DataFrame({
        'Date': params['stock_hist'].index,
        'Price': price_values
    })
    
    fig_hist = px.line(plot_data, x='Date', y='Price', 
                       title=f"{params['ticker']} Historical Adjusted Close Price")
    fig_hist.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig_hist)
    
    recent_data = params['stock_hist'].tail()
    st.write("Recent Price Data:")
    
    try:
        if params['stock_hist'].columns.nlevels > 1:
            display_cols = []
            for col_name in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if (col_name, params['ticker']) in params['stock_hist'].columns:
                    display_cols.append((col_name, params['ticker']))
                elif col_name in [col[0] for col in params['stock_hist'].columns]:
                    matching_cols = [col for col in params['stock_hist'].columns if col[0] == col_name]
                    if matching_cols:
                        display_cols.append(matching_cols[0])
            
            if display_cols:
                st.dataframe(recent_data[display_cols])
            else:
                st.dataframe(recent_data)
        else:
            available_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in recent_data.columns]
            if available_cols:
                st.dataframe(recent_data[available_cols])
            else:
                st.dataframe(recent_data)
    except Exception as e:
        st.write("Recent data structure:")
        st.dataframe(recent_data)

if st.sidebar.button("Run Simulation"):
    if params['ticker']:
        st.write(f"Running simulation based on {params['ticker']} data...")
    else:
        st.write("Running simulation with manual parameters...")
    
    portfolio_values = run_monte_carlo_simulation(
        params['initial_investment'],
        params['mu'],
        params['sigma'],
        params['time_horizon'],
        params['num_simulations']
    )
    
    st.subheader("Interactive Simulation Outcomes")
    
    st.write(f"**Simulation Parameters:**")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"- Expected Annual Return (μ): {params['mu']:.4f} ({params['mu']*100:.2f}%)")
        st.write(f"- Annual Volatility (σ): {params['sigma']:.4f} ({params['sigma']*100:.2f}%)")
    with col2:
        st.write(f"- Time Horizon: {params['time_horizon']} years")
        st.write(f"- Number of Simulations: {params['num_simulations']:,}")
        if params['ticker']:
            st.write(f"- Data Source: {params['ticker']} (Real Market Data)")
        else:
            st.write(f"- Data Source: Manual Parameters")
    
    # Limit the number of simulations shown in interactive plot to avoid browser memory issues
    max_paths_to_show = min(100, params['num_simulations'])
    portfolio_subset = portfolio_values[:, :max_paths_to_show]
    
    sim_df = pd.DataFrame(portfolio_subset)
    sim_df = sim_df.reset_index().melt(id_vars='index', var_name='simulation', value_name='value')
    sim_df.rename(columns={'index': 'day'}, inplace=True)

    fig_interactive = px.line(sim_df, x='day', y='value', color='simulation', 
                              title=f"Monte Carlo Simulation - Sample of {max_paths_to_show} paths (out of {params['num_simulations']:,} total)")
    fig_interactive.update_layout(showlegend=False)
    st.plotly_chart(fig_interactive)
    
    if max_paths_to_show < params['num_simulations']:
        st.info(f"Showing {max_paths_to_show} sample paths for performance. All {params['num_simulations']:,} simulations are used in the statistical analysis below.")

    st.subheader("Probabilistic Forecast with Confidence Interval")
    
    median_values = np.percentile(portfolio_values, 50, axis=1)
    percentile_25 = np.percentile(portfolio_values, 25, axis=1)
    percentile_75 = np.percentile(portfolio_values, 75, axis=1)
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(median_values, label='Median Outcome (50th Percentile)', color='blue')
    ax3.fill_between(range(len(median_values)), percentile_25, percentile_75, color='blue', alpha=0.2, label='Interquartile Range (25th-75th Percentile)')
    ax3.set_title(f"Probabilistic Forecast over {params['time_horizon']} Years")
    ax3.set_xlabel("Trading Days")
    ax3.set_ylabel("Portfolio Value (USD)")
    ax3.legend()
    st.pyplot(fig3)
    
    st.subheader("Risk Assessment")
    final_values = portfolio_values[-1, :]

    risk_metrics = calculate_risk_metrics(portfolio_values)
    simulation_summary = get_simulation_summary(portfolio_values, params['initial_investment'])

    st.markdown("### Key Risk Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Mean Final Value",
            value=f"${risk_metrics['mean_final_value']:,.0f}",
            delta=f"{simulation_summary['total_return']:+.1%} total return"
        )

    with col2:
        st.metric(
            label="Standard Deviation",
            value=f"${risk_metrics['std_final_value']:,.0f}",
            delta=f"{risk_metrics['std_final_value']/risk_metrics['mean_final_value']:.1%} of mean"
        )

    with col3:
        st.metric(
            label="VaR (5%)",
            value=f"${risk_metrics['var_5']:,.0f}",
            delta=f"{(risk_metrics['var_5']/params['initial_investment']-1):+.1%} from initial"
        )

    with col4:
        st.metric(
            label="Success Rate",
            value=f"{simulation_summary['probability_positive_return']:.1%}",
            delta="Probability > Initial Investment"
        )

    st.markdown("### Risk Insights")

    col5, col6, col7 = st.columns(3)

    with col5:
        confidence_interval = f"${simulation_summary['min_portfolio_value']:,.0f} - ${simulation_summary['max_portfolio_value']:,.0f}"
        st.metric(
            label="90% Confidence Interval",
            value=confidence_interval,
            help="Range containing 90% of simulation outcomes"
        )

    with col6:
        st.metric(
            label="Expected Shortfall (5%)",
            value=f"${risk_metrics['expected_shortfall_5']:,.0f}",
            help="Average loss when VaR threshold is breached"
        )

    with col7:
        st.metric(
            label="Median Outcome",
            value=f"${simulation_summary['median_portfolio_value']:,.0f}",
            delta=f"{(simulation_summary['median_portfolio_value']/params['initial_investment']-1):+.1%} from initial"
        )

    st.markdown("### Risk Interpretation")
    if risk_metrics['var_5'] < params['initial_investment'] * 0.8:
        risk_level = "High Risk"
        risk_desc = "High probability of significant losses. Consider conservative approach."
    elif risk_metrics['var_5'] < params['initial_investment'] * 0.95:
        risk_level = "Moderate Risk"
        risk_desc = "Moderate risk with some downside protection."
    else:
        risk_level = "Low Risk"
        risk_desc = "Conservative approach with good downside protection."

    st.info(f"**{risk_level}**: {risk_desc}")

    st.subheader("Distribution of Final Portfolio Values")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(final_values, bins=50, density=True, alpha=0.7, label='Final Values')
    ax2.axvline(risk_metrics['mean_final_value'], color='r', linestyle='--',
                label=f'Mean: ${risk_metrics["mean_final_value"]:,.0f}')
    ax2.axvline(risk_metrics['var_5'], color='g', linestyle='--',
                label=f'VaR (5%): ${risk_metrics["var_5"]:,.0f}')
    ax2.set_title("Histogram of Final Portfolio Values")
    ax2.set_xlabel("Final Portfolio Value (USD)")
    ax2.set_ylabel("Probability Density")
    ax2.legend()
    st.pyplot(fig2)
