"""
Monte Carlo Simulation Module for Financial Risk Assessment

This module contains the core Monte Carlo simulation logic for modeling
portfolio value projections under various market conditions.
"""

import numpy as np


def run_monte_carlo_simulation(initial_investment, mu, sigma, time_horizon, num_simulations, seed=None):
    """
    Run Monte Carlo simulation for portfolio value projection.

    Parameters:
    -----------
    initial_investment : float
        Initial portfolio value in USD
    mu : float
        Expected annual return (as decimal, e.g., 0.08 for 8%)
    sigma : float
        Annual volatility (as decimal, e.g., 0.15 for 15%)
    time_horizon : float
        Investment time horizon in years
    num_simulations : int
        Number of simulation paths to generate
    seed : int, optional
        Random seed for reproducible results

    Returns:
    --------
    numpy.ndarray
        Portfolio values over time, shape: (num_steps+1, num_simulations)
        where num_steps = time_horizon * 252 (trading days)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1/252  # Trading days per year
    num_steps = int(time_horizon * 252)

    daily_returns = np.random.lognormal(
        mean=(mu - 0.5 * sigma**2) * dt,
        sigma=sigma * np.sqrt(dt),
        size=(num_steps, num_simulations)
    )

    daily_returns = np.vstack([np.ones((1, num_simulations)), daily_returns])

    cumulative_returns = daily_returns.cumprod(axis=0)

    portfolio_values = initial_investment * cumulative_returns

    return portfolio_values


def calculate_risk_metrics(portfolio_values):
    """
    Calculate key risk metrics from simulation results.

    Parameters:
    -----------
    portfolio_values : numpy.ndarray
        Portfolio values from Monte Carlo simulation

    Returns:
    --------
    dict
        Dictionary containing risk metrics:
        - mean_final_value: Mean final portfolio value
        - std_final_value: Standard deviation of final values
        - var_5: Value at Risk at 5% confidence
        - var_1: Value at Risk at 1% confidence
        - expected_shortfall_5: Expected shortfall at 5% confidence
    """
    final_values = portfolio_values[-1, :]

    metrics = {
        'mean_final_value': float(np.mean(final_values)),
        'std_final_value': float(np.std(final_values)),
        'var_5': float(np.percentile(final_values, 5)),
        'var_1': float(np.percentile(final_values, 1)),
        'expected_shortfall_5': float(np.mean(final_values[final_values <= np.percentile(final_values, 5)]))
    }

    return metrics


def calculate_confidence_intervals(portfolio_values, confidence_levels=[0.05, 0.95]):
    """
    Calculate confidence intervals for portfolio values at each time step.

    Parameters:
    -----------
    portfolio_values : numpy.ndarray
        Portfolio values from Monte Carlo simulation
    confidence_levels : list
        Confidence levels for intervals (default: [0.05, 0.95] for 90% CI)

    Returns:
    --------
    dict
        Dictionary with confidence interval bounds for each time step
    """
    percentiles = {}
    for level in confidence_levels:
        percentiles[f'p{int(level*100)}'] = np.percentile(portfolio_values, level*100, axis=1)

    return percentiles


def get_simulation_summary(portfolio_values, initial_investment):
    """
    Generate a comprehensive summary of simulation results.

    Parameters:
    -----------
    portfolio_values : numpy.ndarray
        Portfolio values from Monte Carlo simulation
    initial_investment : float
        Initial investment amount

    Returns:
    --------
    dict
        Comprehensive simulation summary
    """
    risk_metrics = calculate_risk_metrics(portfolio_values)
    confidence_intervals = calculate_confidence_intervals(portfolio_values)

    final_values = portfolio_values[-1, :]
    total_return = (risk_metrics['mean_final_value'] - initial_investment) / initial_investment

    summary = {
        'initial_investment': initial_investment,
        'final_values': final_values,
        'risk_metrics': risk_metrics,
        'confidence_intervals': confidence_intervals,
        'total_return': total_return,
        'max_portfolio_value': float(np.max(final_values)),
        'min_portfolio_value': float(np.min(final_values)),
        'median_portfolio_value': float(np.median(final_values)),
        'probability_positive_return': float(np.mean(final_values > initial_investment))
    }

    return summary