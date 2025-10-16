import numpy as np
import pandas as pd
import pytest


def run_monte_carlo_simulation(initial_investment, mu, sigma, time_horizon, num_simulations, seed=None):
    """Run Monte Carlo simulation for portfolio value projection"""
    if seed is not None:
        np.random.seed(seed)

    dt = 1/252  # Trading days per year
    num_steps = int(time_horizon * 252)

    # Generate daily returns using lognormal distribution
    daily_returns = np.random.lognormal(
        mean=(mu - 0.5 * sigma**2) * dt,
        sigma=sigma * np.sqrt(dt),
        size=(num_steps, num_simulations)
    )

    # Prepend initial investment (day 0)
    daily_returns = np.vstack([np.ones((1, num_simulations)), daily_returns])

    # Calculate cumulative returns
    cumulative_returns = daily_returns.cumprod(axis=0)

    # Calculate portfolio values over time
    portfolio_values = initial_investment * cumulative_returns

    return portfolio_values


class TestMonteCarloSimulation:
    """Test suite for Monte Carlo simulation functionality"""

    def test_simulation_basic_properties(self):
        """Test basic properties of simulation output"""
        initial_investment = 10000
        mu = 0.08
        sigma = 0.15
        time_horizon = 5
        num_simulations = 100

        results = run_monte_carlo_simulation(
            initial_investment, mu, sigma, time_horizon, num_simulations
        )

        # Check dimensions
        expected_steps = int(time_horizon * 252) + 1  # +1 for initial day
        assert results.shape == (expected_steps, num_simulations)

        # Check initial values are correct
        assert np.allclose(results[0, :], initial_investment)

        # Check all values are positive
        assert np.all(results > 0)

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with fixed seed"""
        params = {
            'initial_investment': 10000,
            'mu': 0.08,
            'sigma': 0.15,
            'time_horizon': 3,
            'num_simulations': 50
        }

        # Run simulation twice with same seed
        result1 = run_monte_carlo_simulation(seed=42, **params)
        result2 = run_monte_carlo_simulation(seed=42, **params)

        # Results should be identical
        assert np.array_equal(result1, result2)

    def test_different_seeds_give_different_results(self):
        """Test that different seeds produce different results"""
        params = {
            'initial_investment': 10000,
            'mu': 0.08,
            'sigma': 0.15,
            'time_horizon': 2,
            'num_simulations': 30
        }

        result1 = run_monte_carlo_simulation(seed=1, **params)
        result2 = run_monte_carlo_simulation(seed=2, **params)

        # Results should be different (with very high probability)
        assert not np.array_equal(result1, result2)

    def test_expected_return_properties(self):
        """Test that higher expected returns lead to higher final values on average"""
        initial_investment = 10000
        sigma = 0.15
        time_horizon = 10
        num_simulations = 1000

        # Run with low and high expected returns
        low_mu_result = run_monte_carlo_simulation(
            initial_investment, 0.02, sigma, time_horizon, num_simulations, seed=42
        )
        high_mu_result = run_monte_carlo_simulation(
            initial_investment, 0.12, sigma, time_horizon, num_simulations, seed=42
        )

        # High expected return should give higher final values on average
        low_mu_final_avg = np.mean(low_mu_result[-1, :])
        high_mu_final_avg = np.mean(high_mu_result[-1, :])

        assert high_mu_final_avg > low_mu_final_avg

    def test_volatility_properties(self):
        """Test that higher volatility leads to wider distribution of outcomes"""
        initial_investment = 10000
        mu = 0.08
        time_horizon = 5
        num_simulations = 1000

        # Run with low and high volatility
        low_vol_result = run_monte_carlo_simulation(
            initial_investment, mu, 0.05, time_horizon, num_simulations, seed=42
        )
        high_vol_result = run_monte_carlo_simulation(
            initial_investment, mu, 0.25, time_horizon, num_simulations, seed=42
        )

        # High volatility should give wider distribution
        low_vol_std = np.std(low_vol_result[-1, :])
        high_vol_std = np.std(high_vol_result[-1, :])

        assert high_vol_std > low_vol_std

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test with minimal parameters
        result = run_monte_carlo_simulation(1000, 0.0, 0.0, 1, 10)
        assert result.shape == (253, 10)  # 252 trading days + 1 initial

        # Test with zero volatility (deterministic case)
        result_zero_vol = run_monte_carlo_simulation(10000, 0.08, 0.0, 2, 5, seed=42)
        # With zero volatility, all paths should be identical
        final_values = result_zero_vol[-1, :]
        assert np.allclose(final_values, final_values[0])


if __name__ == "__main__":
    pytest.main([__file__])