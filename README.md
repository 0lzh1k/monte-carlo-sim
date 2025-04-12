# Monte Carlo Simulation for Financial Risk Assessment

A web-based application built with Streamlit that performs Monte Carlo simulation to model and visualize financial risk and return. Users can input financial parameters and see potential investment outcomes over a specified time horizon.

## Features

- **Interactive UI**: A user-friendly interface built with Streamlit to input financial parameters.
- **Real Market Data Integration**: Optional integration with Yahoo Finance to fetch real stock data and automatically calculate mu and sigma parameters.
- **Monte Carlo Simulation**: Simulates thousands of potential portfolio outcomes based on user-defined or real market parameters.
- **Interactive Visualizations**: 
  - Historical stock price charts (when using real data)
  - An interactive Plotly chart showing all individual simulation paths.
  - A cleaner plot showing the median outcome and the interquartile range (25th-75th percentile).
  - A histogram of the final portfolio values.
- **Risk Metrics**: Calculates and displays key risk metrics, including:
  - Mean Final Portfolio Value
  - Standard Deviation of Final Value
  - Value at Risk (VaR) at 5%

## Mathematical Background

### Geometric Brownian Motion

The simulation uses Geometric Brownian Motion (GBM) to model stock price movements:

$$dS = \mu S dt + \sigma S dW$$

Where:
- $S$: Stock price
- $\mu$: Expected annual return (drift)
- $\sigma$: Annual volatility
- $dt$: Time increment
- $dW$: Wiener process increment

### Daily Returns Distribution

Daily returns follow a lognormal distribution:

$$r_{daily} \sim \log\mathcal{N}\left((\mu - \frac{1}{2}\sigma^2)\frac{dt}{252}, \sigma\sqrt{\frac{dt}{252}}\right)$$

Where 252 represents trading days per year.

### Risk Metrics

**Value at Risk (VaR) at 5%**: The portfolio value below which only 5% of outcomes fall.

**Expected Shortfall (ES)**: The average loss given that the loss exceeds VaR.

## Example Scenarios

### Conservative Investment (Low Risk)
- Initial Investment: $10,000
- Expected Return (μ): 6% (0.06)
- Volatility (σ): 10% (0.10)
- Time Horizon: 10 years
- Simulations: 1,000

**Expected Outcome**: Final portfolio value around $16,000-$18,000 with 90% confidence.

### Aggressive Investment (High Risk)
- Initial Investment: $10,000
- Expected Return (μ): 12% (0.12)
- Volatility (σ): 25% (0.25)
- Time Horizon: 10 years
- Simulations: 1,000

**Expected Outcome**: Final portfolio value between $20,000-$80,000 with 90% confidence.

### Real Market Data Example (AAPL)
Using historical Apple Inc. data:
- Calculated μ: ~15% annual return
- Calculated σ: ~25% annual volatility
- Time Horizon: 5 years
- Simulations: 2,000

## Sample Output Visualization

### Scenario Comparison Chart
```
Conservative (μ=6%, σ=10%)    Aggressive (μ=12%, σ=25%)
    ▲                                ▲
$18K|     ████████                $80K|     ████████
$16K|   ████████████              $60K|   ████████████
$14K| ████████████████            $40K| ████████████████
$12K|█████████████████████        $20K|█████████████████████
$10K|█████████████████████████      $0|█████████████████████████
     └─────────────────────          └─────────────────────
         10 Years                        10 Years
```

### Risk Metrics Dashboard
```
┌─────────────────────────────────────┐
│          RISK METRICS               │
├─────────────────────────────────────┤
│ Mean Final Value:     $31,450       │
│ Standard Deviation:   $12,800       │
│ VaR (5%):            $18,200        │
│ Expected Shortfall:   $15,800       │
│ Probability > Initial: 85%          │
└─────────────────────────────────────┘
```

## Installation

1. **Clone the repository:**
   ```bash
   cd monte_carlo_fin_risk
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   ```

   **Activate the virtual environment:**

   - **Linux/macOS:**
     ```bash
     source venv/bin/activate
     ```

   - **Windows (Command Prompt):**
     ```bash
     venv\Scripts\activate
     ```

   - **Windows (PowerShell):**
     ```bash
     venv\Scripts\Activate.ps1
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:

```bash
streamlit run app.py
```

The application will open in your default web browser. You can use it in two modes:

### Manual Mode (Default)
- Adjust the financial parameters manually in the sidebar
- Set your own expected return (μ) and volatility (σ) values
- Click "Run Simulation" to see results

### Real Data Mode
1. Enter a valid stock ticker (e.g., AAPL, GOOGL, TSLA)
2. Check "Use Real Market Data" 
3. The app will automatically fetch 2 years of historical data
4. Calculated μ and σ values will be displayed and used as defaults
5. You can still adjust these values manually if desired
6. Click "Run Simulation" to see results based on real market data

## Project Structure

```
.
├── app.py              # The main Streamlit application file
├── simulation.py       # Monte Carlo simulation logic and risk calculations
├── test_simulation.py  # Unit tests for simulation functionality
├── requirements.txt    # The list of required Python packages
└── README.md           # This file
```

## Running Tests

To run the test suite:

```bash
pytest test_simulation.py -v
```

Or run specific tests:

```bash
pytest test_simulation.py::TestMonteCarloSimulation::test_reproducibility_with_seed -v
```
