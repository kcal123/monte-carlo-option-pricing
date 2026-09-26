import numpy as np
import math


def simulate_terminal_price(
    initial_price,
    risk_free_rate,
    volatility,
    time_to_maturity,
    num_simulations,
    seed=None
):
    rng = np.random.default_rng(seed)

    random_shocks = rng.standard_normal(num_simulations)

    terminal_prices = initial_price * np.exp(
        (risk_free_rate - 0.5 * volatility**2) * time_to_maturity
        + volatility * np.sqrt(time_to_maturity) * random_shocks
    )

    return terminal_prices


def simulate_terminal_price_antithetic(
    initial_price,
    risk_free_rate,
    volatility,
    time_to_maturity,
    num_simulations,
    seed=None
):
    if num_simulations % 2 != 0:
        raise ValueError(
            "num_simulations must be even for antithetic sampling"
        )

    rng = np.random.default_rng(seed)

    half = num_simulations // 2

    random_shocks = rng.standard_normal(half)

    antithetic_shocks = np.concatenate(
        [random_shocks, -random_shocks]
    )

    terminal_prices = initial_price * np.exp(
        (risk_free_rate - 0.5 * volatility**2)
        * time_to_maturity
        + volatility
        * np.sqrt(time_to_maturity)
        * antithetic_shocks
    )

    return terminal_prices


def monte_carlo_call_price_antithetic(
    initial_price,
    strike_price,
    risk_free_rate,
    volatility,
    time_to_maturity,
    num_simulations,
    seed=None
):
    terminal_prices = simulate_terminal_price_antithetic(
        initial_price,
        risk_free_rate,
        volatility,
        time_to_maturity,
        num_simulations,
        seed=seed
    )

    payoffs = european_call_payoff(
        terminal_prices,
        strike_price
    )

    expected_payoff = np.mean(payoffs)

    present_value = (
        np.exp(-risk_free_rate * time_to_maturity)
        * expected_payoff
    )

    return present_value


def european_call_payoff(terminal_prices, strike_price):
    return np.maximum(terminal_prices - strike_price, 0)


def monte_carlo_call_price(
    initial_price,
    strike_price,
    risk_free_rate,
    volatility,
    time_to_maturity,
    num_simulations, 
    seed=None
):
    terminal_prices = simulate_terminal_price(
        initial_price,
        risk_free_rate,
        volatility,
        time_to_maturity,
        num_simulations,
        seed=seed
    )

    payoffs = european_call_payoff(
        terminal_prices,
        strike_price
    )

    expected_payoff = np.mean(payoffs)

    present_value = (
        np.exp(-risk_free_rate * time_to_maturity)
        * expected_payoff
    )

    return present_value


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_scholes_call_price(
    initial_price,
    strike_price,
    risk_free_rate,
    volatility,
    time_to_maturity
):
    d1 = (
        math.log(initial_price / strike_price)
        + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity
    ) / (
        volatility * math.sqrt(time_to_maturity)
    )

    d2 = d1 - volatility * math.sqrt(time_to_maturity)

    call_price = (
        initial_price * normal_cdf(d1)
        - strike_price
        * math.exp(-risk_free_rate * time_to_maturity)
        * normal_cdf(d2)
    )

    return call_price


