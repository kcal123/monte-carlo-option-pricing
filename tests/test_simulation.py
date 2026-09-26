import numpy as np

from src.simulation import (
    simulate_terminal_price,
    european_call_payoff,
    monte_carlo_call_price,
    black_scholes_call_price,
    simulate_terminal_price_antithetic,
    monte_carlo_call_price_antithetic
)


def test_call_payoff():
    terminal_prices = np.array([80, 100, 120, 150])

    payoffs = european_call_payoff(
        terminal_prices,
        strike_price=100
    )

    expected = np.array([0, 0, 20, 50])

    assert np.array_equal(payoffs, expected)


def test_terminal_prices_are_positive():
    prices = simulate_terminal_price(
        initial_price=100,
        risk_free_rate=0.05,
        volatility=0.20,
        time_to_maturity=1,
        num_simulations=10_000,
        seed=42
    )

    assert np.all(prices > 0)


def test_reproducible_with_seed():
    price_1 = monte_carlo_call_price(
        initial_price=100,
        strike_price=100,
        risk_free_rate=0.05,
        volatility=0.20,
        time_to_maturity=1,
        num_simulations=100_000,
        seed=42
    )

    price_2 = monte_carlo_call_price(
        initial_price=100,
        strike_price=100,
        risk_free_rate=0.05,
        volatility=0.20,
        time_to_maturity=1,
        num_simulations=100_000,
        seed=42
    )

    assert price_1 == price_2


def test_black_scholes_known_value():
    price = black_scholes_call_price(
        initial_price=100,
        strike_price=100,
        risk_free_rate=0.05,
        volatility=0.20,
        time_to_maturity=1
    )

    assert abs(price - 10.45058357) < 1e-6


def test_monte_carlo_close_to_black_scholes():
    monte_carlo_price = monte_carlo_call_price(
        initial_price=100,
        strike_price=100,
        risk_free_rate=0.05,
        volatility=0.20,
        time_to_maturity=1,
        num_simulations=200_000,
        seed=42
    )

    black_scholes_price = black_scholes_call_price(
        initial_price=100,
        strike_price=100,
        risk_free_rate=0.05,
        volatility=0.20,
        time_to_maturity=1
    )

    assert abs(
        monte_carlo_price - black_scholes_price
    ) < 0.05

def test_antithetic_requires_even_simulations():
    try:
        simulate_terminal_price_antithetic(
            initial_price=100,
            risk_free_rate=0.05,
            volatility=0.20,
            time_to_maturity=1,
            num_simulations=999,
            seed=42
        )

        assert False

    except ValueError:
        assert True


def test_antithetic_reproducible_with_seed():
    price_1 = monte_carlo_call_price_antithetic(
        initial_price=100,
        strike_price=100,
        risk_free_rate=0.05,
        volatility=0.20,
        time_to_maturity=1,
        num_simulations=100_000,
        seed=42
    )

    price_2 = monte_carlo_call_price_antithetic(
        initial_price=100,
        strike_price=100,
        risk_free_rate=0.05,
        volatility=0.20,
        time_to_maturity=1,
        num_simulations=100_000,
        seed=42
    )

    assert price_1 == price_2


def test_antithetic_close_to_black_scholes():
    antithetic_price = monte_carlo_call_price_antithetic(
        initial_price=100,
        strike_price=100,
        risk_free_rate=0.05,
        volatility=0.20,
        time_to_maturity=1,
        num_simulations=200_000,
        seed=42
    )

    black_scholes_price = black_scholes_call_price(
        initial_price=100,
        strike_price=100,
        risk_free_rate=0.05,
        volatility=0.20,
        time_to_maturity=1
    )

    assert abs(
        antithetic_price - black_scholes_price
    ) < 0.05