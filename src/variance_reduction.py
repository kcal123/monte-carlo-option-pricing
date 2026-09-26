import numpy as np
import matplotlib.pyplot as plt

from simulation import (
    monte_carlo_call_price,
    monte_carlo_call_price_antithetic,
    black_scholes_call_price
)


initial_price = 100
strike_price = 100
risk_free_rate = 0.05
volatility = 0.20
time_to_maturity = 1

simulation_counts = [
    100,
    1_000,
    10_000,
    100_000,
    1_000_000
]

num_trials = 30

black_scholes_price = black_scholes_call_price(
    initial_price,
    strike_price,
    risk_free_rate,
    volatility,
    time_to_maturity
)

standard_mean_errors = []
antithetic_mean_errors = []

for count in simulation_counts:
    standard_errors = []
    antithetic_errors = []

    for trial in range(num_trials):
        standard_price = monte_carlo_call_price(
            initial_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_maturity,
            count,
            seed=trial
        )

        antithetic_price = monte_carlo_call_price_antithetic(
            initial_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_maturity,
            count,
            seed=trial
        )

        standard_errors.append(
            abs(standard_price - black_scholes_price)
        )

        antithetic_errors.append(
            abs(antithetic_price - black_scholes_price)
        )

    standard_mean_error = np.mean(standard_errors)
    antithetic_mean_error = np.mean(antithetic_errors)

    standard_mean_errors.append(standard_mean_error)
    antithetic_mean_errors.append(antithetic_mean_error)

    print(
        f"Simulations: {count:>8} | "
        f"Standard Error: {standard_mean_error:.6f} | "
        f"Antithetic Error: {antithetic_mean_error:.6f}"
    )


plt.plot(
    simulation_counts,
    standard_mean_errors,
    marker="o",
    label="Standard Monte Carlo"
)

plt.plot(
    simulation_counts,
    antithetic_mean_errors,
    marker="o",
    label="Antithetic Variates"
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Number of Simulations")
plt.ylabel("Mean Absolute Pricing Error")
plt.title("Standard vs. Antithetic Monte Carlo")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/variance_reduction.png",
    dpi=300
)

plt.show()