import numpy as np
import matplotlib.pyplot as plt

from simulation import (
    monte_carlo_call_price,
    black_scholes_call_price
)


black_scholes_price = black_scholes_call_price(
    initial_price=100,
    strike_price=100,
    risk_free_rate=0.05,
    volatility=0.20,
    time_to_maturity=1
)

simulation_counts = [
    100,
    1_000,
    10_000,
    100_000,
    1_000_000
]

num_trials = 30

mean_errors = []
std_errors = []

for count in simulation_counts:
    errors = []

    for _ in range(num_trials):
        monte_carlo_price = monte_carlo_call_price(
            initial_price=100,
            strike_price=100,
            risk_free_rate=0.05,
            volatility=0.20,
            time_to_maturity=1,
            num_simulations=count
        )

        error = abs(
            monte_carlo_price - black_scholes_price
        )

        errors.append(error)

    mean_error = np.mean(errors)
    std_error = np.std(errors)

    mean_errors.append(mean_error)
    std_errors.append(std_error)

    print(
        f"Simulations: {count:>8} | "
        f"Mean Error: {mean_error:.6f} | "
        f"Std Dev: {std_error:.6f}"
    )


plt.errorbar(
    simulation_counts,
    mean_errors,
    yerr=std_errors,
    marker="o",
    capsize=4
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Number of Simulations")
plt.ylabel("Mean Absolute Pricing Error")
plt.title("Monte Carlo Convergence to Black-Scholes")

plt.grid(True)
plt.tight_layout()
plt.savefig(
    "results/convergence.png",
    dpi=300
)
plt.show()