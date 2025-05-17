import numpy as np
import matplotlib.pyplot as plt

"""Simple RBC-style DSGE model simulation with impulse response plotting."""

# Model parameters
ALPHA = 0.36   # capital share
BETA = 0.96    # discount factor
DELTA = 0.08   # depreciation rate
RHO = 0.95     # persistence of technology shock
SIGMA = 0.02   # volatility of technology shock


def steady_state_capital():
    """Return the deterministic steady state capital level."""
    return ((ALPHA * BETA) / (1 / BETA - (1 - DELTA))) ** (1 / (1 - ALPHA))


def simulate(T=50, k0=1.0, z0=0.0, eps=None):
    """Simulate a very small DSGE economy.

    Parameters
    ----------
    T : int
        Number of periods to simulate.
    k0 : float
        Initial capital stock.
    z0 : float
        Initial technology shock in logs.
    eps : array-like or None
        Technology innovations for each period.  If ``None`` random innovations
        are drawn.  Length must be at least ``T``.

    Returns
    -------
    tuple of arrays
        Arrays for technology ``z``, capital ``k``, consumption ``c``, output
        ``y`` and investment ``i``.
    """
    if eps is None:
        eps = np.random.randn(T) * SIGMA
    else:
        eps = np.asarray(eps, dtype=float)
        if len(eps) < T:
            eps = np.pad(eps, (0, T - len(eps)))

    z = np.zeros(T)
    k = np.zeros(T)
    c = np.zeros(T)
    y = np.zeros(T)
    i = np.zeros(T)

    z[0] = z0
    k[0] = k0

    for t in range(T - 1):
        # technology shock follows AR(1)
        z[t + 1] = RHO * z[t] + eps[t + 1]

        # output
        y[t] = np.exp(z[t]) * k[t] ** ALPHA

        # investment derived from Euler equation in simple RBC
        i[t] = ALPHA * BETA * np.exp(z[t]) * k[t] ** ALPHA

        # consumption from resource constraint
        c[t] = y[t] - i[t]

        # capital accumulation
        k[t+1] = (1 - DELTA) * k[t] + i[t]

    # final period output and consumption
    y[-1] = np.exp(z[-1]) * k[-1] ** ALPHA
    i[-1] = ALPHA * BETA * np.exp(z[-1]) * k[-1] ** ALPHA
    c[-1] = y[-1] - i[-1]

    return z, k, c, y, i


def impulse_response(T=40, shock=SIGMA):
    """Compute impulse responses to a one-time technology shock."""
    k_ss = steady_state_capital()

    zero_eps = np.zeros(T)
    base = simulate(T=T, k0=k_ss, z0=0.0, eps=zero_eps)
    shock_res = simulate(T=T, k0=k_ss, z0=shock, eps=zero_eps)

    variables = ["z", "k", "c", "y", "i"]
    irf = {var: shock_res[idx] - base[idx] for idx, var in enumerate(variables)}
    return irf


def plot_irf(irf):
    """Plot impulse response functions for a dict produced by ``impulse_response``."""
    vars_order = ["z", "y", "c", "i", "k"]
    plt.figure(figsize=(10, 8))
    for idx, v in enumerate(vars_order, 1):
        plt.subplot(3, 2, idx)
        plt.plot(irf[v], marker="o")
        plt.title(v.upper())
        plt.axhline(0, color="black", lw=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage: plot impulse responses for 40 periods
    irf = impulse_response(T=40)
    plot_irf(irf)
