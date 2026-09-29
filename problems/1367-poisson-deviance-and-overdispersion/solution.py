import numpy as np

def poisson_deviance(y, mu):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)

    log_ratio = np.zeros_like(y)
    mask = y > 0
    log_ratio[mask] = np.log(y[mask] / mu[mask])

    deviance = 2 * np.sum(
        y * log_ratio - (y - mu)
    )

    return float(deviance)


def dispersion_ratio(y, mu, n_params):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)

    pearson_chi2 = np.sum((y - mu) ** 2 / mu)
    df = len(y) - n_params

    return float(pearson_chi2 / df)