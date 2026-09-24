import numpy as np
def bootstrap_sample(data, compute_stat, n_bootstrap=1000):
    """
    Generate the bootstrap distribution of a statistic

    Parameters
    ----------
    data : array-like
        original sample (for regression: 2D array with columns [x, y])

    compute_stat : callable
        function that computes a univariate statistic from data
    
    n_bootstrap : int, default 1000
        number of bootstrap replicates to generate

    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics, length n_bootstrap

    Raises
    ------
    ValueError
        If data is empty, n_bootstrap < 1, or data has wrong shape
    TypeError
        If compute_stat is not callable
    

    Example
    -------
    TBA

    """
    if not callable(compute_stat):
        raise TypeError("compute_stat must be callable")
    
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    
    data = np.asarray(data)
    
    if data.size == 0:
        raise ValueError("data must not be empty")
    
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("data must have shape (n, 2)")
    
    n = data.shape[0]
    bootstrap_stats = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        resampled_data = data[indices]
        bootstrap_stats[i] = compute_stat(resampled_data)
        
    return bootstrap_stats

def bootstrap_ci(bootstrap_stats, alpha=0.05):
    """
    Calculate a CI from bootstrap distribution

    Parameters
    ----------
    bootstrap_stats : numpy.ndarray
        bootstrap statistics from bootstrap_sample(...)

    alpha : float, default 0.05
        significance level

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of the CI

    Raises
    ------
    ValueError
        If alpha not in (0, 1) or if bootstrap_stats is empty
    
    Example
    -------
    TBA

    """
    bootstrap_stats = np.asarray(bootstrap_stats)

    if bootstrap_stats.size == 0:
        raise ValueError("bootstrap_stats cannot be empty")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    lower = np.percentile(bootstrap_stats, lower_percentile)
    upper = np.percentile(bootstrap_stats, upper_percentile)

    return (lower, upper)


def r_squared(data):
    """
    Calculate R^2 from a linear regression

    Parameters
    ----------
    data : array-like, shape (n, 2)
        Data with columns [x, y]

    Returns
    -------
    float
        R-squared value between 0 and 1

    Raises
    ------
    ValueError
        If data doesn't have exactly 2 columns or < 2 rows
    """
    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise ValueError("data must be a 2D array")

    if data.shape[1] != 2:
        raise ValueError("data must have exactly 2 columns")

    if data.shape[0] < 2:
        raise ValueError("data must have at least 2 rows")

    x = data[:, 0]
    y = data[:, 1]

    X = np.column_stack((np.ones(len(x)), x))

    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]

    y_pred = X @ coefficients

    sse = np.sum((y - y_pred) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)

    if np.isclose(sst, 0):
        return 1.0 if np.isclose(sse, 0) else 0.0

    r2 = 1 - sse / sst

    return float(r2)
