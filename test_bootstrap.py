import numpy as np
import pytest

from bootstrap import bootstrap_sample, bootstrap_ci, r_squared


def mean_y(data):
    """Return the mean of the y column."""
    return np.mean(data[:, 1])


def test_bootstrap_sample_returns_correct_length():
    np.random.seed(42)
    data = np.array([
        [1, 3],
        [2, 5],
        [3, 7],
        [4, 9],
        [5, 11],
    ])

    result = bootstrap_sample(data, mean_y, n_bootstrap=100)

    assert isinstance(result, np.ndarray)
    assert len(result) == 100


def test_bootstrap_sample_values_are_within_y_range():
    np.random.seed(42)
    data = np.array([
        [1, 3],
        [2, 5],
        [3, 7],
        [4, 9],
        [5, 11],
    ])

    result = bootstrap_sample(data, mean_y, n_bootstrap=100)

    assert np.all(result >= 3)
    assert np.all(result <= 11)


def test_bootstrap_sample_default_n_bootstrap():
    np.random.seed(42)
    data = np.array([
        [1, 3],
        [2, 5],
        [3, 7],
    ])

    result = bootstrap_sample(data, mean_y)

    assert len(result) == 1000


def test_bootstrap_sample_empty_data():
    data = np.empty((0, 2))

    with pytest.raises(ValueError):
        bootstrap_sample(data, mean_y, n_bootstrap=10)


def test_bootstrap_sample_zero_bootstraps():
    data = np.array([
        [1, 3],
        [2, 5],
        [3, 7],
    ])

    with pytest.raises(ValueError):
        bootstrap_sample(data, mean_y, n_bootstrap=0)


def test_bootstrap_sample_non_callable_stat():
    data = np.array([
        [1, 3],
        [2, 5],
        [3, 7],
    ])

    with pytest.raises(TypeError):
        bootstrap_sample(data, "mean", n_bootstrap=10)


def test_full_bootstrap_integration():
    np.random.seed(42)
    data = np.array([
        [1, 3],
        [2, 5],
        [3, 7],
        [4, 9],
        [5, 11],
    ])

    observed_r2 = r_squared(data)
    bootstrap_stats = bootstrap_sample(data, r_squared, n_bootstrap=200)
    lower, upper = bootstrap_ci(bootstrap_stats, alpha=0.05)

    assert observed_r2 == pytest.approx(1.0)
    assert len(bootstrap_stats) == 200
    assert lower <= upper
    
def test_bootstrap_ci_returns_percentile_interval():
    stats = np.arange(1, 1001)

    lower, upper = bootstrap_ci(stats, alpha=0.05)

    assert lower == pytest.approx(25.975)
    assert upper == pytest.approx(975.025)


def test_bootstrap_ci_rejects_empty_input():
    with pytest.raises(ValueError):
        bootstrap_ci([])


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1])
def test_bootstrap_ci_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        bootstrap_ci([1, 2, 3], alpha=alpha)


def test_r_squared_is_one_for_perfect_linear_data():
    data = np.array([
        [1, 3],
        [2, 5],
        [3, 7],
        [4, 9],
    ])

    assert r_squared(data) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "data",
    [
        np.array([1, 2, 3]),          # 2차원 배열이 아님
        np.array([[1, 2, 3], [4, 5, 6]]),  # 열이 2개가 아님
        np.array([[1, 2]]),           # 행이 2개 미만
    ],
)
def test_r_squared_rejects_invalid_data_shape(data):
    with pytest.raises(ValueError):
        r_squared(data)