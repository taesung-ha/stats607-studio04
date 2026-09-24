import numpy as np
import pytest

from bootstrap import bootstrap_sample, bootstrap_ci, r_squared

def test_bootstrap_sample_returns_correct_length():
    np.random.seed(42)

    data = np.array([1, 2, 3, 4, 5])

    result = bootstrap_sample(
        data,
        np.mean,
        n_bootstrap = 100
    )

    assert isinstance(result, np.ndarray)
    assert len(result) == 100

def test_bootstrap_sample_values_are_reasonable():
    np.random.seed(42)

    data = np.array([1, 2, 3, 4, 5])

    result = bootstrap_sample(
        data,
        np.mean,
        n_bootstrap=100
    )

    assert np.all(result >= 1)
    assert np.all(result <= 5)

def test_bootstrap_sample_default_n_bootstrap():
    np.random.seed(42)

    data = np.array([1, 2, 3])

    result = bootstrap_sample(
        data,
        np.mean
    )

    assert len(result) == 1000

def test_bootstrap_sample_empty_data():
    with pytest.raises(ValueError):
        bootstrap_sample(
            [],
            np.mean,
            n_bootstrap=10
        )


def test_bootstrap_sample_zero_bootstraps():
    data = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        bootstrap_sample(
            data,
            np.mean,
            n_bootstrap=0
        )


def test_bootstrap_sample_non_callable_stat():
    data = np.array([1, 2, 3])

    with pytest.raises(TypeError):
        bootstrap_sample(
            data,
            "mean",
            n_bootstrap=10
        )


def test_full_bootstrap_integration():
    np.random.seed(42)

    data = np.array([
        [1, 3],
        [2, 5],
        [3, 7],
        [4, 9],
        [5, 11]
    ])

    observed_r2 = r_squared(data)

    bootstrap_stats = bootstrap_sample(
        data,
        r_squared,
        n_bootstrap=200
    )

    lower, upper = bootstrap_ci(
        bootstrap_stats,
        alpha=0.05
    )

    assert observed_r2 == pytest.approx(1.0)
    assert len(bootstrap_stats) == 200
    assert 0 <= lower <= upper <= 1