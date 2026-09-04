import pytest

from mcpricer.data import load_market_data


def test_columns_and_ordering():
    data = load_market_data()
    assert list(data.columns) == ["spot", "iv", "rate"]
    assert data.index.is_monotonic_increasing
    assert not data.isna().any().any()


def test_units_are_decimals_not_percentage_points():
    data = load_market_data()
    assert 0.05 < data["iv"].min() < data["iv"].max() < 1.0
    assert 0.0 <= data["rate"].max() < 0.2


def test_spot_is_a_plausible_index_level():
    data = load_market_data()
    assert 1000 < data["spot"].min() < data["spot"].max() < 20_000


def test_missing_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_market_data(data_dir=tmp_path)
