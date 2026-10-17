"""
Regression tests for data-layer and bookkeeping fixes: daily price
forward-fill without a first trading day, asset db version checks, the
last-traded lookup, quandl tar extraction, %-style logging, and the lazily
built position exposure Series.
"""

import io
import logging
import tarfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa

from zipline.assets import AssetDBWriter, Equity, ExchangeInfo
from zipline.assets.asset_db_schema import ASSET_DB_VERSION, version_info
from zipline.data.bar_reader import NoDataOnDate
from zipline.data.bcolz_daily_bars import BcolzDailyBarReader
from zipline.data.bcolz_minute_bars import convert_cols
from zipline.data.bundles import quandl
from zipline.data.data_portal import DataPortal
from zipline.errors import AssetDBVersionError
from zipline.finance._finance_ext import (
    PositionStats,
    calculate_position_tracker_stats,
)
from zipline.finance.position import Position
from zipline.utils.calendar_utils import get_calendar

EXCHANGE = ExchangeInfo("TEST", "TEST", "US")


class TestDailyPriceForwardFill:
    """``DataPortal`` accepts ``first_trading_day=None``; the daily price
    forward-fill must then walk back until the reader runs out of data, not
    compare a Timestamp against None.
    """

    def _portal(self, closes):
        def get_value(asset, dt, field):
            try:
                return closes[dt]
            except KeyError:
                raise NoDataOnDate(dt)

        return SimpleNamespace(
            trading_calendar=get_calendar("XNYS"),
            _first_trading_day=None,
            _get_pricing_reader=lambda freq: SimpleNamespace(get_value=get_value),
            get_adjusted_value=lambda asset, column, found_dt, dt, freq, spot_value: (
                spot_value
            ),
        )

    def test_all_nan_closes_return_nan(self):
        sessions = get_calendar("XNYS").sessions_in_range("2024-01-02", "2024-01-10")
        portal = self._portal({s: np.nan for s in sessions})

        price = DataPortal._get_daily_spot_value(portal, 1, "price", sessions[-1])

        assert np.isnan(price)

    def test_forward_fills_last_close(self):
        sessions = get_calendar("XNYS").sessions_in_range("2024-01-02", "2024-01-10")
        closes = {s: np.nan for s in sessions}
        closes[sessions[2]] = 12.5
        portal = self._portal(closes)

        price = DataPortal._get_daily_spot_value(portal, 1, "price", sessions[-1])

        assert price == 12.5


class TestAssetDBInit:
    """A db from an older ``ASSET_DB_VERSION`` that lacks a newer table must
    raise ``AssetDBVersionError``, not be treated as a new db (whose second
    ``version_info`` row then violates ``CHECK(id <= 1)``).
    """

    def _version(self, engine):
        with engine.connect() as conn:
            return conn.execute(sa.select(version_info.c.version)).scalar()

    def test_new_db_writes_version(self):
        engine = sa.create_engine("sqlite://")
        AssetDBWriter(engine).init_db()

        assert self._version(engine) == ASSET_DB_VERSION

    def test_reopening_current_db_passes(self):
        engine = sa.create_engine("sqlite://")
        AssetDBWriter(engine).init_db()
        AssetDBWriter(engine).init_db()

        assert self._version(engine) == ASSET_DB_VERSION

    def test_older_db_missing_a_table_raises_version_error(self):
        engine = sa.create_engine("sqlite://")
        AssetDBWriter(engine).init_db()
        with engine.begin() as conn:
            conn.execute(version_info.update().values(version=ASSET_DB_VERSION - 1))
            conn.execute(sa.text("DROP TABLE equity_supplementary_mappings"))

        with pytest.raises(AssetDBVersionError):
            AssetDBWriter(engine).init_db()


def _walk_back_last_traded(reader, asset, day):
    """The pre-vectorization implementation: one session per iteration."""
    from zipline.data.bar_reader import NoDataAfterDate, NoDataBeforeDate

    volumes = reader._spot_col("volume")
    search_day = day
    while True:
        try:
            ix = reader.sid_day_index(asset, search_day)
        except NoDataBeforeDate:
            return pd.NaT
        except NoDataAfterDate:
            prev_day_ix = reader.sessions.get_loc(search_day) - 1
            if prev_day_ix > -1:
                search_day = reader.sessions[prev_day_ix]
            else:
                return pd.NaT
            continue
        except NoDataOnDate:
            return pd.NaT
        if volumes[ix] != 0:
            return search_day
        prev_day_ix = reader.sessions.get_loc(search_day) - 1
        if prev_day_ix > -1:
            search_day = reader.sessions[prev_day_ix]
        else:
            return pd.NaT


class TestGetLastTradedDt:
    """The vectorized ``get_last_traded_dt`` must agree with the per-session
    walk it replaced, including zero-volume runs, days after the asset's last
    row, and an earlier asset's rows sitting right before this one's.
    """

    @pytest.fixture
    def reader(self):
        sessions = get_calendar("XNYS").sessions_in_range("2024-01-02", "2024-02-14")
        # sid 2: rows 0-6 = sessions 0-6, all traded.
        # sid 1: rows 7-21 = sessions 5-19, ending in a zero-volume run.
        volumes = np.array(
            [9] * 7 + [0, 0, 100, 0, 0, 50, 0, 0, 0, 10, 0, 0, 0, 0, 0],
            dtype=np.uint32,
        )
        reader = SimpleNamespace(
            sessions=sessions,
            _first_rows={2: 0, 1: 7},
            _last_rows={2: 6, 1: 21},
            _calendar_offsets={2: 0, 1: 5},
            _spot_col=lambda colname: volumes,
        )
        reader.sid_day_index = lambda sid, day: BcolzDailyBarReader.sid_day_index(
            reader, sid, day
        )
        return reader

    @pytest.mark.parametrize("sid", [1, 2])
    def test_matches_per_session_walk(self, reader, sid):
        for day in reader.sessions:
            expected = _walk_back_last_traded(reader, sid, day)
            actual = BcolzDailyBarReader.get_last_traded_dt(reader, sid, day)
            assert (pd.isnull(expected) and pd.isnull(actual)) or expected == actual, day

    def test_after_last_row_finds_last_traded_session(self, reader):
        actual = BcolzDailyBarReader.get_last_traded_dt(
            reader, 1, reader.sessions[-1]
        )

        assert actual == reader.sessions[5 + 9]

    def test_non_session_day_is_nat(self, reader):
        saturday = pd.Timestamp("2024-01-13")

        assert pd.isnull(BcolzDailyBarReader.get_last_traded_dt(reader, 1, saturday))


class TestQuandlTarExtraction:
    """A member written through a symlink created earlier in the same archive
    must not land outside the output directory.
    """

    def test_symlink_cannot_redirect_a_later_member(self, tmp_path, monkeypatch):
        outside = tmp_path / "outside"
        outside.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            link = tarfile.TarInfo("link")
            link.type = tarfile.SYMTYPE
            link.linkname = str(outside)
            tar.addfile(link)
            payload = b"escaped"
            member = tarfile.TarInfo("link/escaped.txt")
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))
        buf.seek(0)
        monkeypatch.setattr(quandl, "download_without_progress", lambda url: buf)

        with pytest.raises((tarfile.TarError, ValueError)):
            quandl.quantopian_quandl_bundle(
                environ={},
                asset_db_writer=None,
                minute_bar_writer=None,
                daily_bar_writer=None,
                adjustment_writer=None,
                calendar=None,
                start_session=None,
                end_session=None,
                cache=None,
                show_progress=False,
                output_dir=str(output_dir),
            )

        assert not (outside / "escaped.txt").exists()


def test_convert_cols_overflow_warning_is_formatted(caplog):
    """stdlib logging formats with %-style args; a ``{}`` template fails with
    "not all arguments converted" and the data-loss warning is never shown.
    """
    cols = {name: np.array([1.0, 1e9]) for name in ("open", "high", "low", "close")}
    cols["volume"] = np.array([1.0, 1.0])

    with caplog.at_level(logging.WARNING):
        convert_cols(cols, 1000, sid=7, invalid_data_behavior="warn")

    messages = [record.getMessage() for record in caplog.records]
    assert any("sid=7, col=open" in message for message in messages)


class TestPositionExposureSeries:
    """``position_exposure_series`` is built on first access; it must still
    reflect the latest stats update, including a shrinking position count.
    """

    def test_tracks_latest_update(self):
        a = Equity(1, exchange_info=EXCHANGE)
        b = Equity(2, exchange_info=EXCHANGE)
        positions = {
            a: Position(a, amount=10, last_sale_price=5.0),
            b: Position(b, amount=-2, last_sale_price=3.0),
        }
        stats = PositionStats.new()

        calculate_position_tracker_stats(positions, stats)
        assert stats.position_exposure_series.to_dict() == {1: 50.0, 2: -6.0}

        del positions[b]
        calculate_position_tracker_stats(positions, stats)
        assert stats.position_exposure_series.to_dict() == {1: 50.0}
        np.testing.assert_array_equal(stats.position_exposure_array, [50.0])
