"""
Regression tests for simulation-correctness fixes that change backtest results
without raising when they regress: split processing, history adjustments,
slippage fill sizes, before_trading_start timing and zero order prices.
"""

import sqlite3
from types import SimpleNamespace

import pandas as pd
import pytest

from zipline.algorithm import TradingAlgorithm
from zipline.assets import Equity, ExchangeInfo
from zipline.data.data_portal import DataPortal
from zipline.data.history_loader import merge_adjustments
from zipline.errors import UnsupportedOrderParameters
from zipline.finance.order import Order
from zipline.finance.slippage import FixedSlippage, NoSlippage
from zipline.gens.sim_engine import BAR, BEFORE_TRADING_START_BAR, SESSION_START
from zipline.lib.adjustment import Float64Multiply
from zipline.utils.calendar_utils import get_calendar


class TestGetSplitsWindow:
    """``DataPortal.get_splits`` must report a split exactly once, on the
    first session at or after its effective date, even when that date is not
    a session. Otherwise the position's share count is never adjusted while
    the pricing data around it is, and the position appears to lose (or gain)
    the whole split ratio overnight.
    """

    SID = 1

    def _portal(self, effective_date):
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE SPLITS (sid INTEGER, ratio REAL, effective_date INTEGER)"
        )
        conn.execute(
            "INSERT INTO SPLITS VALUES (?, ?, ?)",
            (self.SID, 0.25, int(pd.Timestamp(effective_date).value // 10**9)),
        )
        portal = SimpleNamespace(
            trading_calendar=get_calendar("XNYS"),
            _adjustment_reader=SimpleNamespace(conn=conn),
            asset_finder=SimpleNamespace(retrieve_asset=lambda sid: sid),
        )
        portal._splits_lower_bound = lambda dt: DataPortal._splits_lower_bound(
            portal, dt
        )
        return portal

    def _sessions_reporting_split(self, portal, start, end):
        sessions = portal.trading_calendar.sessions_in_range(start, end)
        return [
            session
            for session in sessions
            if DataPortal.get_splits(portal, {self.SID}, session)
        ]

    @pytest.mark.parametrize(
        "effective_date, expected_session",
        [
            pytest.param("2024-01-10", "2024-01-10", id="on_session"),
            pytest.param("2024-01-13", "2024-01-16", id="weekend_and_holiday"),
            pytest.param("2024-03-29", "2024-04-01", id="exchange_holiday"),
        ],
    )
    def test_split_is_reported_once_on_first_session_at_or_after(
        self, effective_date, expected_session
    ):
        portal = self._portal(effective_date)

        reported = self._sessions_reporting_split(portal, "2024-01-02", "2024-04-30")

        assert reported == [pd.Timestamp(expected_session)]

    def test_split_ratio_is_returned(self):
        portal = self._portal("2024-01-13")

        splits = DataPortal.get_splits(
            portal, {self.SID}, pd.Timestamp("2024-01-16")
        )

        assert splits == [(self.SID, 0.25)]

    def test_split_for_other_assets_is_ignored(self):
        portal = self._portal("2024-01-10")

        assert DataPortal.get_splits(portal, {2}, pd.Timestamp("2024-01-10")) == []

    @pytest.mark.parametrize("tz", ["UTC", "US/Eastern", "Asia/Tokyo"])
    @pytest.mark.parametrize(
        "effective_date, expected_session",
        [
            pytest.param("2024-01-10", "2024-01-10", id="on_session"),
            pytest.param("2024-01-13", "2024-01-16", id="weekend_and_holiday"),
        ],
    )
    def test_aware_session_label_reports_the_split_once(
        self, tz, effective_date, expected_session
    ):
        """An aware ``dt`` names the session of its own wall-clock date. The
        upper bound used to come from its UTC instant while the lower bound
        came from that date, so east of UTC a split effective on a session
        fell outside every window and was never reported."""
        portal = self._portal(effective_date)
        sessions = portal.trading_calendar.sessions_in_range(
            "2024-01-02", "2024-02-29"
        )

        reported = [
            session
            for session in sessions
            if DataPortal.get_splits(portal, {self.SID}, session.tz_localize(tz))
        ]

        assert reported == [pd.Timestamp(expected_session)]


class TestMergeAdjustments:
    """Assets routinely share an ex-date. Two assets' adjustments landing on
    the same row of a multi-asset history window must both be kept; a plain
    ``dict.update`` kept only the last one, leaving the other asset's history
    unadjusted.
    """

    def test_adjustments_at_the_same_location_are_all_kept(self):
        a = Float64Multiply(0, 4, 0, 0, 0.5)
        b = Float64Multiply(0, 4, 1, 1, 0.25)
        c = Float64Multiply(0, 6, 1, 1, 0.9)

        merged = {}
        merge_adjustments(merged, {4: [a]})
        merge_adjustments(merged, {4: [b], 6: [c]})

        assert merged == {4: [a, b], 6: [c]}

    def test_input_lists_are_not_mutated(self):
        a = Float64Multiply(0, 4, 0, 0, 0.5)
        b = Float64Multiply(0, 4, 1, 1, 0.25)
        first = {4: [a]}

        merged = merge_adjustments({}, first)
        merge_adjustments(merged, {4: [b]})

        assert first == {4: [a]}


class TestSlippageFillsOnlyTheOpenAmount:
    """A partially filled order must only fill what is still open. Filling
    ``order.amount`` again overfills the position by the amount already
    filled.
    """

    @pytest.fixture
    def partially_filled_order(self):
        order = Order(
            dt=pd.Timestamp("2024-01-10 21:00", tz="UTC"),
            asset=Equity(1, exchange_info=ExchangeInfo("TEST", "TEST", "US")),
            amount=100,
        )
        order.filled = 40
        return order

    @pytest.fixture
    def data(self):
        return SimpleNamespace(current=lambda asset, field: 10.0)

    def test_no_slippage(self, data, partially_filled_order):
        _, amount = NoSlippage().process_order(data, partially_filled_order)

        assert amount == 60

    def test_fixed_slippage(self, data, partially_filled_order):
        _, amount = FixedSlippage(spread=0.02).process_order(
            data, partially_filled_order
        )

        assert amount == 60


class TestBeforeTradingStartPrecedesTheSession:
    """``before_trading_start`` - and every pipeline with it - must run before
    a session's first bar. The 08:45 US/Eastern default lands after the close
    on Asian calendars, where the clock dropped the hook, and mid-session on
    European ones, where the clock ran hours of bars against the previous
    session's pipeline output first.
    """

    def _events_by_session(self, calendar_name, data_frequency):
        calendar = get_calendar(calendar_name)
        sessions = calendar.sessions_in_range("2024-03-04", "2024-03-08")
        algo = SimpleNamespace(
            trading_calendar=calendar,
            sim_params=SimpleNamespace(
                sessions=sessions,
                data_frequency=data_frequency,
                emission_rate="daily",
            ),
        )
        algo._before_trading_start_minutes = lambda *args: (
            TradingAlgorithm._before_trading_start_minutes(algo, *args)
        )

        by_session = []
        for dt, action in TradingAlgorithm._create_clock(algo):
            if action == SESSION_START:
                by_session.append([])
            elif action in (BAR, BEFORE_TRADING_START_BAR):
                by_session[-1].append((dt, action))
        return by_session

    @pytest.mark.parametrize("data_frequency", ["minute", "daily"])
    @pytest.mark.parametrize("calendar_name", ["XNYS", "XLON", "XTKS"])
    def test_fires_once_before_the_first_bar(self, calendar_name, data_frequency):
        by_session = self._events_by_session(calendar_name, data_frequency)

        assert len(by_session) == 5
        for events in by_session:
            actions = [action for _, action in events]
            assert actions.count(BEFORE_TRADING_START_BAR) == 1
            assert actions[0] == BEFORE_TRADING_START_BAR

    def test_us_equities_keep_0845_eastern(self):
        for events in self._events_by_session("XNYS", "minute"):
            bts = events[0][0].tz_convert("US/Eastern")
            assert (bts.hour, bts.minute) == (8, 45)


class TestZeroOrderPrices:
    """order() once read a price of 0 as "no price" and placed a market order.
    It now builds the order at that price, where a buy limit at 0 never fills
    and a sell limit at 0 fills at any price, so 0 must be refused.
    """

    @staticmethod
    def _validate(limit_price=None, stop_price=None):
        algo = SimpleNamespace(initialized=True, trading_controls=[])
        TradingAlgorithm.validate_order_params(
            algo, None, 100, limit_price, stop_price, None
        )

    @pytest.mark.parametrize(
        "prices",
        [
            {"limit_price": 0},
            {"stop_price": 0.0},
            {"limit_price": 0, "stop_price": 5.0},
        ],
    )
    def test_zero_price_is_refused(self, prices):
        with pytest.raises(UnsupportedOrderParameters, match="pass None"):
            self._validate(**prices)

    def test_none_and_positive_prices_are_accepted(self):
        self._validate()
        self._validate(limit_price=0.01, stop_price=5.0)
