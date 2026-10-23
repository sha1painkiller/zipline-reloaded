#
# Copyright 2017 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for dividend accrual and payment in finance.ledger.

These cover the pay dates that a bundle can hand us but that do not line up
with a trading session. Matching pay dates by equality used to drop those
payments entirely: the dividend was earned, filed under a key no session
would ever equal, and never paid. The position was still marked down when
the price went ex, so the portfolio lost the dividend outright.
"""

import sqlite3
from collections import namedtuple
from types import SimpleNamespace

import pandas as pd
import pytest

from zipline.assets import Equity, ExchangeInfo
from zipline.data.adjustments import SQLiteAdjustmentReader
from zipline.finance.ledger import Ledger, PositionTracker
from zipline.finance.metrics import load as load_metrics_set
from zipline.finance.metrics.tracker import MetricsTracker
from zipline.utils.calendar_utils import get_calendar, session_before

Dividend = namedtuple("Dividend", ["asset", "amount", "pay_date"])
StockDividend = namedtuple(
    "StockDividend",
    ["asset", "payment_asset", "ratio", "pay_date"],
)

# 2015-06-06 and 2015-06-07 are a Saturday and a Sunday, so the gap between
# 06-05 and 06-08 gives us a pay date that is not a session.
SESSIONS = pd.DatetimeIndex(
    [
        "2015-06-01",
        "2015-06-02",
        "2015-06-03",
        "2015-06-04",
        "2015-06-05",
        "2015-06-08",
        "2015-06-09",
    ],
    tz="UTC",
)
EX_DATE = SESSIONS[0]

SHARES = 100
AMOUNT_PER_SHARE = 0.50
EXPECTED_CASH = SHARES * AMOUNT_PER_SHARE


def make_asset(sid):
    return Equity(
        sid,
        ExchangeInfo("NYSE", "NYSE", "US"),
        symbol="A%d" % sid,
        start_date=pd.Timestamp("2000-01-01").value,
        end_date=pd.Timestamp("2030-01-01").value,
    )


def make_tracker(asset, amount=SHARES):
    tracker = PositionTracker("daily")
    tracker.update_position(asset, amount=amount, last_sale_price=100.0)
    return tracker


def run_sessions(tracker):
    """Pay dividends across every session, returning (total cash, session)."""
    total = 0.0
    paid_on = None
    for session in SESSIONS:
        cash = tracker.pay_dividends(session)
        if cash:
            total += cash
            paid_on = session
    return total, paid_on


@pytest.mark.parametrize(
    "pay_date, expected_session",
    [
        # A pay date the bundle left blank falls back to the ex date, so the
        # cash lands on the session where the price goes ex.
        pytest.param(pd.NaT, EX_DATE, id="missing_pay_date"),
        # A pay date that is a session is paid on that session, unchanged.
        pytest.param(SESSIONS[3], SESSIONS[3], id="pay_date_on_session"),
        # A weekend pay date is paid on the next session rather than dropped.
        pytest.param(
            pd.Timestamp("2015-06-06", tz="UTC"), SESSIONS[5], id="pay_date_on_weekend"
        ),
        # So is a pay date that predates the whole window.
        pytest.param(
            pd.Timestamp("2015-05-29", tz="UTC"), SESSIONS[0], id="pay_date_in_past"
        ),
    ],
)
def test_cash_dividend_is_paid(pay_date, expected_session):
    asset = make_asset(1)
    tracker = make_tracker(asset)
    tracker.earn_dividends(
        [Dividend(asset, amount=AMOUNT_PER_SHARE, pay_date=pay_date)],
        [],
        ex_date=EX_DATE,
    )

    total, paid_on = run_sessions(tracker)

    assert total == EXPECTED_CASH
    assert paid_on == expected_session
    # The payment is consumed, not left to be paid a second time.
    assert not tracker._unpaid_dividends


def test_cash_dividend_is_not_paid_before_its_pay_date():
    asset = make_asset(1)
    tracker = make_tracker(asset)
    tracker.earn_dividends(
        [Dividend(asset, amount=AMOUNT_PER_SHARE, pay_date=SESSIONS[3])],
        [],
        ex_date=EX_DATE,
    )

    for session in SESSIONS[:3]:
        assert tracker.pay_dividends(session) == 0.0

    assert tracker.pay_dividends(SESSIONS[3]) == EXPECTED_CASH


def test_short_position_owes_the_dividend():
    asset = make_asset(1)
    tracker = make_tracker(asset, amount=-SHARES)
    tracker.earn_dividends(
        [Dividend(asset, amount=AMOUNT_PER_SHARE, pay_date=pd.NaT)],
        [],
        ex_date=EX_DATE,
    )

    total, _ = run_sessions(tracker)

    assert total == -EXPECTED_CASH


def test_multiple_dividends_due_on_the_same_session():
    """Two pay dates that come due together are both paid, not just one."""
    asset_a = make_asset(1)
    asset_b = make_asset(2)

    tracker = PositionTracker("daily")
    tracker.update_position(asset_a, amount=SHARES, last_sale_price=100.0)
    tracker.update_position(asset_b, amount=SHARES, last_sale_price=100.0)

    tracker.earn_dividends(
        [
            # Saturday and Sunday both roll forward onto 06-08.
            Dividend(asset_a, AMOUNT_PER_SHARE, pd.Timestamp("2015-06-06", tz="UTC")),
            Dividend(asset_b, AMOUNT_PER_SHARE, pd.Timestamp("2015-06-07", tz="UTC")),
        ],
        [],
        ex_date=EX_DATE,
    )

    assert tracker.pay_dividends(SESSIONS[5]) == 2 * EXPECTED_CASH


def test_stock_dividend_with_missing_pay_date_is_paid():
    asset = make_asset(1)
    payment_asset = make_asset(2)
    tracker = make_tracker(asset)

    tracker.earn_dividends(
        [],
        [StockDividend(asset, payment_asset, ratio=0.5, pay_date=pd.NaT)],
        ex_date=EX_DATE,
    )

    tracker.pay_dividends(EX_DATE)

    assert tracker.positions[payment_asset].amount == SHARES * 0.5
    assert not tracker._unpaid_stock_dividends


def test_missing_pay_date_without_an_ex_date_is_reported(caplog):
    """Without a fallback there is nothing to do but say the cash is gone."""
    asset = make_asset(1)
    tracker = make_tracker(asset)

    tracker.earn_dividends(
        [Dividend(asset, amount=AMOUNT_PER_SHARE, pay_date=pd.NaT)],
        [],
    )

    total, _ = run_sessions(tracker)

    assert total == 0.0
    assert "will never be made" in caplog.text


@pytest.mark.parametrize(
    "pay_date, session",
    [
        pytest.param(
            pd.Timestamp("2015-06-03", tz="UTC"),
            pd.Timestamp("2015-06-03"),
            id="aware_pay_date_naive_session",
        ),
        pytest.param(
            pd.Timestamp("2015-06-03"),
            pd.Timestamp("2015-06-03", tz="UTC"),
            id="naive_pay_date_aware_session",
        ),
    ],
)
def test_naive_and_aware_dates_are_compared(pay_date, session):
    """The adjustments db hands back UTC-aware pay dates; a caller may pass
    naive sessions. Comparing the two raised TypeError, which used to drop
    the payment and log the same warning every session after."""
    asset = make_asset(1)
    tracker = make_tracker(asset)
    tracker.earn_dividends(
        [Dividend(asset, amount=AMOUNT_PER_SHARE, pay_date=pay_date)],
        [],
        ex_date=EX_DATE,
    )

    assert tracker.pay_dividends(session - pd.Timedelta(days=1)) == 0.0
    assert tracker.pay_dividends(session) == EXPECTED_CASH
    assert not tracker._unpaid_dividends


class TestExDateWindow:
    """A dividend must be earned exactly once, on the first session at or
    after its ex_date, even when the ex_date is not a session. Prices are
    adjusted from that session either way, so matching ex_date to the
    session exactly lost the cash while still marking the position down.
    """

    SID = 1
    CALENDAR = get_calendar("XNYS")

    def _reader(self, ex_date):
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE dividend_payouts (sid INTEGER, ex_date INTEGER,"
            " declared_date INTEGER, record_date INTEGER, pay_date INTEGER,"
            " amount REAL)"
        )
        conn.execute(
            "CREATE TABLE stock_dividend_payouts (sid INTEGER, ex_date INTEGER,"
            " declared_date INTEGER, record_date INTEGER, pay_date INTEGER,"
            " payment_sid INTEGER, ratio REAL)"
        )
        seconds = int(pd.Timestamp(ex_date).value // 10**9)
        conn.execute(
            "INSERT INTO dividend_payouts VALUES (?, ?, NULL, NULL, ?, ?)",
            (self.SID, seconds, seconds, AMOUNT_PER_SHARE),
        )
        conn.execute(
            "INSERT INTO stock_dividend_payouts VALUES (?, ?, NULL, NULL, ?, ?, ?)",
            (self.SID, seconds, seconds, 2, 0.5),
        )
        return SQLiteAdjustmentReader(conn)

    @staticmethod
    def _finder():
        return SimpleNamespace(retrieve_asset=make_asset)

    def _sessions_earning(self, reader, method):
        sessions = self.CALENDAR.sessions_in_range("2024-01-02", "2024-04-30")
        earned = []
        for session in sessions:
            aware = session.tz_localize("UTC")
            found = getattr(reader, method)(
                {self.SID},
                aware,
                self._finder(),
                session_before(self.CALENDAR.sessions, aware),
            )
            earned.extend([session] * len(found))
        return earned

    @pytest.mark.parametrize(
        "method", ["get_dividends_with_ex_date", "get_stock_dividends_with_ex_date"]
    )
    @pytest.mark.parametrize(
        "ex_date, expected_session",
        [
            pytest.param("2024-01-10", "2024-01-10", id="on_session"),
            pytest.param("2024-01-13", "2024-01-16", id="weekend_and_holiday"),
            pytest.param("2024-03-29", "2024-04-01", id="exchange_holiday"),
        ],
    )
    def test_earned_once_on_first_session_at_or_after(
        self, method, ex_date, expected_session
    ):
        earned = self._sessions_earning(self._reader(ex_date), method)

        assert earned == [pd.Timestamp(expected_session)]

    def test_without_previous_session_only_exact_ex_date_matches(self):
        reader = self._reader("2024-01-13")
        finder = self._finder()

        for day in ("2024-01-12", "2024-01-13", "2024-01-16"):
            found = reader.get_dividends_with_ex_date(
                {self.SID}, pd.Timestamp(day, tz="UTC"), finder
            )
            assert len(found) == (day == "2024-01-13"), day

    def test_ledger_earns_weekend_ex_date_and_pays_it(self):
        reader = self._reader("2024-01-13")
        asset = make_asset(self.SID)
        ledger = Ledger(
            self.CALENDAR.sessions_in_range("2024-01-02", "2024-01-31"),
            capital_base=100_000.0,
            data_frequency="daily",
        )
        ledger.position_tracker.update_position(
            asset, amount=SHARES, last_sale_price=100.0
        )
        cash_before = ledger.portfolio.cash

        for session in ("2024-01-12", "2024-01-16"):
            ledger.process_dividends(
                pd.Timestamp(session, tz="UTC"),
                self._finder(),
                reader,
                previous_session=session_before(
                    self.CALENDAR.sessions, pd.Timestamp(session)
                ),
            )

        assert ledger.portfolio.cash - cash_before == EXPECTED_CASH

    def test_metrics_tracker_passes_the_previous_session(self):
        reader = self._reader("2024-01-13")
        asset = make_asset(self.SID)
        tracker = MetricsTracker(
            trading_calendar=self.CALENDAR,
            first_session=pd.Timestamp("2024-01-02"),
            last_session=pd.Timestamp("2024-01-31"),
            capital_base=100_000.0,
            emission_rate="daily",
            data_frequency="daily",
            asset_finder=self._finder(),
            metrics=load_metrics_set("none"),
        )
        ledger = tracker._ledger
        ledger.position_tracker.update_position(
            asset, amount=SHARES, last_sale_price=100.0
        )
        cash_before = ledger.portfolio.cash
        data_portal = SimpleNamespace(adjustment_reader=reader)

        tracker.handle_market_open(pd.Timestamp("2024-01-16", tz="UTC"), data_portal)

        assert ledger.portfolio.cash - cash_before == EXPECTED_CASH
