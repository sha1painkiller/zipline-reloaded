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

from collections import namedtuple

import pandas as pd
import pytest

from zipline.assets import Equity, ExchangeInfo
from zipline.finance.ledger import PositionTracker

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
