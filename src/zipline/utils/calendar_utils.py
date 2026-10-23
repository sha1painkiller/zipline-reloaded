import inspect
from functools import partial

import pandas as pd
from exchange_calendars import ExchangeCalendar as TradingCalendar
from exchange_calendars import clear_calendars
from exchange_calendars import get_calendar as ec_get_calendar  # get_calendar,
from exchange_calendars import (
    get_calendar_names,
    register_calendar,
    register_calendar_alias,
)
from exchange_calendars.calendar_utils import global_calendar_dispatcher

# from exchange_calendars.errors import InvalidCalendarName
from exchange_calendars.utils.pandas_utils import days_at_time  # noqa: reexport


# https://stackoverflow.com/questions/56753846/python-wrapping-function-with-signature
def wrap_with_signature(signature):
    def wrapper(func):
        func.__signature__ = signature
        return func

    return wrapper


@wrap_with_signature(inspect.signature(ec_get_calendar))
def get_calendar(*args, **kwargs):
    if args[0] in ["us_futures", "CMES", "XNYS", "NYSE"]:
        return ec_get_calendar(*args, side="right", start=pd.Timestamp("1990-01-01"))
    return ec_get_calendar(*args, side="right")


# get_calendar = compose(partial(get_calendar, side="right"), "XNYS")
# NOTE Sessions are now timezone-naive (previously UTC).
# Schedule columns now have timezone set as UTC
# (whilst the times have always been defined in terms of UTC,
# previously the dtype was timezone-naive).


def naive_session_label(dt):
    """``dt`` as a naive session label.

    Session labels are naive midnights that name a date. An aware ``dt``
    keeps its own wall-clock date and time and only drops the zone: a
    US/Eastern or Asia/Tokyo midnight names the same session as a UTC one,
    whereas converting it to UTC first would move it onto the wrong side of
    a session boundary.
    """
    if dt.tzinfo is not None:
        return dt.tz_localize(None)
    return dt


def session_before(sessions, dt):
    """The last of ``sessions`` strictly before the session labelled ``dt``.

    If no session precedes ``dt``, returns ``dt`` minus one day, so the
    result can always serve as the exclusive lower bound of a
    ``(previous session, dt]`` window. Compute that window's upper bound from
    ``naive_session_label(dt)`` as well: mixing a label with ``dt``'s UTC
    instant puts the two bounds hours apart for any zone but UTC.
    """
    dt = naive_session_label(dt)

    loc = sessions.searchsorted(dt)
    if loc == 0:
        return dt - pd.Timedelta(days=1)

    return sessions[loc - 1]
