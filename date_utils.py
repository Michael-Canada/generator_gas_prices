from datetime import date, datetime, time, timedelta
from typing import Set, List, NewType, Iterable, Tuple, Optional, Union, TYPE_CHECKING

import pytz


# Because typeshed `BaseTzInfo` omits the `is_dst` argument for `localize()`, it is
# not usable for our purposes. The `Union` below instead does what we want.
#
# https://github.com/python/typeshed/blob/master/stubs/pytz/pytz/tzinfo.pyi#L10
if TYPE_CHECKING:
    # pylint: disable=protected-access
    Tz = Union[pytz._UTCclass, pytz.tzinfo.StaticTzInfo, pytz.tzinfo.DstTzInfo]
else:
    # `_UTCclass` is only available in the type stub.
    Tz = Union[pytz.tzinfo.StaticTzInfo, pytz.tzinfo.DstTzInfo]


def parse_multiple_date_format(date_string: str, formats: List[str]) -> datetime:

    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            pass

    raise Exception("Unable to parse date: '%s'" % date_string)


def is_dst(timestamp: datetime) -> bool:
    assert (
        timestamp.tzinfo is not None
    ), "can only be computed on a datetime with timezone"
    return timestamp.dst() == timedelta(hours=1)


def make_dst_transition_date_set(timezone: Tz) -> Set[date]:
    # pylint: disable=no-member, protected-access
    return {d.date() for d in timezone._utc_transition_times}  # type: ignore
    # pylint: enable=no-member, protected-access


def truncate_date_to_hour(in_datetime: datetime) -> datetime:
    return in_datetime.replace(minute=0, second=0, microsecond=0)


def is_non_existent_datetime(dtm: datetime, timezone: Tz) -> bool:
    try:
        timezone.localize(dtm, is_dst=None)
    except pytz.NonExistentTimeError:
        return True
    except pytz.AmbiguousTimeError:
        return False
    return False


def is_ambiguous_datetime(dtm: datetime, timezone: Tz) -> bool:
    try:
        timezone.localize(dtm, is_dst=None)
    except pytz.AmbiguousTimeError:
        return True
    except pytz.NonExistentTimeError:
        return False
    return False


LocalizedDateTime = NewType("LocalizedDateTime", datetime)
NaiveDateTime = NewType("NaiveDateTime", datetime)


def increment(localized_dtm: LocalizedDateTime, delta: timedelta) -> LocalizedDateTime:
    assert localized_dtm.tzinfo is not None

    return LocalizedDateTime(
        (localized_dtm.astimezone(pytz.UTC) + delta).astimezone(localized_dtm.tzinfo)
    )


def increment_naive(
    localized_dtm: LocalizedDateTime,
    naive_delta: timedelta,
    result_tz: Tz,
    result_dst: bool = False,
) -> LocalizedDateTime:

    assert localized_dtm.tzinfo is not None

    naive_result = localized_dtm.replace(tzinfo=None) + naive_delta

    localized_result = result_tz.localize(naive_result, is_dst=result_dst)

    return LocalizedDateTime(localized_result)


def localized(in_datetime: datetime) -> LocalizedDateTime:
    assert in_datetime.tzinfo is not None
    return LocalizedDateTime(in_datetime)


def naive_from_isoformat(naive_str: str) -> NaiveDateTime:
    naive = datetime.fromisoformat(naive_str)
    assert naive.tzinfo is None

    return NaiveDateTime(naive)


def localized_from_isoformat(
    localized_str: str, to_tz: Optional[Tz] = None
) -> LocalizedDateTime:
    localized_dtm = datetime.fromisoformat(localized_str)
    assert localized_dtm.tzinfo is not None

    if to_tz is not None:
        localized_dtm = localized_dtm.astimezone(to_tz)

    return LocalizedDateTime(localized_dtm)


def naive_from_parts(date_part: date, time_part: time) -> NaiveDateTime:
    return NaiveDateTime(datetime.combine(date_part, time_part))


def iter_dt_exclusive(
    start: LocalizedDateTime, exclusive_end: LocalizedDateTime, delta: timedelta
) -> Iterable[LocalizedDateTime]:

    assert exclusive_end > start
    assert delta > timedelta(seconds=0)

    dt = start

    while dt < exclusive_end:
        yield dt
        dt = increment(dt, delta)


def _align(dt: LocalizedDateTime, delta: timedelta) -> LocalizedDateTime:
    # Alignment supported for intervals <= 1 hour which evenly divide 1 hour.
    # In this case, the proper alignment for every hour is independent.

    delta_secs = int(delta.total_seconds())
    assert delta_secs == delta.total_seconds()
    assert delta_secs <= 60 * 60
    assert 60 * 60 % delta_secs == 0

    hour_passed = dt - dt.replace(minute=0, second=0, microsecond=0)
    extra_s = hour_passed.total_seconds() % delta.total_seconds()

    extra = timedelta(seconds=extra_s)

    return increment(dt, -extra)


def prev_aligned_on_or_before(
    dt: LocalizedDateTime, delta: timedelta
) -> LocalizedDateTime:

    return _align(dt, delta)


def localize_datetime(
    naive: NaiveDateTime, is_dst_flag: bool, time_zone: Tz
) -> LocalizedDateTime:

    assert naive.tzinfo is None

    localized_dtm = time_zone.localize(naive, is_dst_flag)

    return LocalizedDateTime(localized_dtm)


def now(tz: Tz) -> LocalizedDateTime:
    return astimezone(utcnow(), tz)


def utcnow() -> LocalizedDateTime:
    return localize_datetime(
        NaiveDateTime(datetime.utcnow()), is_dst_flag=False, time_zone=pytz.utc
    )


def as_naive(localized_dtm: LocalizedDateTime, tz: Tz) -> Tuple[NaiveDateTime, bool]:
    assert localized_dtm.tzinfo is not None

    with_tz = localized_dtm.astimezone(tz)

    return NaiveDateTime(with_tz.replace(tzinfo=None)), is_dst(with_tz)


def utcfromtimestamp(timestamp: int) -> LocalizedDateTime:
    return localized(pytz.utc.localize(datetime.utcfromtimestamp(timestamp)))


def convert_to_unix_timestamp(dtm: LocalizedDateTime) -> float:
    dtm_utc = dtm.astimezone(pytz.timezone("UTC"))
    return dtm_utc.timestamp()


def astimezone(dtm: LocalizedDateTime, tz: Tz) -> LocalizedDateTime:
    assert dtm.tzinfo is not None

    return localized(dtm.astimezone(tz))


def beginning_of(naive_date: date, tz: Tz) -> LocalizedDateTime:
    naive_dt = NaiveDateTime(datetime.combine(naive_date, time()))
    return localize_datetime(naive=naive_dt, is_dst_flag=False, time_zone=tz)
