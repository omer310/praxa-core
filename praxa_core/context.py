"""Surface-agnostic execution context for Praxa tools.

Replaces the LiveKit-coupled ContextVar (`_current_session`) used by the voice
agent with an explicit dependency container that any surface (voice, SMS,
in-app, background) can build and pass into the `*_impl` tool functions.
"""
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

from ._utils import logger

try:  # supabase is an optional dependency of the package itself
    from supabase import Client
except Exception:  # pragma: no cover - typing fallback
    Client = Any  # type: ignore

EMAIL_CACHE_TTL = 300
CALENDAR_DAY_LOOKBACK_DAYS = 1
CALENDAR_DAY_LOOKAHEAD_DAYS = 30


@dataclass
class ToolContext:
    """Everything a tool needs, passed explicitly instead of read from a ContextVar.

    `surface` tells the impl which channel it is running on ("voice", "sms",
    "in_app", "background"). The cache fields are scoped to a single
    conversation/session and let email tools resolve references like "reply to
    the first one".
    """

    user_id: str = ""
    supabase: Optional[Client] = None
    email_grant_id: Optional[str] = None
    calendar_grant_id: Optional[str] = None
    timezone: str = "UTC"
    surface: str = "voice"
    nylas_api_key: Optional[str] = None

    # session-scoped working memory (email references + classification cache)
    last_fetched_emails: list = field(default_factory=list)
    email_classification_cache: list = field(default_factory=list)
    email_classification_cache_ids: Optional[tuple] = None
    email_classification_cache_at: float = 0.0

    def __post_init__(self) -> None:
        if self.nylas_api_key is None:
            self.nylas_api_key = os.getenv("NYLAS_API_KEY")

    @property
    def uid(self) -> str:
        return self.user_id or ""

    @property
    def sb(self) -> Optional[Client]:
        return self.supabase

    def tz(self) -> str:
        return self.timezone or "UTC"

    def now(self) -> datetime:
        """Current datetime in the user's local timezone."""
        try:
            return datetime.now(ZoneInfo(self.tz()))
        except Exception:
            return datetime.now(ZoneInfo("UTC"))

    async def ensure_calendar_grant(self) -> Optional[str]:
        """Return the calendar grant id, fetching from the DB if not already set."""
        if self.calendar_grant_id:
            return self.calendar_grant_id
        grant = await self._fetch_grant("calendar")
        if grant:
            self.calendar_grant_id = grant
        return self.calendar_grant_id

    async def ensure_email_grant(self) -> Optional[str]:
        if self.email_grant_id:
            return self.email_grant_id
        grant = await self._fetch_grant("email")
        if grant:
            self.email_grant_id = grant
        return self.email_grant_id

    async def _fetch_grant(self, integration_type: str) -> Optional[str]:
        import asyncio

        if not self.supabase or not self.uid:
            return None
        try:
            sb = self.supabase
            uid = self.uid
            resp = await asyncio.to_thread(
                lambda: sb.table("nylas_oauth_tokens")
                .select("grant_id")
                .eq("user_id", uid)
                .eq("integration_type", integration_type)
                .maybe_single()
                .execute()
            )
            if resp and getattr(resp, "data", None) and resp.data.get("grant_id"):
                return resp.data["grant_id"]
        except Exception as e:  # pragma: no cover - network/db
            logger.error(f"[ToolContext] Error fetching {integration_type} grant: {e}")
        return None


# ---------------------------------------------------------------------------
# Calendar helpers (ported from the voice agent's context.py; now timezone-pure)
# ---------------------------------------------------------------------------

def format_event_time(start_time, tz: str, fmt: str = "%A, %B %d at %I:%M %p") -> str:
    """Format a Nylas event start_time (Unix int or ISO string) in the given tz."""
    try:
        zone = ZoneInfo(tz)
        if isinstance(start_time, (int, float)):
            dt = datetime.fromtimestamp(start_time, tz=zone)
        else:
            dt = datetime.fromisoformat(str(start_time).replace("Z", "+00:00")).astimezone(zone)
        return dt.strftime(fmt).lstrip("0")
    except Exception:
        return str(start_time)


def pick_primary_calendar(calendars: list) -> Optional[str]:
    """Return the primary calendar ID from a Nylas calendar list."""
    for cal in calendars:
        if cal.get("is_primary"):
            return cal.get("id")
    for cal in calendars:
        if not cal.get("read_only", False):
            return cal.get("id")
    return calendars[0].get("id") if calendars else None


def auto_update_tz(ctx: ToolContext, calendars: list, primary_id: Optional[str]) -> None:
    """Read the timezone from the primary Google Calendar and update the context."""
    candidates = []
    if primary_id:
        candidates += [c for c in calendars if c.get("id") == primary_id]
    candidates += [c for c in calendars if not c.get("read_only", False)]
    for cal in candidates:
        tz_str = cal.get("timezone")
        if tz_str:
            try:
                ZoneInfo(tz_str)
                if ctx.timezone != tz_str:
                    logger.info(f"Timezone auto-detected from Google Calendar: {tz_str}")
                    ctx.timezone = tz_str
                return
            except Exception:
                logger.warning(f"Ignoring invalid timezone from calendar: {tz_str}")


def _calendar_single_day_window(now_dt: datetime) -> tuple[date, date]:
    today = now_dt.date()
    lo = today - timedelta(days=CALENDAR_DAY_LOOKBACK_DAYS)
    hi = today + timedelta(days=CALENDAR_DAY_LOOKAHEAD_DAYS)
    return lo, hi


def calendar_outside_window_message(lo: date, hi: date) -> str:
    return (
        f"I can only check specific days between {lo.strftime('%B %d, %Y')} and {hi.strftime('%B %d, %Y')} "
        f"(about one month from now). Ask again for a date in that range, or say what's coming up for a broader summary."
    )


def resolve_calendar_target_day(day_raw: str, now_dt: datetime):
    """Parse a day string to (target_date, day_label, error_message)."""
    from dateutil import parser as date_parser

    s = (day_raw or "").strip()
    if not s:
        return None, None, "Say which day you want — for example tomorrow, Monday, or April 19."

    today = now_dt.date()
    tz = now_dt.tzinfo
    lo, hi = _calendar_single_day_window(now_dt)
    sl = s.lower()

    if sl in ("today", "now", "tonight"):
        target = today
    elif sl in ("tomorrow", "tmr", "tmrw"):
        target = today + timedelta(days=1)
    elif sl == "yesterday":
        target = today - timedelta(days=1)
    else:
        day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        target = None
        for i, name in enumerate(day_names):
            if name in sl:
                days_ahead_delta = i - today.weekday()
                if days_ahead_delta <= 0:
                    days_ahead_delta += 7
                target = today + timedelta(days=days_ahead_delta)
                break
        if target is None:
            iso_m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s.strip())
            if iso_m:
                try:
                    y, mo, d = int(iso_m.group(1)), int(iso_m.group(2)), int(iso_m.group(3))
                    target = date(y, mo, d)
                except ValueError:
                    return None, None, f"I couldn't parse '{s}' as a calendar date. Try a format like 2026-04-19 or April 19."
            else:
                try:
                    dt_parsed = date_parser.parse(s, default=now_dt, fuzzy=False)
                    target = dt_parsed.date()
                except (ValueError, TypeError, OverflowError):
                    try:
                        dt_parsed = date_parser.parse(s, default=now_dt, fuzzy=True)
                        target = dt_parsed.date()
                    except (ValueError, TypeError, OverflowError):
                        return None, None, (
                            f"I couldn't understand '{s}'. Try a specific date (April 19, 4/19/2026), "
                            f"tomorrow, or a weekday like Friday."
                        )

    if target is None:
        return None, None, "I couldn't work out which day you meant. Try a specific date like April 19 or 2026-04-19."

    if target < lo or target > hi:
        return None, None, calendar_outside_window_message(lo, hi)

    day_label = datetime(target.year, target.month, target.day, tzinfo=tz).strftime("%A, %B %d, %Y")
    return target, day_label, None


def calendar_no_events_for_day_message(day_label: str) -> str:
    return (
        f"I checked your connected primary calendar for {day_label} and didn't find any events in that day's window. "
        f"If something should appear, it may be on another calendar in your account or still syncing — double-check the app."
    )


def calendar_no_events_upcoming_message() -> str:
    return (
        "I queried your connected primary calendar for roughly the next 30 days and don't see events in that range. "
        "If you're expecting meetings, confirm they're on the calendar account you linked, or try again after sync catches up."
    )
