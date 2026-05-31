"""Calendar tool implementations (Nylas). Pure logic, no LiveKit.

Reads are direct against Nylas. `reschedule_calendar_event_impl` performs the
write directly today (it also syncs the linked Praxa loop); the action pipeline
can wrap it later, but it stays callable for the voice surface's confirm-first flow.
"""
import asyncio
from datetime import datetime, timedelta, timezone as dt_tz

import httpx
from dateutil import parser as dateutil_parser

from .._utils import logger
from ..context import (
    ToolContext,
    pick_primary_calendar,
    auto_update_tz,
    format_event_time,
    resolve_calendar_target_day,
    calendar_no_events_for_day_message,
    calendar_no_events_upcoming_message,
)

_NYLAS_BASE = "https://api.us.nylas.com/v3/grants"


def _no_signin() -> str:
    return "I can't access your calendar right now — it looks like you're not signed in. Please open the app and sign in first."


def _headers(ctx: ToolContext, content_type: bool = False) -> dict:
    h = {"Authorization": f"Bearer {ctx.nylas_api_key}", "Accept": "application/json"}
    if content_type:
        h["Content-Type"] = "application/json"
    return h


async def get_upcoming_events_impl(ctx: ToolContext, days_ahead: int = 30) -> str:
    if not ctx.uid:
        return _no_signin()
    grant = await ctx.ensure_calendar_grant()
    if not grant:
        return "I don't have access to your calendar yet. Please connect your calendar in the app settings."
    if not ctx.nylas_api_key:
        return "Calendar service is not configured. Please contact support."
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            cal_response = await client.get(f"{_NYLAS_BASE}/{grant}/calendars", headers=_headers(ctx))
            if cal_response.status_code != 200:
                if cal_response.status_code in (401, 404):
                    return "I'm having trouble accessing your calendar. The connection may have expired. Please try reconnecting your calendar."
                return "I couldn't access your calendar. Please reconnect your calendar in the app."
            calendars = cal_response.json().get("data", [])
            if not calendars:
                return "No calendars found in your account."
            calendar_id = pick_primary_calendar(calendars)
            auto_update_tz(ctx, calendars, calendar_id)

            _n = ctx.now()
            params = {
                "calendar_id": calendar_id,
                "limit": 50,
                "start": int(_n.timestamp()),
                "end": int((_n + timedelta(days=days_ahead)).timestamp()),
            }
            response = await client.get(f"{_NYLAS_BASE}/{grant}/events", headers=_headers(ctx), params=params)
            if response.status_code != 200:
                if response.status_code in (401, 404):
                    return "I'm having trouble accessing your calendar. The connection may have expired. Please try reconnecting your calendar."
                return "I'm having trouble accessing your calendar right now. Please try again."

            events = response.json().get("data", [])
            if not events:
                return calendar_no_events_upcoming_message()

            user_events, holiday_events = [], []
            for event in events:
                when = event.get("when", {})
                title = event.get("title", "").lower()
                is_holiday = any(k in title for k in [
                    "holiday", "observance", "public holiday", "national holiday",
                    "federal holiday", "bank holiday", "religious holiday",
                ])
                is_all_day = when.get("date") and not when.get("start_time")
                if when.get("start_time") and not is_holiday:
                    user_events.append(event)
                elif is_holiday or is_all_day:
                    holiday_events.append(event)

            all_events = user_events + holiday_events
            if not all_events:
                return calendar_no_events_upcoming_message()

            event_list = []
            for event in all_events[:20]:
                title = event.get("title", "Untitled")
                when = event.get("when", {})
                start_time = when.get("start_time")
                if start_time:
                    event_list.append(f"{title} ({format_event_time(start_time, ctx.tz())})")
                else:
                    date_str = when.get("date", "")
                    if date_str:
                        try:
                            dt = datetime.fromisoformat(date_str)
                            event_list.append(f"{title} ({dt.strftime('%A, %B %d')})")
                        except Exception:
                            event_list.append(f"{title} ({date_str})")
                    else:
                        event_list.append(title)

            user_count = len(user_events)
            total_count = len(event_list)
            if user_count > 0:
                return f"Found {total_count} upcoming events ({user_count} scheduled): " + "; ".join(event_list)
            return f"Found {total_count} upcoming events: " + "; ".join(event_list)
    except httpx.TimeoutException:
        return "The calendar service is taking too long to respond. Please try again."
    except Exception as e:
        logger.error(f"Error getting calendar events: {type(e).__name__}: {e}")
        return "Sorry, I couldn't access your calendar"


async def find_calendar_event_impl(ctx: ToolContext, event_name: str) -> str:
    if not ctx.uid:
        return _no_signin()
    grant = await ctx.ensure_calendar_grant()
    if not grant:
        return "Calendar access isn't configured yet. Please connect your calendar in the app settings."
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            cal_response = await client.get(f"{_NYLAS_BASE}/{grant}/calendars", headers=_headers(ctx))
            if cal_response.status_code != 200:
                return "Couldn't access calendar"
            calendars = cal_response.json().get("data", [])
            if not calendars:
                return "No calendars found"
            calendar_id = pick_primary_calendar(calendars)
            auto_update_tz(ctx, calendars, calendar_id)
            response = await client.get(
                f"{_NYLAS_BASE}/{grant}/events", headers=_headers(ctx),
                params={"calendar_id": calendar_id, "limit": 50},
            )
            if response.status_code != 200:
                return "Couldn't search calendar events"
            events = response.json().get("data", [])
            matching = [e for e in events if event_name.lower() in e.get("title", "").lower()]
            if not matching:
                return f"No events found matching '{event_name}'"
            event_list = []
            for event in matching[:5]:
                title = event.get("title", "Untitled")
                when = event.get("when", {})
                start_time = when.get("start_time") or when.get("date")
                event_list.append(f"- {title}" + (f" ({start_time})" if start_time else ""))
            return f"Events matching '{event_name}':\n" + "\n".join(event_list)
    except Exception as e:
        logger.error(f"[find_calendar_event] Error: {type(e).__name__}: {e}")
        return f"Sorry, I couldn't search for '{event_name}'"


async def get_todays_events_impl(ctx: ToolContext) -> str:
    if not ctx.uid:
        return _no_signin()
    grant = await ctx.ensure_calendar_grant()
    if not grant:
        return "Calendar access isn't configured yet. Please connect your calendar in the app settings."
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            cal_response = await client.get(f"{_NYLAS_BASE}/{grant}/calendars", headers=_headers(ctx))
            if cal_response.status_code != 200:
                return "Couldn't access calendar"
            calendars = cal_response.json().get("data", [])
            if not calendars:
                return "No calendars found"
            calendar_id = pick_primary_calendar(calendars)
            auto_update_tz(ctx, calendars, calendar_id)
            now_dt = ctx.now()
            tz = now_dt.tzinfo
            today_start = int(datetime(now_dt.year, now_dt.month, now_dt.day, 0, 0, 0, tzinfo=tz).timestamp())
            today_end = int(datetime(now_dt.year, now_dt.month, now_dt.day, 23, 59, 59, tzinfo=tz).timestamp())
            response = await client.get(
                f"{_NYLAS_BASE}/{grant}/events", headers=_headers(ctx),
                params={"calendar_id": calendar_id, "start": today_start, "end": today_end, "limit": 20},
            )
            if response.status_code != 200:
                return "I couldn't load today's events from the calendar service. Try again shortly — I haven't confirmed your schedule."
            events = response.json().get("data", [])
            if not events:
                return calendar_no_events_for_day_message(now_dt.strftime("%A, %B %d, %Y"))
            event_list = []
            for event in events:
                title = event.get("title", "Untitled")
                start_time = event.get("when", {}).get("start_time", "")
                if start_time:
                    event_list.append(f"{title} at {format_event_time(start_time, ctx.tz(), '%I:%M %p')}")
                else:
                    event_list.append(title)
            return "Today's schedule: " + "; ".join(event_list)
    except Exception as e:
        logger.error(f"[get_todays_events] Error: {type(e).__name__}: {e}")
        return "Something went wrong checking today — try again in a moment; I didn't verify your calendar."


async def get_events_for_day_impl(ctx: ToolContext, day: str) -> str:
    if not ctx.uid:
        return "I can't access your calendar right now — it looks like you're not signed in."
    grant = await ctx.ensure_calendar_grant()
    if not grant:
        return "Calendar access isn't configured yet. Please connect your calendar in the app settings."
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            cal_response = await client.get(f"{_NYLAS_BASE}/{grant}/calendars", headers=_headers(ctx))
            if cal_response.status_code != 200:
                return "I couldn't open your calendar list from the provider, so I didn't check that day. Try again shortly."
            calendars = cal_response.json().get("data", [])
            if not calendars:
                return "No calendars were returned for your account, so I couldn't look up that day."
            calendar_id = pick_primary_calendar(calendars)
            auto_update_tz(ctx, calendars, calendar_id)

            now_dt = ctx.now()
            target_date, day_label, day_err = resolve_calendar_target_day(day, now_dt)
            if day_err:
                return day_err
            tz = now_dt.tzinfo
            day_start = int(datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=tz).timestamp())
            day_end = int(datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59, tzinfo=tz).timestamp())
            response = await client.get(
                f"{_NYLAS_BASE}/{grant}/events", headers=_headers(ctx),
                params={"calendar_id": calendar_id, "start": day_start, "end": day_end, "limit": 50},
            )
            if response.status_code != 200:
                return f"I couldn't load events for {day_label} from the calendar service right now. Try again in a moment."
            events = response.json().get("data", [])
            if not events:
                return calendar_no_events_for_day_message(day_label)
            event_list = []
            for event in events:
                title = event.get("title", "Untitled")
                start_time = event.get("when", {}).get("start_time", "")
                if start_time:
                    event_list.append(f"{title} at {format_event_time(start_time, ctx.tz(), '%I:%M %p')}")
                else:
                    event_list.append(f"{title} (all day)")
            return f"{day_label}: " + "; ".join(event_list)
    except Exception as e:
        logger.error(f"[get_events_for_day] Error: {type(e).__name__}: {e}")
        return "Something went wrong while checking that day — I didn't confirm your calendar, so try again in a moment."


async def check_free_time_impl(ctx: ToolContext, date: str) -> str:
    if not ctx.uid:
        return _no_signin()
    grant = await ctx.ensure_calendar_grant()
    if not grant:
        return "Calendar access isn't configured yet. Please connect your calendar in the app settings."
    try:
        try:
            check_date = dateutil_parser.parse(date).date()
        except Exception:
            try:
                check_date = datetime.fromisoformat(date).date()
            except Exception:
                return f"Couldn't understand the date '{date}'. Please use YYYY-MM-DD format or natural language like 'tomorrow'."

        async with httpx.AsyncClient(timeout=10.0) as client:
            cal_response = await client.get(f"{_NYLAS_BASE}/{grant}/calendars", headers=_headers(ctx))
            if cal_response.status_code != 200:
                return "Couldn't access calendar"
            calendars = cal_response.json().get("data", [])
            if not calendars:
                return "No calendars found"
            calendar_id = pick_primary_calendar(calendars)
            start_of_day = datetime.combine(check_date, datetime.min.time()).isoformat()
            end_of_day = datetime.combine(check_date, datetime.max.time()).isoformat()
            events_response = await client.get(
                f"{_NYLAS_BASE}/{grant}/events", headers=_headers(ctx),
                params={"calendar_id": calendar_id, "start": start_of_day, "end": end_of_day, "limit": 50},
            )
            if events_response.status_code != 200:
                return "Couldn't fetch events for that day"
            events = events_response.json().get("data", [])

            timed_events = []
            for event in events:
                when = event.get("when", {})
                if when.get("start_time") and when.get("end_time"):
                    try:
                        start_dt = datetime.fromisoformat(when["start_time"].replace("Z", "+00:00"))
                        end_dt = datetime.fromisoformat(when["end_time"].replace("Z", "+00:00"))
                        timed_events.append((start_dt, end_dt))
                    except Exception:
                        pass
            timed_events.sort(key=lambda x: x[0])

            day_start = datetime.combine(check_date, datetime.min.time().replace(hour=8))
            day_end = datetime.combine(check_date, datetime.min.time().replace(hour=20))
            free_slots = []
            current_time = day_start
            for event_start, event_end in timed_events:
                if event_start < day_start:
                    current_time = max(current_time, event_end)
                    continue
                if event_start > day_end:
                    break
                if current_time < event_start:
                    free_slots.append((current_time, event_start))
                current_time = max(current_time, event_end)
            if current_time < day_end:
                free_slots.append((current_time, day_end))

            if not free_slots:
                return f"On {check_date.strftime('%A %B %d')}, you're fully booked from 8am to 8pm."
            slot_strings = []
            for start, end in free_slots:
                duration = int((end - start).total_seconds() / 60)
                slot_strings.append(f"{start.strftime('%I:%M %p')}-{end.strftime('%I:%M %p')} ({duration} minutes)")
            return f"On {check_date.strftime('%A %B %d')}, you have free time: " + "; ".join(slot_strings)
    except Exception as e:
        logger.error(f"[check_free_time] Error: {type(e).__name__}: {e}")
        return "Sorry, I couldn't check your free time"


def _sync_praxa_loop_after_event_reschedule(ctx: ToolContext, event_id, event_title, new_start_iso, due_date_str) -> None:
    sb, uid = ctx.supabase, ctx.uid
    if not sb or not uid:
        return
    try:
        now_iso = datetime.now(dt_tz.utc).isoformat()
        payload = {"scheduled_time": new_start_iso, "due_date": due_date_str, "updated_at": now_iso}
        res = sb.table("loops").update(payload).eq("user_id", uid).eq("calendar_event_id", event_id).execute()
        if getattr(res, "data", None):
            return
        et = (event_title or "").strip()
        if not et:
            return
        r2 = sb.table("loops").select("id, calendar_event_id, title").eq("user_id", uid).eq("title", et).limit(8).execute()
        for row in r2.data or []:
            cid = row.get("calendar_event_id")
            if cid and cid != event_id:
                continue
            extra = {} if cid else {"calendar_event_id": event_id}
            sb.table("loops").update({**payload, **extra}).eq("id", row["id"]).execute()
            return
    except Exception as e:
        logger.warning("[reschedule_calendar_event] Praxa task sync skipped: %s", e)


async def reschedule_calendar_event_impl(ctx: ToolContext, event_name: str, new_date_time: str) -> str:
    if not ctx.uid:
        return _no_signin()
    grant = await ctx.ensure_calendar_grant()
    if not grant:
        return "Calendar access isn't configured yet. Please connect your calendar in the app settings."
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = _headers(ctx, content_type=True)
            cal_response = await client.get(f"{_NYLAS_BASE}/{grant}/calendars", headers=headers)
            if cal_response.status_code != 200:
                return "Couldn't access calendar"
            calendars = cal_response.json().get("data", [])
            if not calendars:
                return "No calendars found"
            calendar_id = pick_primary_calendar(calendars)
            events_url = f"{_NYLAS_BASE}/{grant}/events"
            events_response = await client.get(events_url, headers=headers, params={"calendar_id": calendar_id, "limit": 50})
            if events_response.status_code != 200:
                return "Couldn't fetch events"
            events = events_response.json().get("data", [])
            matching_event = None
            event_name_lower = event_name.lower()
            for event in events:
                title = event.get("title", "").lower()
                if event_name_lower in title or title in event_name_lower:
                    matching_event = event
                    break
            if not matching_event:
                return f"Couldn't find an event matching '{event_name}'. Please check the exact name."

            event_id = matching_event.get("id")
            event_title = matching_event.get("title", "Untitled")
            try:
                if "T" in new_date_time or (len(new_date_time) >= 19 and new_date_time.replace("-", "").replace(":", "").replace(" ", "").isdigit()):
                    new_dt = datetime.fromisoformat(new_date_time.replace("Z", "+00:00"))
                else:
                    new_dt = dateutil_parser.parse(new_date_time)
                new_start_time = new_dt.isoformat()

                original_when = matching_event.get("when", {})
                original_start = original_when.get("start_time")
                original_end = original_when.get("end_time")
                duration_minutes = 60
                if original_start and original_end:
                    try:
                        s = datetime.fromisoformat(original_start.replace("Z", "+00:00"))
                        e = datetime.fromisoformat(original_end.replace("Z", "+00:00"))
                        duration_minutes = int((e - s).total_seconds() / 60)
                    except Exception:
                        pass
                new_end_dt = new_dt + timedelta(minutes=duration_minutes)
                new_end_time = new_end_dt.isoformat()

                update_url = f"{_NYLAS_BASE}/{grant}/events/{event_id}"
                update_data = {"when": {"start_time": new_start_time, "end_time": new_end_time}}
                update_response = await client.put(update_url, headers=headers, json=update_data)
                if update_response.status_code in (200, 201):
                    await asyncio.to_thread(
                        _sync_praxa_loop_after_event_reschedule,
                        ctx, event_id, event_title, new_start_time, new_dt.date().isoformat(),
                    )
                    return f"Done! Moved '{event_title}' to {new_dt.strftime('%A %B %d at %I:%M %p')}"
                logger.error(f"Failed to update event: {update_response.status_code} - {update_response.text}")
                return "Couldn't reschedule the event. Please try again."
            except Exception as parse_error:
                logger.error(f"Error parsing date/time: {parse_error}")
                return f"Couldn't understand the date/time '{new_date_time}'. Please use a format like 'tomorrow at 2pm'."
    except Exception as e:
        logger.error(f"[reschedule_calendar_event] Error: {type(e).__name__}: {e}")
        return "Sorry, I couldn't reschedule that event"
