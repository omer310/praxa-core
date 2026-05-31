"""Email tool implementations (Nylas). Reads are direct; sending a reply is a
CONFIRM action, so `reply_to_email_impl` queues it through the action pipeline
unless `direct=True` is passed (used by the dispatcher to actually send)."""
import re

import httpx

from .._utils import logger
from ..context import ToolContext
from ..email_filter import filter_reply_worthy_emails, resolve_email_id

_NYLAS_BASE = "https://api.us.nylas.com/v3/grants"


def _no_signin() -> str:
    return "I can't access your emails right now — it looks like you're not signed in. Please open the app and sign in first."


def _headers(ctx: ToolContext, content_type: bool = False) -> dict:
    h = {"Authorization": f"Bearer {ctx.nylas_api_key}", "Accept": "application/json"}
    if content_type:
        h["Content-Type"] = "application/json"
    return h


def _format_listing(messages: list, limit: int) -> str:
    email_list = []
    for i, msg in enumerate(messages[:limit], 1):
        msg_id = msg.get("id", "")
        sender = (msg.get("from", [{}]) or [{}])[0]
        sender_name = sender.get("name") or sender.get("email", "Unknown")
        sender_email_addr = sender.get("email", "")
        subject = msg.get("subject", "No subject")
        snippet = (msg.get("snippet", "") or "")[:150]
        entry = f"[{i}] ID:{msg_id} | From: {sender_name} <{sender_email_addr}> | Subject: {subject}"
        if snippet:
            entry += f" | Preview: {snippet}"
        email_list.append(entry)
    return "\n".join(email_list)


async def get_recent_emails_impl(ctx: ToolContext, limit: int = 5) -> str:
    if not ctx.uid:
        return _no_signin()
    grant = await ctx.ensure_email_grant()
    if not grant:
        return "Email access isn't configured yet. Please connect your email in the app settings."
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{_NYLAS_BASE}/{grant}/messages", headers=_headers(ctx),
                params={"limit": limit * 3, "in": "INBOX"},
            )
            if response.status_code != 200:
                logger.error(f"Nylas email API error: {response.status_code}")
                return "Couldn't fetch emails"
            messages = response.json().get("data", [])
            important = (await filter_reply_worthy_emails(ctx, messages))[:limit]
            if not important:
                return "No recent important emails that need your attention."
            ctx.last_fetched_emails = important
            return (
                f"Found {len(important)} recent important emails. "
                "Call get_email_body with the number (e.g. '1') or subject to read any email in full.\n\n"
                + _format_listing(important, limit)
            )
    except Exception as e:
        logger.error(f"Error getting emails: {e}")
        return "Sorry, I couldn't access your emails"


async def get_emails_needing_response_impl(ctx: ToolContext) -> str:
    if not ctx.uid:
        return _no_signin()
    grant = await ctx.ensure_email_grant()
    if not grant:
        return "Email access isn't configured yet. Please connect your email in the app settings."
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{_NYLAS_BASE}/{grant}/messages", headers=_headers(ctx),
                params={"limit": 20, "unread": "true", "in": "INBOX"},
            )
            if response.status_code != 200:
                logger.error(f"Nylas email API error: {response.status_code}")
                return "Couldn't fetch emails"
            messages = response.json().get("data", [])
            needs_response = await filter_reply_worthy_emails(ctx, messages)
            if not needs_response:
                return "No unread emails from real people right now. Your inbox looks clear."
            ctx.last_fetched_emails = needs_response[:7]
            return (
                f"Here are {len(needs_response[:7])} emails that may need a response. "
                "Call get_email_body with the number (e.g. '1') or subject to read the full email.\n\n"
                + _format_listing(needs_response, 7)
            )
    except Exception as e:
        logger.error(f"Error checking emails: {e}")
        return "Sorry, I couldn't check for emails needing response"


async def get_unread_emails_impl(ctx: ToolContext) -> str:
    if not ctx.uid:
        return _no_signin()
    grant = await ctx.ensure_email_grant()
    if not grant:
        return "Email access isn't configured yet. Please connect your email in the app settings."
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{_NYLAS_BASE}/{grant}/messages", headers=_headers(ctx),
                params={"unread": "true", "limit": 20, "in": "INBOX"},
            )
            if response.status_code != 200:
                logger.error(f"Nylas email API error: {response.status_code}")
                return "Couldn't fetch unread emails"
            messages = response.json().get("data", [])
            important_unread = await filter_reply_worthy_emails(ctx, messages)
            count = len(important_unread)
            if count == 0:
                return "No unread important emails that need your attention."
            ctx.last_fetched_emails = important_unread[:5]
            result = (
                f"Found {count} unread important email{'s' if count > 1 else ''} that may need your attention. "
                "Call get_email_body with the number (e.g. '1') to read the full email.\n\n"
                + _format_listing(important_unread, 5)
            )
            if count > 5:
                result += f"\n(and {count - 5} more)"
            return result
    except Exception as e:
        logger.error(f"Error getting unread emails: {e}")
        return "Sorry, I couldn't get unread emails"


async def search_emails_impl(ctx: ToolContext, search_term: str) -> str:
    if not ctx.uid:
        return _no_signin()
    grant = await ctx.ensure_email_grant()
    if not grant:
        return "Email access isn't configured yet. Please connect your email in the app settings."
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{_NYLAS_BASE}/{grant}/messages", headers=_headers(ctx), params={"limit": 50},
            )
            if response.status_code != 200:
                return f"Couldn't search for emails matching '{search_term}'"
            all_messages = response.json().get("data", [])
            term = search_term.lower()
            matching = []
            for msg in all_messages:
                from_info = (msg.get("from", [{}]) or [{}])[0]
                if (term in (from_info.get("name") or "").lower()
                        or term in (from_info.get("email") or "").lower()
                        or term in (msg.get("subject") or "").lower()):
                    matching.append(msg)
            if not matching:
                search_response = await client.get(
                    f"{_NYLAS_BASE}/{grant}/messages", headers=_headers(ctx),
                    params={"search": search_term, "limit": 20},
                )
                if search_response.status_code == 200:
                    matching = search_response.json().get("data", []) or matching
            if not matching:
                return f"I couldn't find any emails from or about '{search_term}'. They might be in an older conversation."
            email_list = []
            for msg in matching[:10]:
                from_info = (msg.get("from", [{}]) or [{}])[0]
                sender_name = from_info.get("name") or from_info.get("email", "Unknown")
                sender_email = from_info.get("email", "")
                subject = msg.get("subject", "No subject")
                date = msg.get("date", "")
                sender_display = f"{sender_name} ({sender_email})" if (sender_email and sender_name != sender_email) else (sender_name or sender_email)
                email_list.append(f"{sender_display}: {subject}" + (f" ({date})" if date else ""))
            result = f"Found {len(matching)} email" + ("s" if len(matching) > 1 else "")
            result += (" (showing 10): " if len(matching) > 10 else ": ") + "; ".join(email_list)
            return result
    except Exception as e:
        logger.error(f"Error searching emails: {type(e).__name__}: {e}")
        return f"Sorry, I couldn't search for '{search_term}'"


async def get_email_body_impl(ctx: ToolContext, email_ref: str) -> str:
    if not ctx.uid:
        return "I can't access your emails right now — it looks like you're not signed in."
    grant = await ctx.ensure_email_grant()
    if not grant:
        return "Email access isn't configured yet. Please connect your email in the app settings."
    email_id = resolve_email_id(ctx, email_ref)
    if not email_id:
        return (
            "I couldn't identify which email you mean. Ask me to list your emails first, "
            "then say 'read email 1' or mention the sender's name."
        )
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{_NYLAS_BASE}/{grant}/messages/{email_id}", headers=_headers(ctx),
            )
            if response.status_code == 404:
                return f"Couldn't find an email with ID {email_id}. It may have been deleted."
            if response.status_code != 200:
                return "Couldn't retrieve the email content."
            msg = response.json().get("data", {})
            sender_info = (msg.get("from", [{}]) or [{}])[0]
            sender_name = sender_info.get("name") or sender_info.get("email", "Unknown")
            sender_email_addr = sender_info.get("email", "")
            subject = msg.get("subject", "No subject")
            date = msg.get("date", "")
            body = msg.get("body", "") or ""
            if body:
                body = re.sub(r"<[^>]+>", " ", body)
                body = re.sub(r"[ \t]+", " ", body)
                body = re.sub(r"\n{3,}", "\n\n", body).strip()
            else:
                body = msg.get("snippet", "") or "No content available."
            result = f"Email from {sender_name} <{sender_email_addr}>\nSubject: {subject}\n"
            if date:
                result += f"Date: {date}\n"
            result += f"\n{body}"
            return result
    except Exception as e:
        logger.error(f"Error getting email body: {e}")
        return "Sorry, I couldn't load that email."


async def send_email_reply_direct(ctx: ToolContext, email_id: str, reply_body: str) -> str:
    """Actually send a reply via Nylas. Used by the dispatcher after confirmation."""
    grant = await ctx.ensure_email_grant()
    if not grant:
        return "Email access isn't configured yet."
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            orig_response = await client.get(
                f"{_NYLAS_BASE}/{grant}/messages/{email_id}", headers=_headers(ctx),
            )
            if orig_response.status_code != 200:
                return "Couldn't find the original email to reply to."
            orig = orig_response.json().get("data", {})
            orig_from = (orig.get("from", [{}]) or [{}])[0]
            orig_subject = orig.get("subject", "")
            reply_to = orig.get("reply_to") or [orig_from]
            subject = orig_subject if orig_subject.lower().startswith("re:") else f"Re: {orig_subject}"
            send_payload = {
                "subject": subject,
                "to": reply_to,
                "reply_to_message_id": email_id,
                "body": reply_body,
            }
            send_response = await client.post(
                f"{_NYLAS_BASE}/{grant}/messages/send", headers=_headers(ctx, content_type=True),
                json=send_payload,
            )
            if send_response.status_code in (200, 201):
                recipient_name = reply_to[0].get("name") or reply_to[0].get("email", "them")
                return f"Reply sent to {recipient_name}."
            logger.error(f"Nylas send reply error: {send_response.status_code} - {send_response.text[:300]}")
            return "Couldn't send the reply. The email service returned an error."
    except Exception as e:
        logger.error(f"Error sending reply: {e}")
        return "Sorry, I couldn't send that reply."


async def reply_to_email_impl(ctx: ToolContext, email_id: str, reply_body: str, direct: bool = False) -> str:
    """Reply to an email. By default this is a CONFIRM action: it queues the
    send through the action pipeline so the user reviews/approves it. Set
    `direct=True` (dispatcher only) to send immediately."""
    if not ctx.uid:
        return "I can't send emails right now — it looks like you're not signed in."
    email_id = resolve_email_id(ctx, email_id) or email_id
    if direct:
        return await send_email_reply_direct(ctx, email_id, reply_body)

    from ..actions import queue_action

    # Build a descriptive summary for push/SMS.
    recipient = ""
    subject = ""
    for msg in ctx.last_fetched_emails or []:
        if msg.get("id") == email_id:
            sender = (msg.get("from", [{}]) or [{}])[0]
            recipient = sender.get("name") or sender.get("email", "")
            subject = msg.get("subject", "")
            break
    summary = f"Reply to {recipient or 'an email'}" + (f" re: {subject}" if subject else "")
    await queue_action(
        ctx, "email", "reply_to_email",
        {"email_id": email_id, "reply_body": reply_body, "recipient": recipient, "subject": subject},
        summary=summary,
    )
    return f"I've drafted that reply{(' to ' + recipient) if recipient else ''}. I'll send it once you approve it."
