"""Inbox triage. P1 ports the voice agent's reply-worthy filter to be
context-based; P2 (`triage.py`) layers a 4-tier classifier on top of these
primitives. Kept here so every surface shares one notion of "worth attention".
"""
import os
import time

import httpx

from ._utils import logger
from .context import ToolContext, EMAIL_CACHE_TTL

_PRAXA_SENDER_EMAIL: str = os.getenv("SENDGRID_FROM_EMAIL", "").lower().strip()

_AUTOMATED_LOCAL_PREFIXES = (
    "newsletter", "notifications", "notification", "marketing", "promo",
    "bounce", "mailing", "digest", "mailer",
)
_AUTOMATED_ADDRESS_KEYWORDS = (
    "noreply", "no-reply", "donotreply", "do-not-reply", "mailer-daemon", "postmaster",
)
_AUTOMATED_SNIPPET_KEYWORDS = (
    "unsubscribe", "opt out", "manage your email preferences", "view in browser", "list-unsubscribe",
)


def is_obviously_automated(from_email: str, snippet: str) -> bool:
    """Fast pre-filter for emails that are obviously bulk/automated - no AI needed."""
    from_email = (from_email or "").lower()
    snippet = (snippet or "").lower()

    if _PRAXA_SENDER_EMAIL and _PRAXA_SENDER_EMAIL in from_email:
        return True
    if any(kw in from_email for kw in _AUTOMATED_ADDRESS_KEYWORDS):
        return True

    local = from_email.split("@", 1)[0] if "@" in from_email else from_email
    if any(local.startswith(p) for p in _AUTOMATED_LOCAL_PREFIXES):
        return True
    if any(kw in snippet for kw in _AUTOMATED_SNIPPET_KEYWORDS):
        return True
    return False


async def filter_reply_worthy_emails(ctx: ToolContext, emails: list) -> list:
    """Keep emails where a human would reasonably need to read or reply.

    When a DB-backed context is available this delegates to the 4-tier triage
    pipeline (headers -> sender registry -> VIP/relationship -> AI), which also
    learns sender verdicts and records insights. Falls back to the AI-only path
    (below) when there's no Supabase context.
    """
    if not emails:
        return []

    if ctx.supabase and ctx.uid:
        try:
            from .triage import triage_emails
            return await triage_emails(ctx, emails)
        except Exception as e:
            logger.warning(f"4-tier triage failed, falling back to AI-only filter: {e}")

    incoming_ids = tuple(str(m.get("id") or "") for m in emails)

    if (
        ctx.email_classification_cache
        and ctx.email_classification_cache_ids == incoming_ids
        and (time.time() - ctx.email_classification_cache_at) < EMAIL_CACHE_TTL
    ):
        logger.info("Returning cached email classification")
        return ctx.email_classification_cache

    candidates = [
        msg
        for msg in emails
        if not is_obviously_automated(
            (msg.get("from", [{}]) or [{}])[0].get("email", ""),
            msg.get("snippet", "") or "",
        )
    ]
    if not candidates:
        return []

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return candidates

    lines = []
    for i, msg in enumerate(candidates, 1):
        sender = (msg.get("from", [{}]) or [{}])[0]
        from_name = sender.get("name", "") or ""
        from_addr = sender.get("email", "") or ""
        subject = (msg.get("subject", "") or "")[:140]
        snippet = (msg.get("snippet", "") or "")[:260]
        lines.append(f"{i}. From: {from_name} <{from_addr}> | Subject: {subject} | Preview: {snippet}")

    prompt = (
        "You filter inbox messages for a voice assistant. The user only wants to hear about mail "
        "where a real person would reasonably need to read or reply - not broadcast noise.\n\n"
        "For each line, output P (PASS - include) or A (SKIP - exclude).\n\n"
        "PASS (P) only when:\n"
        "- Someone is writing to the user in a way that expects a reply, decision, or personal follow-up.\n"
        "- Direct 1:1 or small-group thread, real meeting coordination, or a question clearly aimed at the user.\n\n"
        "SKIP (A) for:\n"
        "- Newsletters, digests, marketing, promos, event announcements, workshop invites, or program roundups.\n"
        "- Career office / campus / institutional announcements or generic event mailings.\n"
        "- Automated transactional mail: receipts, alerts, shipping, password resets, billing, no-reply flows.\n"
        "- Anything that feels like a mailing list or blast rather than a message to the user personally.\n\n"
        "If unsure: if it sounds like a newsletter or mass announcement, choose A. If a person wrote the user "
        "to continue a conversation, choose P.\n\n"
        + "\n".join(lines)
        + f"\n\nReply with exactly {len(candidates)} lines. Format: '1. P' or '1. A'. Nothing else."
    )

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": len(candidates) * 8,
                    "temperature": 0,
                },
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                passed = []
                for line in content.splitlines():
                    parts = line.strip().replace(".", " ").split()
                    if len(parts) >= 2 and parts[0].isdigit():
                        idx = int(parts[0]) - 1
                        if parts[-1].upper() == "P" and 0 <= idx < len(candidates):
                            passed.append(candidates[idx])

                ctx.email_classification_cache = passed
                ctx.email_classification_cache_ids = incoming_ids
                ctx.email_classification_cache_at = time.time()
                return passed
    except Exception as e:
        logger.warning(f"AI email classification failed, using pre-filtered results: {e}")

    return candidates


def resolve_email_id(ctx: ToolContext, ref: str):
    """Resolve a user/agent reference to a Nylas message ID.

    Accepts a 1-based index, ordinal word, subject/sender snippet, or raw ID.
    """
    if not ref:
        return None

    emails = ctx.last_fetched_emails or []
    ordinals = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
    ref_lower = ref.strip().lower()

    if ref_lower in ordinals:
        idx = ordinals[ref_lower] - 1
        if 0 <= idx < len(emails):
            return emails[idx].get("id")

    try:
        idx = int(ref_lower) - 1
        if 0 <= idx < len(emails):
            return emails[idx].get("id")
    except ValueError:
        pass

    for msg in emails:
        subject = (msg.get("subject") or "").lower()
        sender = (msg.get("from", [{}]) or [{}])[0]
        sender_name = (sender.get("name") or "").lower()
        sender_email = (sender.get("email") or "").lower()
        if ref_lower in subject or ref_lower in sender_name or ref_lower in sender_email:
            return msg.get("id")

    return ref
