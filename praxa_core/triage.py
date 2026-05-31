"""4-tier inbox triage, shared by every surface.

Tiers (cheapest/most certain first; AI only for the genuinely ambiguous):
  1. Headers / heuristics  - obvious bulk/automated mail is skipped with no AI.
  2. Sender registry        - learned per-user verdicts (user_email_senders).
  3. VIP / relationship     - known VIP or reciprocal contact => attention.
  4. AI                     - classify only the remainder, then learn the result.

Attention-worthy verdicts are written to email_insights; sender verdicts are
learned back into user_email_senders so tier 2 keeps getting cheaper.
"""
import asyncio
import os
import time
from datetime import datetime, timezone

import httpx

from ._utils import logger
from .context import ToolContext, EMAIL_CACHE_TTL
from .email_filter import is_obviously_automated

_AI_MODEL = os.getenv("TRIAGE_MODEL", "gpt-4o-mini")


def _sender(msg: dict) -> tuple[str, str]:
    s = (msg.get("from", [{}]) or [{}])[0]
    return (s.get("email", "") or "").lower(), (s.get("name", "") or "")


def _header_automated(msg: dict) -> bool:
    """Tier 1: header-based bulk detection, falling back to the snippet heuristic."""
    headers = msg.get("headers") or {}
    if isinstance(headers, dict):
        lowered = {str(k).lower(): str(v).lower() for k, v in headers.items()}
        if "list-unsubscribe" in lowered:
            return True
        if lowered.get("precedence") in ("bulk", "list", "junk"):
            return True
        auto = lowered.get("auto-submitted", "no")
        if auto and auto != "no":
            return True
    email, _ = _sender(msg)
    return is_obviously_automated(email, msg.get("snippet", "") or "")


async def _load_sender_verdicts(ctx: ToolContext, emails: list[str]) -> dict[str, str]:
    if not ctx.supabase or not ctx.uid or not emails:
        return {}
    try:
        res = await asyncio.to_thread(
            lambda: ctx.supabase.table("user_email_senders")
            .select("sender_email, verdict")
            .eq("user_id", ctx.uid)
            .in_("sender_email", emails)
            .execute()
        )
        return {r["sender_email"]: r["verdict"] for r in (res.data or [])}
    except Exception as e:
        logger.warning(f"[triage] sender registry lookup failed: {e}")
        return {}


async def _load_vip_emails(ctx: ToolContext, emails: list[str]) -> set[str]:
    if not ctx.supabase or not ctx.uid or not emails:
        return set()
    try:
        res = await asyncio.to_thread(
            lambda: ctx.supabase.table("email_contacts")
            .select("email, is_vip")
            .eq("user_id", ctx.uid)
            .eq("is_vip", True)
            .in_("email", emails)
            .execute()
        )
        return {(r["email"] or "").lower() for r in (res.data or [])}
    except Exception as e:
        logger.warning(f"[triage] VIP lookup failed: {e}")
        return set()


async def _ai_classify(candidates: list[dict]) -> set[int]:
    """Return the set of indices (into candidates) judged attention-worthy."""
    key = os.getenv("OPENAI_API_KEY")
    if not key or not candidates:
        return set(range(len(candidates)))  # no AI -> be inclusive

    lines = []
    for i, msg in enumerate(candidates, 1):
        email, name = _sender(msg)
        subject = (msg.get("subject", "") or "")[:140]
        snippet = (msg.get("snippet", "") or "")[:260]
        lines.append(f"{i}. From: {name} <{email}> | Subject: {subject} | Preview: {snippet}")

    prompt = (
        "You triage a busy professional's inbox. Output P (PASS - needs the user's attention/reply) "
        "or A (SKIP - noise) for each line.\n"
        "PASS: a real person writing to the user expecting a reply, decision, or personal follow-up; "
        "1:1 or small-group threads; real meeting coordination.\n"
        "SKIP: newsletters, digests, marketing, promos, announcements, automated/transactional mail.\n"
        "If unsure and it reads like a blast, choose A.\n\n"
        + "\n".join(lines)
        + f"\n\nReply with exactly {len(candidates)} lines like '1. P' or '1. A'. Nothing else."
    )
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": _AI_MODEL, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": len(candidates) * 8, "temperature": 0},
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                passed: set[int] = set()
                for line in content.splitlines():
                    parts = line.strip().replace(".", " ").split()
                    if len(parts) >= 2 and parts[0].isdigit():
                        idx = int(parts[0]) - 1
                        if parts[-1].upper() == "P" and 0 <= idx < len(candidates):
                            passed.add(idx)
                return passed
    except Exception as e:
        logger.warning(f"[triage] AI classification failed, being inclusive: {e}")
    return set(range(len(candidates)))


async def _learn_sender(ctx: ToolContext, email: str, verdict: str, source: str, reason: str = "") -> None:
    if not ctx.supabase or not ctx.uid or not email:
        return
    try:
        await asyncio.to_thread(
            lambda: ctx.supabase.rpc("learn_email_sender", {
                "p_user_id": ctx.uid, "p_sender_email": email,
                "p_verdict": verdict, "p_source": source, "p_reason": reason or None,
            }).execute()
        )
    except Exception as e:
        logger.warning(f"[triage] learn_email_sender failed: {e}")


async def _upsert_matter(ctx: ToolContext, msg: dict) -> None:
    """Best-effort: track an attention-worthy email thread as an open matter.

    Deduped by (user_id, source_type, source_id) inside upsert_matter, so one
    matter per thread; repeat hits just bump last_activity. Never fatal.
    """
    thread_id = msg.get("thread_id")
    subject = (msg.get("subject") or "").strip()
    if not (ctx.supabase and ctx.uid and thread_id and subject):
        return
    try:
        await asyncio.to_thread(
            lambda: ctx.supabase.rpc("upsert_matter", {
                "p_user_id": ctx.uid,
                "p_title": subject[:160],
                "p_source_type": "email_thread",
                "p_source_id": thread_id,
                "p_description": (msg.get("snippet") or "")[:500] or None,
            }).execute()
        )
    except Exception as e:
        logger.warning(f"[triage] upsert_matter failed: {e}")


async def _write_insight(ctx: ToolContext, msg: dict, priority: float, source: str) -> None:
    if not ctx.supabase or not ctx.uid:
        return
    email, name = _sender(msg)
    email_id = msg.get("id")
    if not email_id:
        return
    try:
        # Avoid dupes: skip if we already recorded this email_id.
        existing = await asyncio.to_thread(
            lambda: ctx.supabase.table("email_insights")
            .select("id").eq("user_id", ctx.uid).eq("email_id", email_id).limit(1).execute()
        )
        if existing.data:
            return
        await asyncio.to_thread(
            lambda: ctx.supabase.table("email_insights").insert({
                "user_id": ctx.uid,
                "email_id": email_id,
                "from_email": email,
                "from_name": name,
                "subject": msg.get("subject", ""),
                "thread_id": msg.get("thread_id"),
                "snippet": (msg.get("snippet", "") or "")[:500],
                "insight_type": "needs_attention",
                "priority_score": priority,
                "action_suggested": f"triage:{source}",
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
                "is_addressed": False,
            }).execute()
        )
        # Track the thread as an open matter (conservative: only on new attention).
        await _upsert_matter(ctx, msg)
    except Exception as e:
        logger.warning(f"[triage] write insight failed: {e}")


async def triage_emails(ctx: ToolContext, emails: list, *, write_insights: bool = True) -> list:
    """Return the attention-worthy subset of `emails` using the 4-tier pipeline."""
    if not emails:
        return []

    incoming_ids = tuple(str(m.get("id") or "") for m in emails)
    if (
        ctx.email_classification_cache
        and ctx.email_classification_cache_ids == incoming_ids
        and (time.time() - ctx.email_classification_cache_at) < EMAIL_CACHE_TTL
    ):
        return ctx.email_classification_cache

    sender_emails = sorted({_sender(m)[0] for m in emails if _sender(m)[0]})
    registry = await _load_sender_verdicts(ctx, sender_emails)
    vips = await _load_vip_emails(ctx, sender_emails)

    passed: list[dict] = []
    ambiguous: list[dict] = []

    for msg in emails:
        email, _ = _sender(msg)

        # Tier 1: headers / heuristics
        if _header_automated(msg):
            await _learn_sender(ctx, email, "skip", "header", "automated/bulk headers")
            continue

        # Tier 2: learned sender registry
        verdict = registry.get(email)
        if verdict == "skip":
            continue
        if verdict == "attention":
            passed.append(msg)
            continue

        # Tier 3: VIP / relationship
        if email in vips:
            passed.append(msg)
            await _learn_sender(ctx, email, "attention", "vip", "VIP/reciprocal contact")
            continue

        # Tier 4: defer to AI
        ambiguous.append(msg)

    if ambiguous:
        ai_pass = await _ai_classify(ambiguous)
        for i, msg in enumerate(ambiguous):
            email, _ = _sender(msg)
            if i in ai_pass:
                passed.append(msg)
                await _learn_sender(ctx, email, "attention", "ai")
            else:
                await _learn_sender(ctx, email, "skip", "ai")

    if write_insights:
        for msg in passed:
            await _write_insight(ctx, msg, priority=0.7, source="triage")

    ctx.email_classification_cache = passed
    ctx.email_classification_cache_ids = incoming_ids
    ctx.email_classification_cache_at = time.time()
    return passed
