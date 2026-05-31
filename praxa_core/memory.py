"""Unified context retrieval across user facts, session summaries, integration
context (Notion/Slack) and the relationship graph.

`get_relevant_context` embeds the query once and fans out to the pgvector RPCs
(`match_user_facts`, `match_session_summaries`, `match_integration_context`),
then assembles a single context block usable by any surface. Falls back to
recency when embeddings are unavailable.
"""
import asyncio
import os

import httpx

from ._utils import logger
from .context import ToolContext

EMBED_MODEL = "text-embedding-3-small"


async def embed_text(text: str) -> list[float] | None:
    """Embed text with OpenAI. Returns None on failure (callers should fall back)."""
    key = os.getenv("OPENAI_API_KEY")
    if not key or not text:
        return None
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": EMBED_MODEL, "input": text[:8000]},
            )
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
            logger.warning(f"Embedding API returned {resp.status_code}")
    except Exception as e:
        logger.warning(f"Embedding call failed: {e}")
    return None


async def _rpc(ctx: ToolContext, name: str, params: dict):
    sb = ctx.supabase
    if not sb:
        return []
    try:
        res = await asyncio.to_thread(lambda: sb.rpc(name, params).execute())
        return res.data or []
    except Exception as e:
        logger.warning(f"RPC {name} failed: {e}")
        return []


async def get_integration_context(
    ctx: ToolContext,
    query: str,
    *,
    integration_count: int = 6,
    threshold: float = 0.3,
) -> str:
    """Return ONLY a Markdown block of semantic matches from connected tools
    (Notion/Slack). Use when facts/summaries are loaded separately by a surface."""
    sb, uid = ctx.supabase, ctx.uid
    if not sb or not uid:
        return ""
    embedding = await embed_text(query)
    if not embedding:
        return ""
    rows = await _rpc(ctx, "match_integration_context", {
        "p_user_id": uid, "p_query_embedding": embedding,
        "p_match_count": integration_count, "p_match_threshold": threshold,
    })
    if not rows:
        return ""
    lines = ["### From your connected tools"]
    for r in rows:
        provider = (r.get("provider") or "").capitalize()
        title = r.get("title") or "Untitled"
        content = (r.get("content") or "")[:200]
        lines.append(f"- [{provider}] {title}: {content}")
    return "\n".join(lines)


async def get_relevant_context(
    ctx: ToolContext,
    query: str,
    *,
    fact_count: int = 12,
    summary_count: int = 4,
    integration_count: int = 6,
    matter_count: int = 4,
    threshold: float = 0.3,
) -> str:
    """Return a single Markdown context block relevant to `query`.

    Unions semantic matches across facts, recent session summaries and synced
    integration content. Used by every surface to ground the agent.
    """
    sb, uid = ctx.supabase, ctx.uid
    if not sb or not uid:
        return ""

    embedding = await embed_text(query)

    facts: list[dict] = []
    summaries: list[dict] = []
    integration_rows: list[dict] = []
    matters: list[dict] = []

    if embedding:
        facts_task = _rpc(ctx, "match_user_facts", {
            "p_user_id": uid, "p_query_embedding": embedding,
            "p_match_count": fact_count, "p_match_threshold": threshold,
        })
        summaries_task = _rpc(ctx, "match_session_summaries", {
            "p_user_id": uid, "p_query_embedding": embedding,
            "p_match_count": summary_count, "p_match_threshold": threshold,
        })
        integration_task = _rpc(ctx, "match_integration_context", {
            "p_user_id": uid, "p_query_embedding": embedding,
            "p_match_count": integration_count, "p_match_threshold": threshold,
        })
        matters_task = _rpc(ctx, "match_matters", {
            "p_user_id": uid, "p_query_embedding": embedding,
            "p_match_count": matter_count, "p_match_threshold": threshold,
        })
        facts, summaries, integration_rows, matters = await asyncio.gather(
            facts_task, summaries_task, integration_task, matters_task
        )

    if not matters:
        # recency fallback: surface recent open matters even before embeddings exist
        try:
            mres = await asyncio.to_thread(
                lambda: sb.table("user_matters").select("title, description, status")
                .eq("user_id", uid).order("last_activity_at", desc=True).limit(matter_count + 5).execute()
            )
            matters = [m for m in (mres.data or []) if (m.get("status") or "open") not in ("closed", "done", "archived")][:matter_count]
        except Exception:
            pass

    if not facts and not summaries:
        # recency fallback
        try:
            facts_res = await asyncio.to_thread(
                lambda: sb.table("user_facts").select("fact_key, fact_value")
                .eq("user_id", uid).order("last_confirmed_at", desc=True).limit(fact_count).execute()
            )
            facts = facts_res.data or []
        except Exception:
            pass
        try:
            sums_res = await asyncio.to_thread(
                lambda: sb.table("session_summaries").select("summary, surface")
                .eq("user_id", uid).neq("surface", "compressed").eq("is_archived", False)
                .order("created_at", desc=True).limit(summary_count).execute()
            )
            summaries = sums_res.data or []
        except Exception:
            pass

    if not (facts or summaries or integration_rows or matters):
        return ""

    lines: list[str] = ["## Relevant Context\n"]
    if facts:
        lines.append("### What I know about you")
        for f in facts:
            key = f.get("fact_key") or f.get("key") or ""
            val = f.get("fact_value") or f.get("value") or ""
            lines.append(f"- {key}: {val}")
        lines.append("")
    if summaries:
        lines.append("### Recent sessions")
        for s in summaries:
            surface = s.get("surface", "")
            lines.append(f"- [{surface}] {s.get('summary', '')}")
        lines.append("")
    if integration_rows:
        lines.append("### From your connected tools")
        for r in integration_rows:
            provider = (r.get("provider") or "").capitalize()
            title = r.get("title") or "Untitled"
            content = (r.get("content") or "")[:200]
            lines.append(f"- [{provider}] {title}: {content}")
        lines.append("")
    if matters:
        lines.append("### Open matters")
        for m in matters:
            title = m.get("title") or "Untitled"
            desc = (m.get("description") or "")[:160]
            lines.append(f"- {title}" + (f": {desc}" if desc else ""))
        lines.append("")
    return "\n".join(lines)
