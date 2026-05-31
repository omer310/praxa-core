"""Notion / Slack tool implementations.

Reads (search, unread) go direct. Writes (create/update page, send message) are
CONFIRM actions queued through the action pipeline and executed by the
dispatcher via n8n -- this removes the old "split-brain" behaviour where
`create_notion_page` both queued AND called the Notion API directly.
"""
import asyncio

import httpx

from .._utils import logger
from ..context import ToolContext
from ..actions import queue_action


async def _fetch_integration_token(ctx: ToolContext, provider: str) -> str | None:
    sb, uid = ctx.supabase, ctx.uid
    if not sb or not uid:
        return None
    try:
        result = await asyncio.to_thread(
            lambda: sb.table("user_integrations")
            .select("access_token")
            .eq("user_id", uid)
            .eq("provider", provider)
            .eq("status", "connected")
            .maybe_single()
            .execute()
        )
        if result and result.data:
            return result.data.get("access_token")
    except Exception as e:
        logger.warning(f"[integration_tools] Could not fetch {provider} token: {e}")
    return None


def _extract_notion_title(page: dict) -> str:
    props = page.get("properties", {})
    for key in ("title", "Name", "Title"):
        if key in props:
            title_arr = props[key].get("title", [])
            if title_arr:
                return title_arr[0].get("plain_text", "Untitled")
    title_arr = page.get("title", [])
    if title_arr:
        return title_arr[0].get("plain_text", "Untitled")
    return page.get("id", "Untitled")


async def search_notion_impl(ctx: ToolContext, query: str) -> str:
    token = await _fetch_integration_token(ctx, "notion")
    if not token:
        return "Notion is not connected. The user can connect it from their profile settings."
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                "https://api.notion.com/v1/search",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Notion-Version": "2022-06-28",
                    "Content-Type": "application/json",
                },
                json={"query": query, "page_size": 10},
            )
            if resp.status_code != 200:
                return f"Notion search failed (status {resp.status_code})."
            results = resp.json().get("results", [])
            if not results:
                return f"No Notion pages found for '{query}'."
            lines = [f"Found {len(results)} Notion result(s) for '{query}':"]
            for r in results:
                title = _extract_notion_title(r)
                url = r.get("url", "")
                last_edited = r.get("last_edited_time", "")[:10]
                lines.append(f"- {title} (edited {last_edited}) — {url}")
            return "\n".join(lines)
    except Exception as e:
        logger.error(f"[search_notion] Error: {e}")
        return "Could not search Notion right now. Please try again."


async def create_notion_page_impl(ctx: ToolContext, title: str, content: str, parent_page_id: str = "") -> str:
    if not ctx.uid:
        return "No active session."
    token = await _fetch_integration_token(ctx, "notion")
    if not token:
        return "Notion is not connected. The user can connect it from their profile settings."

    page_body = {
        "parent": {"type": "page_id", "page_id": parent_page_id} if parent_page_id else {"type": "workspace", "workspace": True},
        "properties": {"title": {"title": [{"type": "text", "text": {"content": title}}]}},
        "children": [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": content}}]},
            }
        ],
    }
    await queue_action(
        ctx, "notion", "create_notion_page",
        {"title": title, "content": content, "parent_page_id": parent_page_id, "body": page_body},
        summary=f"Create Notion page: {title}",
    )
    return f"I've drafted a Notion page titled '{title}'. I'll create it once you approve it."


async def update_notion_page_impl(ctx: ToolContext, page_id: str, new_title: str = "", append_text: str = "") -> str:
    if not ctx.uid:
        return "No active session."
    token = await _fetch_integration_token(ctx, "notion")
    if not token:
        return "Notion is not connected. The user can connect it from their profile settings."
    if not new_title and not append_text:
        return "Nothing to update — tell me a new title or text to append."
    descriptor = []
    if new_title:
        descriptor.append(f"rename to '{new_title}'")
    if append_text:
        descriptor.append("append text")
    await queue_action(
        ctx, "notion", "update_notion_page",
        {"page_id": page_id, "new_title": new_title, "append_text": append_text},
        summary=f"Update Notion page ({', '.join(descriptor)})",
    )
    return "I've prepared that Notion update. I'll apply it once you approve it."


async def send_slack_message_impl(ctx: ToolContext, channel: str, message: str) -> str:
    if not ctx.uid:
        return "No active session."
    token = await _fetch_integration_token(ctx, "slack")
    if not token:
        return "Slack is not connected. The user can connect it from their profile settings."
    channel_id = channel if channel.startswith(("C", "D", "G", "U", "#")) else f"#{channel}"
    preview = message[:60] + ("…" if len(message) > 60 else "")
    await queue_action(
        ctx, "slack", "send_slack_message",
        {"channel": channel_id, "message": message},
        summary=f"Send Slack message to {channel}: {preview}",
    )
    return f"I've drafted that Slack message to {channel}. I'll send it once you approve it."


async def get_slack_unread_impl(ctx: ToolContext, limit: int = 15) -> str:
    sb, uid = ctx.supabase, ctx.uid
    if not sb or not uid:
        return "No active session."
    token = await _fetch_integration_token(ctx, "slack")
    if not token:
        return "Slack is not connected. The user can connect it from their profile settings."
    try:
        result = await asyncio.to_thread(
            lambda: sb.table("integration_context")
            .select("title, content, source_updated_at, metadata")
            .eq("user_id", uid)
            .eq("provider", "slack")
            .order("source_updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = result.data if result else []
        if not rows:
            return "No recent Slack messages found. Messages will appear here as they arrive."
        lines = [f"Recent Slack messages ({len(rows)}):"]
        for row in rows:
            meta = row.get("metadata") or {}
            channel = meta.get("channel", row.get("title", "unknown"))
            ts_raw = row.get("source_updated_at", "")
            ts = ts_raw[:16].replace("T", " ") if ts_raw else ""
            text = (row.get("content") or "")[:150]
            lines.append(f"- #{channel} [{ts}]: {text}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"[get_slack_unread] Error: {e}")
        return "Could not load Slack messages right now. Please try again."
