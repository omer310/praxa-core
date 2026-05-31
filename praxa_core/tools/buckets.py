"""Bucket (initiative) tool implementations. Pure logic, no LiveKit."""
import asyncio
from datetime import datetime

from .._utils import logger, sanitize_sql_like_pattern, infer_bucket_style
from ..context import ToolContext


def _not_signed_in() -> str:
    return "I can't access your buckets right now — sign in to the app first."


async def _run(fn):
    return await asyncio.to_thread(fn)


async def get_all_buckets_impl(ctx: ToolContext) -> str:
    if not ctx.uid:
        return _not_signed_in()
    try:
        sb, uid = ctx.supabase, ctx.uid
        response = await _run(lambda: sb.table("buckets").select("name, description, goal").eq(
            "user_id", uid
        ).eq("archived", False).execute())
        if not response.data:
            return "No buckets found."
        bucket_names = [b["name"] for b in response.data]
        result = f"Available buckets: {', '.join(bucket_names)}"
        details = []
        for b in response.data:
            if b.get("description") or b.get("goal"):
                context_info = b.get("description") or b.get("goal") or ""
                details.append(f"{b['name']} ({context_info})" if context_info else b["name"])
        if details:
            result += ". Details: " + "; ".join(details)
        return result
    except Exception as e:
        logger.error(f"Error getting buckets: {e}")
        return "Sorry, I couldn't fetch your buckets"


async def get_bucket_goal_impl(ctx: ToolContext, bucket_name: str) -> str:
    if not ctx.uid:
        return _not_signed_in()
    try:
        sb, uid = ctx.supabase, ctx.uid
        safe_bucket_name = sanitize_sql_like_pattern(bucket_name)
        response = await _run(lambda: sb.table("buckets").select("goal, description").eq(
            "user_id", uid
        ).ilike("name", safe_bucket_name).eq("archived", False).execute())
        if not response.data:
            return f"Couldn't find bucket '{bucket_name}'"
        bucket = response.data[0]
        goal = bucket.get("goal", "No goal set")
        desc = bucket.get("description", "")
        result = f"Goal for {bucket_name}: {goal}"
        if desc:
            result += f"\nDescription: {desc}"
        return result
    except Exception as e:
        logger.error(f"Error getting bucket goal: {e}")
        return f"Sorry, I couldn't get the goal for {bucket_name}"


async def create_bucket_impl(
    ctx: ToolContext,
    name: str,
    description: str | None = None,
    goal: str | None = None,
) -> str:
    if not ctx.uid:
        return _not_signed_in()
    sb, uid = ctx.supabase, ctx.uid
    if not sb:
        return "Sorry, I couldn't create that initiative — the assistant isn't fully connected to your data yet."
    try:
        icon, color = infer_bucket_style(name, goal, description)
        new_bucket = {
            "user_id": uid, "name": name, "description": description,
            "goal": goal, "color": color, "icon": icon,
        }
        response = await _run(lambda: sb.table("buckets").insert(new_bucket).execute())
        if not response.data:
            return "Sorry, I couldn't create the bucket."
        return f"Created bucket: {name}" + (f" with goal: {goal}" if goal else "")
    except Exception as e:
        logger.error(f"Error creating bucket: {e}", exc_info=True)
        err_s = str(e).lower()
        if "row-level security" in err_s or "42501" in str(e):
            return "Sorry, I couldn't save that initiative due to a permissions issue."
        return "Sorry, I couldn't create the bucket"


async def update_bucket_impl(
    ctx: ToolContext,
    bucket_name: str,
    goal: str | None = None,
    description: str | None = None,
) -> str:
    if not ctx.uid:
        return _not_signed_in()
    try:
        sb, uid = ctx.supabase, ctx.uid
        safe_bucket_name = sanitize_sql_like_pattern(bucket_name)
        updates: dict = {}
        changes: list[str] = []
        if goal is not None:
            updates["goal"] = goal
            changes.append("goal updated")
        if description is not None:
            updates["description"] = description
            changes.append("description updated")
        if not updates:
            return "Nothing to update — tell me what you'd like to change."
        updates["updated_at"] = datetime.now().isoformat()
        response = await _run(lambda: sb.table("buckets").update(updates).eq(
            "user_id", uid
        ).ilike("name", f"%{safe_bucket_name}%").execute())
        if response.data:
            return f"Updated '{bucket_name}': {', '.join(changes)}."
        return f"I couldn't find an initiative matching '{bucket_name}'"
    except Exception as e:
        logger.error(f"Error updating bucket: {e}", exc_info=True)
        return "Sorry, I couldn't update that initiative"


async def rename_bucket_impl(ctx: ToolContext, current_name: str, new_name: str) -> str:
    if not ctx.uid:
        return _not_signed_in()
    sb, uid = ctx.supabase, ctx.uid
    if not sb:
        return "Sorry, I couldn't rename that initiative right now."
    try:
        safe_old = sanitize_sql_like_pattern(current_name)
        response = await _run(lambda: sb.table("buckets").update({
            "name": new_name, "updated_at": datetime.now().isoformat(),
        }).eq("user_id", uid).ilike("name", f"%{safe_old}%").eq("archived", False).execute())
        if response.data:
            return f"Renamed that initiative to '{new_name}'."
        return f"I couldn't find an initiative matching '{current_name}'."
    except Exception as e:
        logger.error(f"Error renaming bucket: {e}", exc_info=True)
        err_s = str(e).lower()
        if "row-level security" in err_s or "42501" in str(e):
            return "Sorry, I couldn't save that rename due to a permissions issue."
        if "unique" in err_s or "23505" in str(e) or "duplicate" in err_s:
            return "That name is already used. Try a different name."
        return "Sorry, I couldn't rename the initiative"


async def archive_bucket_impl(ctx: ToolContext, bucket_name: str, archived: bool) -> str:
    if not ctx.uid:
        return _not_signed_in()
    sb, uid = ctx.supabase, ctx.uid
    if not sb:
        return "Sorry, I couldn't update that initiative right now."
    try:
        safe_name = sanitize_sql_like_pattern(bucket_name)
        response = await _run(lambda: sb.table("buckets").update({
            "archived": archived, "updated_at": datetime.now().isoformat(),
        }).eq("user_id", uid).ilike("name", f"%{safe_name}%").execute())
        if response.data:
            return f"{'Archived' if archived else 'Restored'} '{bucket_name}'."
        return f"I couldn't find an initiative matching '{bucket_name}'."
    except Exception as e:
        logger.error(f"Error archiving bucket: {e}", exc_info=True)
        err_s = str(e).lower()
        if "row-level security" in err_s or "42501" in str(e):
            return "Sorry, I couldn't update that initiative due to a permissions issue."
        return "Sorry, I couldn't update that initiative"
