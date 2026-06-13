"""Task (loop) tool implementations. Pure logic, no LiveKit.

NOTE: `loops.status` is standardized on "done" for completed tasks across all
surfaces (see P1-2).
"""
import asyncio
from datetime import datetime, timedelta

from .._utils import logger, sanitize_sql_like_pattern
from ..context import ToolContext

STATUS_DONE = "done"


def _not_signed_in() -> str:
    return "I can't access your tasks right now — it looks like you're not signed in. Please open the app and sign in first."


async def _run(fn):
    return await asyncio.to_thread(fn)


async def get_all_tasks_impl(ctx: ToolContext, view_tab: str | None = None) -> str:
    if not ctx.uid:
        return _not_signed_in()
    try:
        sb = ctx.supabase
        uid = ctx.uid
        query = sb.table("loops").select(
            "id, title, status, priority, due_date, view_tab, buckets(name)"
        ).eq("user_id", uid).neq("status", STATUS_DONE)
        if view_tab:
            query = query.eq("view_tab", view_tab)
        response = await _run(lambda: query.eq("archived", False).execute())

        if not response.data:
            if view_tab == "sprint":
                return "No tasks scheduled for this week. Your sprint is clear!"
            if view_tab == "backlog":
                return "No tasks in your backlog."
            return "No active tasks found."

        tasks = []
        for loop in response.data:
            bucket = loop.get("buckets", {}).get("name", "No bucket") if loop.get("buckets") else "No bucket"
            priority = loop.get("priority", "medium")
            status = loop.get("status", "open")
            due = f", due {loop['due_date']}" if loop.get("due_date") else ""
            tasks.append(f"{loop['title']} ({priority} priority, {bucket} bucket, {status}{due})")

        context_str = " for this week (sprint)" if view_tab == "sprint" else (" in your backlog" if view_tab == "backlog" else "")
        return f"Found {len(tasks)} active tasks{context_str}: " + "; ".join(tasks)
    except Exception as e:
        logger.error(f"Error getting tasks: {e}")
        return "Sorry, I couldn't fetch your tasks."


async def get_sprint_tasks_impl(ctx: ToolContext) -> str:
    if not ctx.uid:
        return _not_signed_in()
    try:
        sb, uid = ctx.supabase, ctx.uid
        response = await _run(lambda: sb.table("loops").select(
            "id, title, status, priority, due_date, scheduled_time, buckets(name)"
        ).eq("user_id", uid).eq("view_tab", "sprint").neq("status", STATUS_DONE).eq("archived", False).execute())

        if not response.data:
            return "No tasks scheduled for this week. Your sprint is clear!"

        tasks = []
        for loop in response.data:
            bucket = loop.get("buckets", {}).get("name", "No bucket") if loop.get("buckets") else "No bucket"
            priority = loop.get("priority", "medium")
            status = loop.get("status", "open")
            scheduled = f", scheduled {loop['scheduled_time']}" if loop.get("scheduled_time") else ""
            due = f", due {loop['due_date']}" if loop.get("due_date") else ""
            tasks.append(f"{loop['title']} ({priority} priority, {bucket}, {status}{scheduled}{due})")
        return f"Found {len(tasks)} tasks scheduled for this week: " + "; ".join(tasks)
    except Exception as e:
        logger.error(f"Error getting sprint tasks: {e}")
        return "Sorry, I couldn't fetch your sprint tasks."


async def get_backlog_tasks_impl(ctx: ToolContext) -> str:
    if not ctx.uid:
        return _not_signed_in()
    try:
        sb, uid = ctx.supabase, ctx.uid
        response = await _run(lambda: sb.table("loops").select(
            "id, title, status, priority, due_date, buckets(name)"
        ).eq("user_id", uid).eq("view_tab", "backlog").neq("status", STATUS_DONE).eq("archived", False).execute())

        if not response.data:
            return "No tasks in your backlog. All caught up!"

        tasks = []
        for loop in response.data:
            bucket = loop.get("buckets", {}).get("name", "No bucket") if loop.get("buckets") else "No bucket"
            priority = loop.get("priority", "medium")
            status = loop.get("status", "open")
            due = f", due {loop['due_date']}" if loop.get("due_date") else ""
            tasks.append(f"{loop['title']} ({priority} priority, {bucket}, {status}{due})")
        return f"Found {len(tasks)} tasks in your backlog: " + "; ".join(tasks)
    except Exception as e:
        logger.error(f"Error getting backlog tasks: {e}")
        return "Sorry, I couldn't fetch your backlog tasks."


async def get_tasks_by_bucket_impl(ctx: ToolContext, bucket_name: str) -> str:
    if not ctx.uid:
        return _not_signed_in()
    try:
        sb, uid = ctx.supabase, ctx.uid
        safe_bucket_name = sanitize_sql_like_pattern(bucket_name)
        bucket_response = await _run(lambda: sb.table("buckets").select("id").eq(
            "user_id", uid
        ).ilike("name", safe_bucket_name).eq("archived", False).execute())
        if not bucket_response.data:
            return f"I couldn't find a bucket named '{bucket_name}'"
        bucket_id = bucket_response.data[0]["id"]
        response = await _run(lambda: sb.table("loops").select(
            "title, status, priority, due_date"
        ).eq("user_id", uid).eq("bucket_id", bucket_id).eq("archived", False).execute())
        if not response.data:
            return f"No tasks found in {bucket_name}"
        tasks = []
        for loop in response.data:
            due = f", Due: {loop['due_date']}" if loop.get("due_date") else ""
            tasks.append(f"- {loop['title']} ({loop['status']}, {loop['priority']} priority{due})")
        return f"Tasks in {bucket_name}:\n" + "\n".join(tasks)
    except Exception as e:
        logger.error(f"Error getting tasks by bucket: {e}")
        return f"Sorry, I couldn't fetch tasks for {bucket_name}"


async def get_high_priority_tasks_impl(ctx: ToolContext) -> str:
    if not ctx.uid:
        return _not_signed_in()
    try:
        sb, uid = ctx.supabase, ctx.uid
        response = await _run(lambda: sb.table("loops").select(
            "title, status, due_date, buckets(name)"
        ).eq("user_id", uid).eq("priority", "high").eq("archived", False).execute())
        if not response.data:
            return "No high priority tasks found."
        tasks = []
        for loop in response.data:
            bucket = loop.get("buckets", {}).get("name", "No bucket") if loop.get("buckets") else "No bucket"
            status = loop.get("status", "open")
            due = f", due {loop['due_date']}" if loop.get("due_date") else ""
            tasks.append(f"{loop['title']} ({bucket}, {status}{due})")
        return f"Found {len(tasks)} high priority tasks: " + "; ".join(tasks)
    except Exception as e:
        logger.error(f"Error getting high priority tasks: {e}")
        return "Sorry, I couldn't fetch high priority tasks"


async def get_tasks_by_status_impl(ctx: ToolContext, status: str) -> str:
    if not ctx.uid:
        return _not_signed_in()
    try:
        sb, uid = ctx.supabase, ctx.uid
        response = await _run(lambda: sb.table("loops").select(
            "title, priority, due_date, buckets(name)"
        ).eq("user_id", uid).eq("status", status).eq("archived", False).execute())
        if not response.data:
            return f"No {status} tasks found."
        tasks = []
        for loop in response.data:
            bucket = loop.get("buckets", {}).get("name", "No bucket") if loop.get("buckets") else "No bucket"
            priority = loop.get("priority", "medium")
            due = f", due {loop['due_date']}" if loop.get("due_date") else ""
            tasks.append(f"{loop['title']} ({priority} priority, {bucket}{due})")
        return f"Found {len(tasks)} {status} tasks: " + "; ".join(tasks)
    except Exception as e:
        logger.error(f"Error getting tasks by status: {e}")
        return f"Sorry, I couldn't fetch {status} tasks"


async def get_upcoming_deadlines_impl(ctx: ToolContext, days_ahead: int = 7) -> str:
    if not ctx.uid:
        return _not_signed_in()
    try:
        sb, uid = ctx.supabase, ctx.uid
        end_date = (ctx.now() + timedelta(days=days_ahead)).date().isoformat()
        response = await _run(lambda: sb.table("loops").select(
            "title, due_date, priority, buckets(name)"
        ).eq("user_id", uid).lte("due_date", end_date).eq("archived", False).neq("status", STATUS_DONE).execute())
        if not response.data:
            return f"No upcoming deadlines in the next {days_ahead} days."
        tasks = []
        for loop in response.data:
            bucket = loop.get("buckets", {}).get("name", "No bucket") if loop.get("buckets") else "No bucket"
            priority = loop.get("priority", "medium")
            tasks.append(f"{loop['title']} (due {loop['due_date']}, {priority} priority, {bucket})")
        return f"Found {len(tasks)} upcoming deadlines in the next {days_ahead} days: " + "; ".join(tasks)
    except Exception as e:
        logger.error(f"Error getting upcoming deadlines: {e}")
        return "Sorry, I couldn't fetch upcoming deadlines"


async def search_tasks_impl(ctx: ToolContext, search_term: str) -> str:
    if not ctx.uid:
        return _not_signed_in()
    try:
        sb, uid = ctx.supabase, ctx.uid
        safe_search_term = sanitize_sql_like_pattern(search_term)
        response = await _run(lambda: sb.table("loops").select(
            "title, status, priority, buckets(name)"
        ).eq("user_id", uid).ilike("title", f"%{safe_search_term}%").eq("archived", False).execute())
        if not response.data:
            return f"No tasks found matching '{search_term}'."
        tasks = []
        for loop in response.data:
            bucket = loop.get("buckets", {}).get("name", "No bucket") if loop.get("buckets") else "No bucket"
            status = loop.get("status", "open")
            priority = loop.get("priority", "medium")
            tasks.append(f"{loop['title']} ({status}, {priority} priority, {bucket})")
        return f"Found {len(tasks)} tasks matching '{search_term}': " + "; ".join(tasks)
    except Exception as e:
        logger.error(f"Error searching tasks: {e}")
        return "Sorry, I couldn't search for tasks"


async def create_task_impl(
    ctx: ToolContext,
    title: str,
    bucket_name: str,
    priority: str = "medium",
    due_date: str | None = None,
) -> str:
    if not ctx.uid:
        return _not_signed_in()
    sb, uid = ctx.supabase, ctx.uid
    if not sb:
        return "Sorry, I couldn't create the task — the assistant isn't fully connected to your data yet."
    try:
        all_buckets_response = await _run(lambda: sb.table("buckets").select("id, name").eq(
            "user_id", uid
        ).eq("archived", False).execute())
        if not all_buckets_response.data:
            return "You don't have any buckets yet. Please create a bucket first before adding tasks."

        available_buckets = {b["name"].lower(): b for b in all_buckets_response.data}
        bucket_data = available_buckets.get(bucket_name.lower())
        if not bucket_data:
            for bucket_key, bucket_val in available_buckets.items():
                if bucket_name.lower() in bucket_key or bucket_key in bucket_name.lower():
                    bucket_data = bucket_val
                    break
        if not bucket_data:
            bucket_names = ", ".join([b["name"] for b in all_buckets_response.data])
            return f"I couldn't find a bucket named '{bucket_name}'. Your available buckets are: {bucket_names}. Which one should I use?"

        new_loop = {
            "user_id": uid,
            "title": title,
            "status": "open",
            "priority": priority,
            "bucket_id": bucket_data["id"],
        }
        if due_date:
            new_loop["due_date"] = due_date

        response = await _run(lambda: sb.table("loops").insert(new_loop).execute())
        if not response.data:
            return "Sorry, I couldn't create the task."
        actual_bucket_name = bucket_data["name"]
        if due_date:
            return f"Got it, added '{title}' to {actual_bucket_name}, due {due_date}"
        return f"Done, added '{title}' to {actual_bucket_name}"
    except Exception as e:
        logger.error(f"Error creating task: {e}", exc_info=True)
        err_s = str(e).lower()
        if "row-level security" in err_s or "42501" in str(e):
            return "Sorry, I couldn't save that task due to a permissions issue."
        return "Sorry, I couldn't create the task"


async def complete_task_impl(ctx: ToolContext, task_title: str) -> str:
    if not ctx.uid:
        return _not_signed_in()
    try:
        sb, uid = ctx.supabase, ctx.uid
        safe_task_title = sanitize_sql_like_pattern(task_title)
        response = await _run(lambda: sb.table("loops").update(
            {"status": STATUS_DONE, "completed_at": datetime.now().isoformat()}
        ).eq("user_id", uid).ilike("title", f"%{safe_task_title}%").execute())
        if response.data:
            return f"Done! Marked '{task_title}' as complete"
        return f"I couldn't find a task matching '{task_title}'"
    except Exception as e:
        logger.error(f"Error completing task: {e}")
        return "Sorry, I couldn't complete the task"


async def update_task_priority_impl(ctx: ToolContext, task_title: str, new_priority: str) -> str:
    if not ctx.uid:
        return _not_signed_in()
    try:
        sb, uid = ctx.supabase, ctx.uid
        safe_task_title = sanitize_sql_like_pattern(task_title)
        response = await _run(lambda: sb.table("loops").update(
            {"priority": new_priority}
        ).eq("user_id", uid).ilike("title", f"%{safe_task_title}%").execute())
        if response.data:
            return f"Updated '{task_title}' priority to {new_priority}"
        return f"Couldn't find task '{task_title}'"
    except Exception as e:
        logger.error(f"Error updating task priority: {e}")
        return "Sorry, I couldn't update the task priority"


async def reschedule_task_impl(ctx: ToolContext, task_title: str, new_due_date: str) -> str:
    if not ctx.uid:
        return _not_signed_in()
    try:
        sb, uid = ctx.supabase, ctx.uid
        due_date_str = new_due_date
        if ":" in new_due_date or "pm" in new_due_date.lower() or "am" in new_due_date.lower():
            due_date_str = datetime.now().date().isoformat()
        safe_task_title = sanitize_sql_like_pattern(task_title)
        response = await _run(lambda: sb.table("loops").update(
            {"due_date": due_date_str}
        ).eq("user_id", uid).ilike("title", f"%{safe_task_title}%").execute())
        if response.data:
            return f"Got it, rescheduled '{task_title}' to {new_due_date}"
        return f"I couldn't find a task called '{task_title}'"
    except Exception as e:
        logger.error(f"Error rescheduling task: {e}")
        return "Sorry, I couldn't reschedule the task"


async def add_task_note_impl(ctx: ToolContext, task_title: str, note: str) -> str:
    if not ctx.uid:
        return _not_signed_in()
    try:
        sb, uid = ctx.supabase, ctx.uid
        safe_task_title = sanitize_sql_like_pattern(task_title)
        existing = await _run(lambda: sb.table("loops").select("id, notes").eq(
            "user_id", uid
        ).ilike("title", f"%{safe_task_title}%").limit(1).execute())
        if not existing.data:
            return f"I couldn't find a task matching '{task_title}'"
        task_id = existing.data[0]["id"]
        existing_notes = existing.data[0].get("notes", "") or ""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        new_notes = f"{existing_notes}\n\n[{timestamp}] {note}".strip()
        await _run(lambda: sb.table("loops").update({
            "notes": new_notes, "updated_at": datetime.now().isoformat()
        }).eq("id", task_id).execute())
        return f"Got it, noted that down on '{task_title}'."
    except Exception as e:
        logger.error(f"Error adding task note: {e}")
        return "Sorry, I couldn't add that note"


async def update_loop_impl(
    ctx: ToolContext,
    task_title: str,
    status: str | None = None,
    description: str | None = None,
    view_tab: str | None = None,
    estimated_duration_minutes: int | None = None,
) -> str:
    if not ctx.uid:
        return _not_signed_in()
    try:
        sb, uid = ctx.supabase, ctx.uid
        safe_task_title = sanitize_sql_like_pattern(task_title)
        updates: dict = {}
        changes: list[str] = []
        if status and status in ("open", "in_progress", STATUS_DONE):
            updates["status"] = status
            changes.append(f"status → {status}")
        if description is not None:
            updates["description"] = description
            changes.append("description updated")
        if view_tab is not None and view_tab in ("sprint", "backlog"):
            updates["view_tab"] = view_tab
            changes.append("moved to sprint (this week)" if view_tab == "sprint" else "moved to backlog")
        if estimated_duration_minutes is not None and estimated_duration_minutes > 0:
            updates["estimated_duration_minutes"] = estimated_duration_minutes
            changes.append(f"estimated time → {estimated_duration_minutes} min")
        if not updates:
            return "Nothing to update — tell me what you'd like to change."
        updates["updated_at"] = datetime.now().isoformat()
        response = await _run(lambda: sb.table("loops").update(updates).eq(
            "user_id", uid
        ).ilike("title", f"%{safe_task_title}%").execute())
        if response.data:
            return f"Updated '{task_title}': {', '.join(changes)}."
        return f"I couldn't find a task matching '{task_title}'"
    except Exception as e:
        logger.error(f"Error updating loop: {e}", exc_info=True)
        return "Sorry, I couldn't update that task"


async def rename_task_impl(ctx: ToolContext, current_title: str, new_title: str) -> str:
    if not ctx.uid:
        return _not_signed_in()
    sb, uid = ctx.supabase, ctx.uid
    if not sb:
        return "Sorry, I couldn't rename the task right now."
    try:
        safe_old = sanitize_sql_like_pattern(current_title)
        response = await _run(lambda: sb.table("loops").update({
            "title": new_title, "updated_at": datetime.now().isoformat(),
        }).eq("user_id", uid).ilike("title", f"%{safe_old}%").execute())
        if response.data:
            return f"Renamed that to '{new_title}'."
        return f"I couldn't find a task matching '{current_title}'."
    except Exception as e:
        logger.error(f"Error renaming task: {e}", exc_info=True)
        err_s = str(e).lower()
        if "row-level security" in err_s or "42501" in str(e):
            return "Sorry, I couldn't save that rename due to a permissions issue."
        return "Sorry, I couldn't rename the task"


async def schedule_loop_impl(ctx: ToolContext, task_title: str, scheduled_time: str) -> str:
    if not ctx.uid:
        return _not_signed_in()
    try:
        sb, uid = ctx.supabase, ctx.uid
        safe_task_title = sanitize_sql_like_pattern(task_title)
        try:
            parsed = datetime.fromisoformat(scheduled_time.replace("Z", "+00:00"))
            display = parsed.strftime("%A, %B %d at %I:%M %p")
        except ValueError:
            display = scheduled_time
        response = await _run(lambda: sb.table("loops").update({
            "scheduled_time": scheduled_time, "view_tab": "sprint", "updated_at": datetime.now().isoformat()
        }).eq("user_id", uid).ilike("title", f"%{safe_task_title}%").execute())
        if response.data:
            return f"Got it! '{task_title}' is scheduled for {display} and added to this week's focus."
        return f"I couldn't find a task matching '{task_title}'"
    except Exception as e:
        logger.error(f"Error scheduling loop: {e}")
        return "Sorry, I couldn't schedule that task"
