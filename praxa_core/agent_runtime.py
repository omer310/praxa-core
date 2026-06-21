"""Surface-agnostic agent runtime.

A single OpenAI function-calling loop and a shared core tool set used by every
text surface (SMS today; Slack and background next). Each surface supplies its
own system prompt and may layer extra tools (e.g. SMS adds pending-action
confirmation), but the conversation loop and the common tools live here so
behavior is identical everywhere.

Reads run directly; risky writes are queued by the underlying *_impl functions
as pending_confirmation, so the runtime never sends anything externally on its
own.
"""
import json
import os
from typing import Awaitable, Callable, Optional

from ._utils import logger
from .context import ToolContext

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

DEFAULT_MODEL = os.getenv("AGENT_RUNTIME_MODEL", "gpt-4o-mini")
DEFAULT_MAX_ROUNDS = 5

# Tool name -> executor. Each surface composes its own executor on top of this.
ToolExecutor = Callable[[ToolContext, str, dict], Awaitable[str]]


# ---------------------------------------------------------------------------
# Core tool schemas (surface-agnostic)
# ---------------------------------------------------------------------------

CORE_TOOLS = [
    # --- Tasks ---
    {"type": "function", "function": {
        "name": "get_tasks",
        "description": "List the user's active tasks. view: 'sprint' (this week), 'backlog', or omit for all.",
        "parameters": {"type": "object", "properties": {"view": {"type": "string", "enum": ["sprint", "backlog", "all"]}}},
    }},
    {"type": "function", "function": {
        "name": "get_sprint_tasks",
        "description": "Get all tasks scheduled for this week (sprint view).",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_backlog_tasks",
        "description": "Get all tasks in the backlog — not scheduled for this week.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_high_priority_tasks",
        "description": "Get all high-priority tasks regardless of view tab.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "search_tasks",
        "description": "Search tasks by title keyword.",
        "parameters": {"type": "object", "properties": {"search_term": {"type": "string"}}, "required": ["search_term"]},
    }},
    {"type": "function", "function": {
        "name": "get_upcoming_deadlines",
        "description": "Get tasks with due dates within the next N days (default 7).",
        "parameters": {"type": "object", "properties": {"days_ahead": {"type": "integer", "description": "Number of days to look ahead (default 7)"}}},
    }},
    {"type": "function", "function": {
        "name": "create_task",
        "description": "Create a task in a bucket/initiative. Infer the bucket from context.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"}, "bucket_name": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
            "due_date": {"type": "string", "description": "YYYY-MM-DD"},
        }, "required": ["title", "bucket_name"]},
    }},
    {"type": "function", "function": {
        "name": "complete_task",
        "description": "Mark a task complete by (partial) title.",
        "parameters": {"type": "object", "properties": {"task_title": {"type": "string"}}, "required": ["task_title"]},
    }},
    # --- Subtasks ---
    {"type": "function", "function": {
        "name": "list_subtasks",
        "description": "List the subtasks (steps) of a task by (partial) title.",
        "parameters": {"type": "object", "properties": {"task_title": {"type": "string"}}, "required": ["task_title"]},
    }},
    {"type": "function", "function": {
        "name": "break_down_task",
        "description": "Break a task into subtasks/steps. ONLY call this after the user has agreed to a breakdown you proposed in conversation. Do not restructure tasks silently.",
        "parameters": {"type": "object", "properties": {
            "task_title": {"type": "string", "description": "The task to break down (partial title)"},
            "subtasks": {"type": "array", "items": {"type": "string"}, "description": "The ordered list of step titles the user agreed to"},
        }, "required": ["task_title", "subtasks"]},
    }},
    {"type": "function", "function": {
        "name": "add_subtask",
        "description": "Add a single subtask (step) to an existing task.",
        "parameters": {"type": "object", "properties": {
            "task_title": {"type": "string"}, "subtask_title": {"type": "string"},
        }, "required": ["task_title", "subtask_title"]},
    }},
    {"type": "function", "function": {
        "name": "complete_subtask",
        "description": "Mark a subtask (step) as done by partial title within a task.",
        "parameters": {"type": "object", "properties": {
            "task_title": {"type": "string"}, "subtask_title": {"type": "string"},
        }, "required": ["task_title", "subtask_title"]},
    }},
    # --- Calendar ---
    {"type": "function", "function": {
        "name": "get_todays_calendar",
        "description": "Get the user's calendar events for today.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_upcoming_events",
        "description": "Get upcoming calendar events for the next N days (default 14).",
        "parameters": {"type": "object", "properties": {"days_ahead": {"type": "integer", "description": "Days to look ahead (default 14)"}}},
    }},
    {"type": "function", "function": {
        "name": "get_events_for_day",
        "description": "Get calendar events for a specific day. Accepts natural language like 'tomorrow', 'Monday', or 'April 19'.",
        "parameters": {"type": "object", "properties": {"day": {"type": "string"}}, "required": ["day"]},
    }},
    {"type": "function", "function": {
        "name": "check_free_time",
        "description": "Check free time slots on a specific date (8am–8pm working hours).",
        "parameters": {"type": "object", "properties": {"date": {"type": "string", "description": "YYYY-MM-DD or natural language date"}}, "required": ["date"]},
    }},
    {"type": "function", "function": {
        "name": "reschedule_calendar_event",
        "description": "Reschedule a calendar event to a new date/time. Queued for user approval.",
        "parameters": {"type": "object", "properties": {
            "event_name": {"type": "string"}, "new_date_time": {"type": "string", "description": "ISO datetime or natural language"},
        }, "required": ["event_name", "new_date_time"]},
    }},
    # --- Email ---
    {"type": "function", "function": {
        "name": "get_emails_needing_response",
        "description": "List unread emails from real people that may need a reply.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_email_body",
        "description": "Read the full body of a specific email by number from the last listing, sender name, subject keyword, or raw ID.",
        "parameters": {"type": "object", "properties": {"email_ref": {"type": "string"}}, "required": ["email_ref"]},
    }},
    {"type": "function", "function": {
        "name": "search_emails",
        "description": "Search emails by sender name, email address, or subject keyword.",
        "parameters": {"type": "object", "properties": {"search_term": {"type": "string"}}, "required": ["search_term"]},
    }},
    {"type": "function", "function": {
        "name": "reply_to_email",
        "description": "Queue a reply to an email for user approval. Use after reading the email body.",
        "parameters": {"type": "object", "properties": {
            "email_id": {"type": "string"}, "reply_body": {"type": "string"},
        }, "required": ["email_id", "reply_body"]},
    }},
    # --- Integrations ---
    {"type": "function", "function": {
        "name": "search_notion",
        "description": "Search the user's connected Notion workspace for pages matching a query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "get_slack_unread",
        "description": "Get recent unread Slack messages from the user's connected workspace.",
        "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "description": "Max messages to return (default 15)"}}},
    }},
    {"type": "function", "function": {
        "name": "send_slack_message",
        "description": "Queue a Slack message to a channel or DM for user approval.",
        "parameters": {"type": "object", "properties": {
            "channel": {"type": "string", "description": "Channel name (e.g. 'general') or user ID"},
            "message": {"type": "string"},
        }, "required": ["channel", "message"]},
    }},
    {"type": "function", "function": {
        "name": "create_notion_page",
        "description": "Queue creation of a new Notion page for user approval.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"}, "content": {"type": "string"},
            "parent_page_id": {"type": "string", "description": "Optional parent page ID"},
        }, "required": ["title", "content"]},
    }},
    {"type": "function", "function": {
        "name": "update_notion_page",
        "description": "Queue an update to an existing Notion page for user approval.",
        "parameters": {"type": "object", "properties": {
            "page_id": {"type": "string"}, "new_title": {"type": "string"}, "append_text": {"type": "string"},
        }, "required": ["page_id"]},
    }},
]


async def execute_core_tool(ctx: ToolContext, name: str, args: dict) -> Optional[str]:
    """Run a core tool. Returns None if `name` is not a core tool (so a surface
    executor can handle its own extra tools)."""
    from .tools import tasks as _tasks, calendar as _calendar, email as _email, integrations as _integrations

    # --- Tasks ---
    if name == "get_tasks":
        view = args.get("view", "all")
        if view == "sprint":
            return await _tasks.get_sprint_tasks_impl(ctx)
        if view == "backlog":
            return await _tasks.get_backlog_tasks_impl(ctx)
        return await _tasks.get_all_tasks_impl(ctx)
    if name == "get_sprint_tasks":
        return await _tasks.get_sprint_tasks_impl(ctx)
    if name == "get_backlog_tasks":
        return await _tasks.get_backlog_tasks_impl(ctx)
    if name == "get_high_priority_tasks":
        return await _tasks.get_high_priority_tasks_impl(ctx)
    if name == "search_tasks":
        return await _tasks.search_tasks_impl(ctx, args.get("search_term", ""))
    if name == "get_upcoming_deadlines":
        return await _tasks.get_upcoming_deadlines_impl(ctx, args.get("days_ahead", 7))
    if name == "create_task":
        return await _tasks.create_task_impl(
            ctx, args.get("title", ""), args.get("bucket_name", ""),
            args.get("priority", "medium"), args.get("due_date"),
        )
    if name == "complete_task":
        return await _tasks.complete_task_impl(ctx, args.get("task_title", ""))
    # --- Subtasks ---
    if name == "list_subtasks":
        return await _tasks.list_subtasks_impl(ctx, args.get("task_title", ""))
    if name == "break_down_task":
        return await _tasks.break_down_task_impl(ctx, args.get("task_title", ""), args.get("subtasks", []))
    if name == "add_subtask":
        return await _tasks.add_subtask_impl(ctx, args.get("task_title", ""), args.get("subtask_title", ""))
    if name == "complete_subtask":
        return await _tasks.complete_subtask_impl(ctx, args.get("task_title", ""), args.get("subtask_title", ""))
    # --- Calendar ---
    if name == "get_todays_calendar":
        return await _calendar.get_todays_events_impl(ctx)
    if name == "get_upcoming_events":
        return await _calendar.get_upcoming_events_impl(ctx, args.get("days_ahead", 14))
    if name == "get_events_for_day":
        return await _calendar.get_events_for_day_impl(ctx, args.get("day", "today"))
    if name == "check_free_time":
        return await _calendar.check_free_time_impl(ctx, args.get("date", ""))
    if name == "reschedule_calendar_event":
        return await _calendar.reschedule_calendar_event_impl(ctx, args.get("event_name", ""), args.get("new_date_time", ""))
    # --- Email ---
    if name == "get_emails_needing_response":
        return await _email.get_emails_needing_response_impl(ctx)
    if name == "get_email_body":
        return await _email.get_email_body_impl(ctx, args.get("email_ref", ""))
    if name == "search_emails":
        return await _email.search_emails_impl(ctx, args.get("search_term", ""))
    if name == "reply_to_email":
        return await _email.reply_to_email_impl(ctx, args.get("email_id", ""), args.get("reply_body", ""))
    # --- Integrations ---
    if name == "search_notion":
        return await _integrations.search_notion_impl(ctx, args.get("query", ""))
    if name == "get_slack_unread":
        return await _integrations.get_slack_unread_impl(ctx, args.get("limit", 15))
    if name == "send_slack_message":
        return await _integrations.send_slack_message_impl(ctx, args.get("channel", ""), args.get("message", ""))
    if name == "create_notion_page":
        return await _integrations.create_notion_page_impl(ctx, args.get("title", ""), args.get("content", ""), args.get("parent_page_id", ""))
    if name == "update_notion_page":
        return await _integrations.update_notion_page_impl(ctx, args.get("page_id", ""), args.get("new_title", ""), args.get("append_text", ""))
    return None


# ---------------------------------------------------------------------------
# Conversation loop
# ---------------------------------------------------------------------------

async def run_agent(
    ctx: ToolContext,
    message: str,
    *,
    system_prompt: str,
    tools: Optional[list] = None,
    execute_tool: Optional[ToolExecutor] = None,
    history: Optional[list[dict]] = None,
    model: Optional[str] = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    max_tokens: int = 400,
    temperature: float = 0.3,
    on_round_end: Optional[Callable[[], Awaitable[None]]] = None,
) -> str:
    """Run one inbound message through the tool-calling loop and return the reply.

    `execute_tool` should handle any surface-specific tools and delegate the rest
    to `execute_core_tool`. If omitted, only core tools are available.
    """
    if AsyncOpenAI is None:
        raise RuntimeError("openai package not available")

    tools = tools if tools is not None else CORE_TOOLS

    async def _default_exec(c: ToolContext, name: str, args: dict) -> str:
        result = await execute_core_tool(c, name, args)
        return result if result is not None else f"Unknown tool: {name}"

    executor = execute_tool or _default_exec

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": message})

    for _ in range(max_rounds):
        resp = await client.chat.completions.create(
            model=model or DEFAULT_MODEL,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0].message
        tool_calls = choice.tool_calls or []
        if not tool_calls:
            return (choice.content or "Got it.").strip()

        messages.append({
            "role": "assistant",
            "content": choice.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            try:
                result = await executor(ctx, tc.function.name, args)
            except Exception as e:
                logger.error(f"[agent_runtime] tool {tc.function.name} failed: {e}")
                result = f"That tool hit an error: {e}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        if on_round_end:
            await on_round_end()

    return "Got it. I've noted that — open the app for more detail."
