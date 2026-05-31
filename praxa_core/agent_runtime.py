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
    {"type": "function", "function": {
        "name": "get_tasks",
        "description": "List the user's active tasks. view: 'sprint' (this week), 'backlog', or omit for all.",
        "parameters": {"type": "object", "properties": {"view": {"type": "string", "enum": ["sprint", "backlog", "all"]}}},
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
    {"type": "function", "function": {
        "name": "get_todays_calendar",
        "description": "Get the user's calendar events for today.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_emails_needing_response",
        "description": "List unread emails from real people that may need a reply.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "search_notion",
        "description": "Search the user's connected Notion workspace for pages matching a query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
]


async def execute_core_tool(ctx: ToolContext, name: str, args: dict) -> Optional[str]:
    """Run a core tool. Returns None if `name` is not a core tool (so a surface
    executor can handle its own extra tools)."""
    from .tools import tasks as _tasks, calendar as _calendar, email as _email, integrations as _integrations

    if name == "get_tasks":
        view = args.get("view", "all")
        if view == "sprint":
            return await _tasks.get_sprint_tasks_impl(ctx)
        if view == "backlog":
            return await _tasks.get_backlog_tasks_impl(ctx)
        return await _tasks.get_all_tasks_impl(ctx)
    if name == "create_task":
        return await _tasks.create_task_impl(
            ctx, args.get("title", ""), args.get("bucket_name", ""),
            args.get("priority", "medium"), args.get("due_date"),
        )
    if name == "complete_task":
        return await _tasks.complete_task_impl(ctx, args.get("task_title", ""))
    if name == "get_todays_calendar":
        return await _calendar.get_todays_events_impl(ctx)
    if name == "get_emails_needing_response":
        return await _email.get_emails_needing_response_impl(ctx)
    if name == "search_notion":
        return await _integrations.search_notion_impl(ctx, args.get("query", ""))
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
