"""Unified action pipeline: risk manifest + the `integration_actions` queue.

Every state-changing write that should run out-of-band (or that needs user
confirmation) is queued here. The dispatcher (in praxa-backend) is the single
execution chokepoint and refuses to run a `confirm` action until it has been
confirmed. All surfaces classify risk identically by reading this manifest.
"""
import asyncio
from dataclasses import dataclass
from typing import Optional

from ._utils import logger
from .context import ToolContext

# Risk tiers
AUTO = "auto"        # reversible / low-risk -> execute immediately
CONFIRM = "confirm"  # external / hard-to-undo -> requires explicit user confirmation

# Action queue statuses
STATUS_QUEUED = "queued"
STATUS_PENDING_CONFIRMATION = "pending_confirmation"
STATUS_CONFIRMED = "confirmed"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"


@dataclass(frozen=True)
class ActionSpec:
    action_type: str
    provider: str
    risk: str
    executor: str          # "n8n" | "native"
    review_route: str      # deep-link target for the review UI
    label: str             # human label used when building summaries


# action_type -> spec. The review_route is where the mobile app sends the user
# to review/edit/approve the action (P1-5). Email reuses the existing compose
# screen; everything else uses the generic action-review screen.
ACTION_MANIFEST: dict[str, ActionSpec] = {
    # --- Notion (external writes -> confirm, executed via n8n) ---
    "create_notion_page": ActionSpec(
        "create_notion_page", "notion", CONFIRM, "n8n", "/action-review", "Create Notion page"
    ),
    "update_notion_page": ActionSpec(
        "update_notion_page", "notion", CONFIRM, "n8n", "/action-review", "Update Notion page"
    ),
    # --- Slack (external writes -> confirm, executed via n8n) ---
    "send_slack_message": ActionSpec(
        "send_slack_message", "slack", CONFIRM, "n8n", "/action-review", "Send Slack message"
    ),
    # --- Email (external writes -> confirm, executed natively via Nylas) ---
    "reply_to_email": ActionSpec(
        "reply_to_email", "email", CONFIRM, "native", "/email-mode", "Reply to email"
    ),
    "send_email": ActionSpec(
        "send_email", "email", CONFIRM, "native", "/email-mode", "Send email"
    ),
    # --- Calendar (changes a shared resource -> confirm, native via Nylas) ---
    "reschedule_calendar_event": ActionSpec(
        "reschedule_calendar_event", "calendar", CONFIRM, "native", "/calendar-mode", "Reschedule event"
    ),
    # --- Tasks / buckets (own data, reversible -> auto, native) ---
    "create_task": ActionSpec("create_task", "tasks", AUTO, "native", "/(tabs)/initiatives", "Create task"),
    "complete_task": ActionSpec("complete_task", "tasks", AUTO, "native", "/(tabs)/initiatives", "Complete task"),
    "update_task": ActionSpec("update_task", "tasks", AUTO, "native", "/(tabs)/initiatives", "Update task"),
    "create_bucket": ActionSpec("create_bucket", "buckets", AUTO, "native", "/(tabs)/initiatives", "Create initiative"),
    "update_bucket": ActionSpec("update_bucket", "buckets", AUTO, "native", "/(tabs)/initiatives", "Update initiative"),
}


def get_action_spec(action_type: str) -> Optional[ActionSpec]:
    return ACTION_MANIFEST.get(action_type)


def classify_action(action_type: str) -> str:
    """Return the risk tier for an action_type. Unknown actions default to CONFIRM (safe)."""
    spec = ACTION_MANIFEST.get(action_type)
    return spec.risk if spec else CONFIRM


def requires_confirmation(action_type: str) -> bool:
    return classify_action(action_type) == CONFIRM


def _derive_match_key(action_type: str, payload: dict) -> Optional[str]:
    """A stable key a user can graduate to AUTO (e.g. a trusted sender domain or
    Slack channel). Returns None when the action has no natural narrow key."""
    if action_type in ("reply_to_email", "send_email"):
        addr = (payload.get("recipient") or payload.get("to") or "").strip().lower()
        if "@" in addr:
            return addr.split("@", 1)[1]
        return None
    if action_type == "send_slack_message":
        return (payload.get("channel") or "").strip().lower() or None
    return None


async def _autonomy_allows_auto(sb, uid: str, action_type: str, payload: dict) -> bool:
    """Opt-in: a user can pre-authorize specific (action_type, match_key) pairs
    (or all of an action_type via a null match_key) to skip confirmation. With no
    rules configured this always returns False, preserving today's confirm-first
    behavior."""
    match_key = _derive_match_key(action_type, payload)
    keys = [None] if match_key is None else [match_key, None]  # exact then wildcard
    try:
        resp = await asyncio.to_thread(
            lambda: sb.table("user_autonomy_rules")
            .select("match_key, mode")
            .eq("user_id", uid)
            .eq("action_type", action_type)
            .execute()
        )
        rows = resp.data or []
    except Exception as e:
        logger.warning("[actions] autonomy rule lookup failed: %s", e)
        return False

    by_key: dict = {}
    for r in rows:
        by_key[r.get("match_key")] = r.get("mode")
    for k in keys:
        mode = by_key.get(k)
        if mode == "auto":
            return True
        if mode == "confirm":
            return False  # an explicit confirm rule wins over a broader auto
    return False


async def queue_action(
    ctx: ToolContext,
    provider: str,
    action_type: str,
    payload: dict,
    *,
    summary: Optional[str] = None,
    review_route: Optional[str] = None,
    force_confirmation: Optional[bool] = None,
) -> str:
    """Insert a row into `integration_actions` and return its id.

    Status is derived from the risk manifest: `confirm` actions land as
    `pending_confirmation`, `auto` actions land as `queued` and are picked up by
    the dispatcher immediately. `summary` is the descriptive one-liner used by
    push/SMS; `review_route` is the deep-link target for the in-app review UI.
    """
    sb = ctx.supabase
    uid = ctx.uid
    if not sb or not uid:
        raise RuntimeError("Supabase or user_id not available")

    spec = ACTION_MANIFEST.get(action_type)
    needs_confirm = force_confirmation if force_confirmation is not None else (
        spec.risk == CONFIRM if spec else True
    )
    # Learned autonomy (P5): if the user has opted a specific (action_type,
    # match_key) into AUTO, a normally-CONFIRM action can skip the gate. Explicit
    # force_confirmation always wins. Default (no rules) leaves behavior unchanged.
    if needs_confirm and force_confirmation is None:
        if await _autonomy_allows_auto(sb, uid, action_type, payload):
            needs_confirm = False
            logger.info("[actions] autonomy rule graduated %s to AUTO", action_type)
    executor = spec.executor if spec else ("n8n" if provider in ("notion", "slack") else "native")
    route = review_route or (spec.review_route if spec else "/action-review")
    status = STATUS_PENDING_CONFIRMATION if needs_confirm else STATUS_QUEUED

    row = {
        "user_id": uid,
        "provider": provider,
        "action_type": action_type,
        "payload": payload,
        "status": status,
        "surface": ctx.surface,
        "executor": executor,
        "requires_confirmation": needs_confirm,
        "review_route": route,
    }
    if summary:
        row["summary"] = summary

    result = await asyncio.to_thread(
        lambda: sb.table("integration_actions").insert(row).execute()
    )
    rows = result.data if result else []
    if rows:
        action_id = rows[0]["id"]
        logger.info(
            "[actions] queued %s (%s) status=%s id=%s surface=%s",
            action_type, provider, status, action_id, ctx.surface,
        )
        return action_id
    raise RuntimeError("Failed to insert integration action")
