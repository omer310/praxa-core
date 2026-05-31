"""Surface-agnostic tool implementations.

Each `*_impl(ctx, ...)` function is pure logic that returns a display string and
never touches LiveKit / Twilio / push. Surface wrappers (voice function_tools,
the SMS agent, background workers) build a `ToolContext` and call these.
"""
