# praxa_core

Shared, surface-agnostic core for every Praxa agent (voice, SMS, in-app, background workers).

It contains:

- `context.ToolContext` - an explicit dependency container (user id, Supabase client, Nylas grants, timezone, surface) that replaces the LiveKit-coupled `ContextVar` used by the voice agent.
- `tools/*` - pure `*_impl(ctx, ...)` functions for tasks, buckets, calendar, email and integrations. They return display strings and never touch LiveKit.
- `actions.py` - the action risk manifest (`auto` vs `confirm`) and the unified `integration_actions` queue helpers.
- `email_filter.py` - the multi-tier inbox triage engine.
- `memory.py` - unified context retrieval across facts, session summaries and integration context.

Each surface builds a `ToolContext` and calls the `_impl` functions. Surface-specific concerns (LiveKit "thinking" indicators, the room data channel, Twilio replies) stay in the surface wrappers.

## Install (editable, local dev)

```bash
pip install -e ../praxa_core
```

Both `Praxa-voice-agent` and `praxa-backend` depend on this package. For Railway, add a git/path requirement that resolves to this package.
