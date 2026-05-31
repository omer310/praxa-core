import logging

logger = logging.getLogger("praxa_core")
if not logger.handlers:
    logger.setLevel(logging.INFO)


def sanitize_sql_like_pattern(user_input: str) -> str:
    """Sanitize user input for SQL LIKE queries to prevent injection.

    Escapes special LIKE wildcards: % and _
    """
    if not user_input:
        return ""
    return user_input.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def infer_bucket_style(name: str, goal: str | None, description: str | None) -> tuple[str, str]:
    """Pick icon + color for a new initiative (matches the Praxa app IconSymbol / bucket palette)."""
    text = f"{name} {goal or ''} {description or ''}".lower()
    rules: list[tuple[tuple[str, ...], str, str]] = [
        (("work", "job", "career", "office", "business", "client", "startup", "company", "slack"), "briefcase.fill", "#6AACD8"),
        (("health", "fitness", "gym", "workout", "run", "doctor", "medical", "wellness", "sleep", "diet"), "dumbbell.fill", "#88C840"),
        (("money", "finance", "invest", "budget", "bank", "tax", "savings", "debt", "crypto"), "dollarsign.circle.fill", "#F2D060"),
        (("family", "kids", "parent", "spouse", "relationship", "wedding"), "person.2.fill", "#F5BC88"),
        (("home", "house", "renovat", "moving", "lease", "rent"), "house.fill", "#C8DC80"),
        (("learn", "study", "course", "school", "degree", "read", "book"), "graduationcap.fill", "#9E98D8"),
        (("creative", "design", "art", "music", "photo", "film", "write"), "paintbrush.fill", "#F5AACF"),
        (("code", "dev", "software", "app", "engineering", "ship"), "hammer.fill", "#3488C0"),
        (("travel", "trip", "flight", "vacation", "hotel"), "airplane", "#70C890"),
        (("side project", "hobby", "game", "fun"), "gamecontroller.fill", "#BDA8D4"),
        (("personal", "growth", "mind", "habit", "journal"), "leaf.fill", "#3A9490"),
        (("spiritual", "faith", "church", "meditat"), "moon.fill", "#7060C0"),
        (("volunteer", "community", "nonprofit", "charity"), "heart.fill", "#E05656"),
        (("legal", "contract", "law"), "building.columns.fill", "#4030A0"),
        (("car", "vehicle", "commute", "driving"), "car.fill", "#6CB8B4"),
        (("pet", "dog", "cat"), "pawprint.fill", "#F5C4A8"),
        (("food", "cook", "meal", "recipe", "grocery", "restaurant"), "fork.knife", "#E8793A"),
        (("email", "inbox", "message", "chat"), "envelope.fill", "#8DCADE"),
        (("calendar", "plan", "schedule", "event"), "calendar", "#3488C0"),
    ]
    for keywords, icon, color in rules:
        if any(k in text for k in keywords):
            return icon, color
    return "sparkles", "#3488C0"
