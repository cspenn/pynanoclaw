# start src/nanoclaw/router.py
"""Message formatting and outbound routing for NanoClaw.

Replaces src/router.ts from the TypeScript implementation.
Handles XML escaping, message formatting, and channel selection.
"""

from __future__ import annotations

import re

from nanoclaw.types import NewMessage


def escape_xml(text: str) -> str:
    """Escape special XML characters in a string.

    Args:
        text: Raw string to escape.

    Returns:
        String with &, <, >, " replaced by XML entities.
    """
    if not text:
        return ""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def format_messages(messages: list[NewMessage]) -> str:
    """Format a list of messages into XML for agent consumption.

    Args:
        messages: List of inbound messages to format.

    Returns:
        XML string with all messages wrapped in <messages> root element.
    """
    lines = [
        f'<message sender="{escape_xml(m.sender_name)}" time="{m.timestamp}">'
        f"{escape_xml(m.content)}</message>"
        for m in messages
    ]
    return f"<messages>\n{chr(10).join(lines)}\n</messages>"


_INTERNAL_TAG_PATTERN: re.Pattern[str] = re.compile(r"<internal>[\s\S]*?</internal>", re.DOTALL)


def strip_internal_tags(text: str) -> str:
    """Remove <internal>...</internal> blocks from agent output.

    These blocks contain the agent's internal reasoning and should not
    be sent to the user.

    Args:
        text: Raw agent output text.

    Returns:
        Text with all internal blocks removed and whitespace trimmed.
    """
    return _INTERNAL_TAG_PATTERN.sub("", text).strip()


def format_outbound(raw_text: str) -> str:
    """Prepare agent output text for sending to the user.

    Strips internal reasoning blocks and trims whitespace.

    Args:
        raw_text: Raw text from the agent container.

    Returns:
        Clean text ready for the user, or empty string if nothing remains.
    """
    return strip_internal_tags(raw_text)


# end src/nanoclaw/router.py
