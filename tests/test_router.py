# start tests/test_router.py
"""Tests for nanoclaw.router module.

Replaces src/routing.test.ts and src/formatting.test.ts from TypeScript.
"""

from __future__ import annotations

import pytest

from nanoclaw.router import escape_xml, format_messages, format_outbound, strip_internal_tags
from nanoclaw.types import NewMessage


def make_message(
    content: str,
    sender_name: str = "Alice",
    timestamp: str = "2024-01-01T00:00:00.000Z",
) -> NewMessage:
    """Create a test NewMessage with sensible defaults."""
    return NewMessage(
        id="msg-1",
        chat_jid="test@g.us",
        sender="alice@s.whatsapp.net",
        sender_name=sender_name,
        content=content,
        timestamp=timestamp,
    )


class TestEscapeXml:
    """Tests for escape_xml function."""

    def test_escapes_ampersand(self) -> None:
        """Ampersands must be escaped to prevent malformed XML."""
        assert escape_xml("Tom & Jerry") == "Tom &amp; Jerry"

    def test_escapes_less_than(self) -> None:
        """Less-than must be escaped."""
        assert escape_xml("a < b") == "a &lt; b"

    def test_escapes_greater_than(self) -> None:
        """Greater-than must be escaped."""
        assert escape_xml("a > b") == "a &gt; b"

    def test_escapes_double_quote(self) -> None:
        """Double quotes must be escaped (used in XML attributes)."""
        assert escape_xml('say "hello"') == "say &quot;hello&quot;"

    def test_escapes_all_special_chars(self) -> None:
        """All special characters escaped in one string."""
        assert escape_xml('<a href="x&y">') == "&lt;a href=&quot;x&amp;y&quot;&gt;"

    def test_empty_string_returns_empty(self) -> None:
        """Empty input returns empty output."""
        assert escape_xml("") == ""

    def test_plain_text_unchanged(self) -> None:
        """Plain text without special chars is returned unchanged."""
        assert escape_xml("hello world") == "hello world"


class TestFormatMessages:
    """Tests for format_messages function."""

    def test_single_message_format(self) -> None:
        """Single message wrapped in correct XML structure."""
        msg = make_message("Hello there", sender_name="Bob", timestamp="2024-01-01T12:00:00.000Z")
        result = format_messages([msg])
        assert result.startswith("<messages>")
        assert result.endswith("</messages>")
        assert 'sender="Bob"' in result
        assert 'time="2024-01-01T12:00:00.000Z"' in result
        assert "Hello there" in result

    def test_multiple_messages(self) -> None:
        """Multiple messages are all included."""
        msgs = [
            make_message("First", sender_name="Alice"),
            make_message("Second", sender_name="Bob"),
        ]
        result = format_messages(msgs)
        assert 'sender="Alice"' in result
        assert 'sender="Bob"' in result
        assert "First" in result
        assert "Second" in result

    def test_empty_list(self) -> None:
        """Empty list produces empty messages element."""
        result = format_messages([])
        assert "<messages>" in result
        assert "</messages>" in result

    def test_content_is_xml_escaped(self) -> None:
        """Message content with XML special chars is escaped."""
        msg = make_message("a < b & c > d")
        result = format_messages([msg])
        assert "&lt;" in result
        assert "&amp;" in result
        assert "&gt;" in result

    def test_sender_name_is_xml_escaped(self) -> None:
        """Sender names with quotes are escaped in attribute."""
        msg = make_message("hi", sender_name='Tom "T" Jones')
        result = format_messages([msg])
        assert "&quot;" in result


class TestStripInternalTags:
    """Tests for strip_internal_tags function."""

    def test_removes_internal_block(self) -> None:
        """Internal blocks are completely removed."""
        text = "Hello <internal>this is hidden</internal> world"
        assert strip_internal_tags(text) == "Hello  world"

    def test_removes_multiline_internal_block(self) -> None:
        """Multiline internal blocks are removed."""
        text = "Start <internal>\nline 1\nline 2\n</internal> End"
        assert strip_internal_tags(text) == "Start  End"

    def test_removes_multiple_internal_blocks(self) -> None:
        """Multiple internal blocks are all removed."""
        text = "<internal>A</internal> visible <internal>B</internal>"
        assert strip_internal_tags(text) == "visible"

    def test_no_internal_block_unchanged(self) -> None:
        """Text without internal blocks passes through unchanged."""
        assert strip_internal_tags("Hello world") == "Hello world"

    def test_trims_whitespace(self) -> None:
        """Result is trimmed of leading/trailing whitespace."""
        text = "  hello  "
        assert strip_internal_tags(text) == "hello"

    def test_only_internal_content_returns_empty(self) -> None:
        """Text that is entirely internal returns empty string."""
        text = "<internal>everything hidden</internal>"
        assert strip_internal_tags(text) == ""


class TestFormatOutbound:
    """Tests for format_outbound function."""

    def test_strips_internal_and_returns_clean(self) -> None:
        """Internal blocks stripped, clean text returned."""
        raw = "Here is my answer <internal>thinking...</internal>"
        assert format_outbound(raw) == "Here is my answer"

    def test_empty_returns_empty(self) -> None:
        """Empty input returns empty output."""
        assert format_outbound("") == ""

    def test_only_internal_returns_empty(self) -> None:
        """All-internal content returns empty string."""
        raw = "<internal>This should be hidden</internal>"
        assert format_outbound(raw) == ""

    def test_plain_text_unchanged(self) -> None:
        """Plain text without internal blocks passes through."""
        assert format_outbound("Hello, world!") == "Hello, world!"
# end tests/test_router.py
