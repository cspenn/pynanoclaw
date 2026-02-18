# start tests/test_channels.py
"""Tests for nanoclaw.channels.whatsapp module.

All neonize imports are mocked so these tests run without the neonize
package installed.  The tests cover the business-logic layer only:
message prefixing, outgoing queue behaviour, JID ownership, and the
group-sync cache.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanoclaw.channels.whatsapp import GROUP_SYNC_INTERVAL_S, WhatsAppChannel
from nanoclaw.config import AppConfig, AssistantConfig, PathsConfig
from nanoclaw.types import RegisteredGroup


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_config(
    *,
    has_own_number: bool = False,
    assistant_name: str = "Andy",
) -> AppConfig:
    """Build a minimal AppConfig for testing.

    Args:
        has_own_number: Value for ``assistant.has_own_number``.
        assistant_name: Value for ``assistant.name``.

    Returns:
        A configured ``AppConfig`` instance.
    """
    return AppConfig(
        assistant=AssistantConfig(name=assistant_name, has_own_number=has_own_number),
        paths=PathsConfig(store_dir="/tmp/test_store"),
    )


def _make_db_ops(*, last_group_sync: str | None = None) -> MagicMock:
    """Build a mock db_ops object with sensible defaults.

    Args:
        last_group_sync: Value returned by ``get_last_group_sync()``.

    Returns:
        A ``MagicMock`` pre-configured with the required methods.
    """
    db = MagicMock()
    db.get_last_group_sync.return_value = last_group_sync
    db.set_last_group_sync.return_value = None
    db.update_chat_name.return_value = None
    return db


class _FakeNeonizeClient:
    """Minimal fake neonize client for dependency injection in tests.

    Records calls to ``send_message`` and ``disconnect`` so tests can
    assert on them.  ``event()`` is a no-op decorator (we drive connected/
    disconnected behaviour directly via the channel's private handlers).
    """

    def __init__(self, auth_path: str) -> None:
        """Initialise recording state.

        Args:
            auth_path: The auth file path (stored but not used).
        """
        self.auth_path = auth_path
        self.sent: list[tuple[str, Any]] = []
        self.disconnected = False
        self.connect_called = False

    def send_message(self, jid: str, message: Any) -> None:
        """Record a send_message call.

        Args:
            jid: Destination JID.
            message: Message payload.
        """
        self.sent.append((jid, message))

    def disconnect(self) -> None:
        """Record a disconnect call."""
        self.disconnected = True

    def connect(self) -> None:
        """Record a connect call (does not block)."""
        self.connect_called = True

    def event(self, event_type: Any) -> Callable[[Any], Any]:
        """No-op decorator so handler registration does not raise.

        Args:
            event_type: Neonize event class (ignored).

        Returns:
            Identity decorator.
        """
        def decorator(fn: Any) -> Any:
            return fn
        return decorator

    def get_joined_groups(self) -> list[Any]:
        """Return empty group list by default.

        Returns:
            Empty list.
        """
        return []


def _make_channel(
    *,
    has_own_number: bool = False,
    assistant_name: str = "Andy",
    last_group_sync: str | None = None,
    fake_client: _FakeNeonizeClient | None = None,
) -> tuple[WhatsAppChannel, _FakeNeonizeClient, MagicMock, MagicMock, MagicMock]:
    """Create a WhatsAppChannel wired up with fakes.

    Args:
        has_own_number: Whether the bot has its own dedicated number.
        assistant_name: The assistant's name.
        last_group_sync: Stub value for ``get_last_group_sync()``.
        fake_client: Optional pre-built fake client; one is created if
            not supplied.

    Returns:
        Tuple of ``(channel, fake_client, on_message_mock,
        on_chat_metadata_mock, db_ops_mock)``.
    """
    config = _make_config(
        has_own_number=has_own_number, assistant_name=assistant_name
    )
    db = _make_db_ops(last_group_sync=last_group_sync)
    on_message = MagicMock()
    on_chat_metadata = MagicMock()
    registered_groups_fn: Callable[[], dict[str, RegisteredGroup]] = lambda: {}

    client = fake_client or _FakeNeonizeClient("/tmp/test_auth.db")

    def factory(auth_path: str) -> _FakeNeonizeClient:
        return client

    channel = WhatsAppChannel(
        config=config,
        db_ops=db,
        on_message=on_message,
        on_chat_metadata=on_chat_metadata,
        registered_groups_fn=registered_groups_fn,
        _neonize_client_factory=factory,
    )
    return channel, client, on_message, on_chat_metadata, db


# ---------------------------------------------------------------------------
# owns_jid
# ---------------------------------------------------------------------------


class TestOwnsJid:
    """Tests for WhatsAppChannel.owns_jid."""

    def test_group_jid_owned(self) -> None:
        """@g.us JIDs belong to WhatsApp."""
        channel, *_ = _make_channel()
        assert channel.owns_jid("12345-67890@g.us") is True

    def test_individual_jid_owned(self) -> None:
        """@s.whatsapp.net JIDs belong to WhatsApp."""
        channel, *_ = _make_channel()
        assert channel.owns_jid("15551234567@s.whatsapp.net") is True

    def test_telegram_jid_not_owned(self) -> None:
        """Non-WhatsApp JIDs are not owned."""
        channel, *_ = _make_channel()
        assert channel.owns_jid("user@telegram.org") is False

    def test_plain_string_not_owned(self) -> None:
        """Random strings are not owned."""
        channel, *_ = _make_channel()
        assert channel.owns_jid("not-a-jid") is False

    def test_empty_string_not_owned(self) -> None:
        """Empty string is not owned."""
        channel, *_ = _make_channel()
        assert channel.owns_jid("") is False


# ---------------------------------------------------------------------------
# is_connected
# ---------------------------------------------------------------------------


class TestIsConnected:
    """Tests for WhatsAppChannel.is_connected."""

    def test_initially_not_connected(self) -> None:
        """Channel starts in disconnected state before connect() is called."""
        channel, *_ = _make_channel()
        assert channel.is_connected() is False

    @pytest.mark.asyncio
    async def test_connected_after_handle_connected(self) -> None:
        """is_connected returns True after the connected handler fires."""
        channel, client, *_ = _make_channel()
        # Wire the running event loop so coroutines dispatched by the
        # handler are properly scheduled (and therefore awaitable).
        channel._loop = asyncio.get_running_loop()
        channel._connected_event = asyncio.Event()
        channel._handle_connected(client, None)
        assert channel.is_connected() is True
        # Drain any coroutines that were submitted via run_coroutine_threadsafe.
        await asyncio.sleep(0)

    def test_disconnected_after_handle_disconnected(self) -> None:
        """is_connected returns False after the disconnected handler fires."""
        channel, client, *_ = _make_channel()
        channel._connected = True
        channel._handle_disconnected(client, None)
        assert channel.is_connected() is False


# ---------------------------------------------------------------------------
# send_message — prefix behaviour
# ---------------------------------------------------------------------------


class TestSendMessagePrefix:
    """Tests for outgoing message prefix logic."""

    @pytest.mark.asyncio
    async def test_prefix_added_when_no_own_number(self) -> None:
        """When has_own_number=False the assistant name is prepended."""
        channel, client, *_ = _make_channel(
            has_own_number=False, assistant_name="Andy"
        )
        channel._connected = True
        channel._client = client

        sent_texts: list[str] = []

        async def fake_send_raw(jid: str, text: str) -> None:
            sent_texts.append(text)

        channel._send_raw = fake_send_raw  # type: ignore[method-assign]

        await channel.send_message("123@g.us", "Hello world")
        assert sent_texts == ["Andy: Hello world"]

    @pytest.mark.asyncio
    async def test_no_prefix_when_has_own_number(self) -> None:
        """When has_own_number=True the text is sent verbatim."""
        channel, client, *_ = _make_channel(
            has_own_number=True, assistant_name="Andy"
        )
        channel._connected = True
        channel._client = client

        sent_texts: list[str] = []

        async def fake_send_raw(jid: str, text: str) -> None:
            sent_texts.append(text)

        channel._send_raw = fake_send_raw  # type: ignore[method-assign]

        await channel.send_message("123@g.us", "Hello world")
        assert sent_texts == ["Hello world"]

    @pytest.mark.asyncio
    async def test_prefix_uses_configured_assistant_name(self) -> None:
        """The assistant name from config is used in the prefix."""
        channel, client, *_ = _make_channel(
            has_own_number=False, assistant_name="Zara"
        )
        channel._connected = True
        channel._client = client

        sent_texts: list[str] = []

        async def fake_send_raw(jid: str, text: str) -> None:
            sent_texts.append(text)

        channel._send_raw = fake_send_raw  # type: ignore[method-assign]

        await channel.send_message("123@g.us", "Hey!")
        assert sent_texts == ["Zara: Hey!"]


# ---------------------------------------------------------------------------
# send_message — outgoing queue
# ---------------------------------------------------------------------------


class TestOutgoingQueue:
    """Tests for disconnected-state message queueing."""

    @pytest.mark.asyncio
    async def test_message_queued_when_disconnected(self) -> None:
        """Messages are added to the queue when not connected."""
        channel, *_ = _make_channel(has_own_number=True)
        # _connected defaults to False.

        await channel.send_message("123@g.us", "queued message")
        assert len(channel._outgoing_queue) == 1
        assert channel._outgoing_queue[0] == ("123@g.us", "queued message")

    @pytest.mark.asyncio
    async def test_multiple_messages_queued_in_order(self) -> None:
        """Multiple queued messages are stored in order."""
        channel, *_ = _make_channel(has_own_number=True)

        await channel.send_message("a@g.us", "first")
        await channel.send_message("b@g.us", "second")
        assert channel._outgoing_queue[0] == ("a@g.us", "first")
        assert channel._outgoing_queue[1] == ("b@g.us", "second")

    @pytest.mark.asyncio
    async def test_queue_flushed_on_reconnect(self) -> None:
        """All queued messages are sent when _flush_outgoing_queue is called."""
        channel, *_ = _make_channel(has_own_number=True)
        channel._outgoing_queue = [
            ("a@g.us", "msg1"),
            ("b@g.us", "msg2"),
        ]

        sent: list[tuple[str, str]] = []

        async def fake_send_raw(jid: str, text: str) -> None:
            sent.append((jid, text))

        channel._send_raw = fake_send_raw  # type: ignore[method-assign]
        channel._connected = True

        await channel._flush_outgoing_queue()

        assert sent == [("a@g.us", "msg1"), ("b@g.us", "msg2")]
        assert channel._outgoing_queue == []

    @pytest.mark.asyncio
    async def test_flush_noop_when_queue_empty(self) -> None:
        """Flushing an empty queue completes without error."""
        channel, *_ = _make_channel(has_own_number=True)
        channel._connected = True

        sent: list[tuple[str, str]] = []

        async def fake_send_raw(jid: str, text: str) -> None:  # pragma: no cover
            sent.append((jid, text))

        channel._send_raw = fake_send_raw  # type: ignore[method-assign]

        await channel._flush_outgoing_queue()
        assert sent == []

    @pytest.mark.asyncio
    async def test_not_sent_directly_when_disconnected(self) -> None:
        """When disconnected, _send_raw is NOT called immediately."""
        channel, *_ = _make_channel(has_own_number=True)
        called = False

        async def fake_send_raw(jid: str, text: str) -> None:  # pragma: no cover
            nonlocal called
            called = True

        channel._send_raw = fake_send_raw  # type: ignore[method-assign]

        await channel.send_message("123@g.us", "hello")
        assert not called


# ---------------------------------------------------------------------------
# sync_group_metadata — 24-hour cache
# ---------------------------------------------------------------------------


class TestSyncGroupMetadata:
    """Tests for the 24-hour group-sync cache behaviour."""

    @pytest.mark.asyncio
    async def test_skips_sync_when_recently_synced(self) -> None:
        """Sync is skipped when last sync was less than 24 hours ago."""
        from datetime import datetime, timezone, timedelta

        recent = (
            datetime.now(timezone.utc) - timedelta(hours=1)
        ).isoformat()

        channel, client, *_, db = _make_channel(last_group_sync=recent)
        channel._connected = True
        channel._client = client

        await channel.sync_group_metadata(force=False)

        db.update_chat_name.assert_not_called()
        db.set_last_group_sync.assert_not_called()

    @pytest.mark.asyncio
    async def test_syncs_when_no_previous_sync(self) -> None:
        """Sync runs when there is no recorded previous sync."""
        channel, client, *_, db = _make_channel(last_group_sync=None)
        channel._connected = True
        channel._client = client

        # Patch get_joined_groups to return one fake group.
        fake_group = MagicMock()
        fake_group.JID.String.return_value = "group1@g.us"
        fake_group.Name = "Test Group"
        client.get_joined_groups = lambda: [fake_group]

        await channel.sync_group_metadata(force=False)

        db.update_chat_name.assert_called_once_with("group1@g.us", "Test Group")
        db.set_last_group_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_force_bypasses_cache(self) -> None:
        """force=True runs the sync even if a recent sync timestamp exists."""
        from datetime import datetime, timezone, timedelta

        recent = (
            datetime.now(timezone.utc) - timedelta(minutes=5)
        ).isoformat()

        channel, client, *_, db = _make_channel(last_group_sync=recent)
        channel._connected = True
        channel._client = client

        fake_group = MagicMock()
        fake_group.JID.String.return_value = "grp@g.us"
        fake_group.Name = "My Group"
        client.get_joined_groups = lambda: [fake_group]

        await channel.sync_group_metadata(force=True)

        db.update_chat_name.assert_called_once_with("grp@g.us", "My Group")
        db.set_last_group_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_syncs_when_last_sync_expired(self) -> None:
        """Sync runs when the last sync was more than 24 hours ago."""
        from datetime import datetime, timezone, timedelta

        old = (
            datetime.now(timezone.utc) - timedelta(hours=25)
        ).isoformat()

        channel, client, *_, db = _make_channel(last_group_sync=old)
        channel._connected = True
        channel._client = client

        fake_group = MagicMock()
        fake_group.JID.String.return_value = "grp2@g.us"
        fake_group.Name = "Old Group"
        client.get_joined_groups = lambda: [fake_group]

        await channel.sync_group_metadata(force=False)

        db.update_chat_name.assert_called_once_with("grp2@g.us", "Old Group")
        db.set_last_group_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_skipped_when_client_none(self) -> None:
        """sync_group_metadata is a no-op when no client is available."""
        channel, *_, db = _make_channel(last_group_sync=None)
        # Do NOT assign a client.
        channel._client = None

        await channel.sync_group_metadata(force=True)

        db.update_chat_name.assert_not_called()
        db.set_last_group_sync.assert_not_called()
# end tests/test_channels.py
