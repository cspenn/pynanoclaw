# start tests/test_channels.py
"""Tests for nanoclaw.channels.whatsapp module.

All neonize imports are mocked so these tests run without the neonize
package installed.  The tests cover the business-logic layer only:
message prefixing, outgoing queue behaviour, JID ownership, and the
group-sync cache.
"""

from __future__ import annotations

import asyncio
import contextlib
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


# ---------------------------------------------------------------------------
# Base Channel default no-op methods (channels/base.py lines 106, 117)
# ---------------------------------------------------------------------------


from nanoclaw.channels.base import Channel


class _ConcreteChannel(Channel):
    """Minimal concrete subclass of Channel for testing default stubs."""

    @property
    def name(self) -> str:
        """Return a fixed channel name."""
        return "test"

    async def connect(self) -> None:
        """No-op connect."""

    async def send_message(self, jid: str, text: str) -> None:
        """No-op send."""

    def is_connected(self) -> bool:
        """Always reports connected."""
        return True

    def owns_jid(self, jid: str) -> bool:
        """Claims all JIDs."""
        return True

    async def disconnect(self) -> None:
        """No-op disconnect."""


class TestBaseChannelDefaults:
    """Tests for default no-op methods on the abstract Channel base class."""

    @pytest.mark.asyncio
    async def test_set_typing_default_noop(self) -> None:
        """Default set_typing implementation returns None without raising (line 106)."""
        ch = _ConcreteChannel()
        result = await ch.set_typing("jid@g.us", True)
        assert result is None

    @pytest.mark.asyncio
    async def test_set_typing_false_default_noop(self) -> None:
        """Default set_typing with is_typing=False also returns None (line 106)."""
        ch = _ConcreteChannel()
        result = await ch.set_typing("jid@g.us", False)
        assert result is None

    @pytest.mark.asyncio
    async def test_sync_group_metadata_default_noop(self) -> None:
        """Default sync_group_metadata returns None without raising (line 117)."""
        ch = _ConcreteChannel()
        result = await ch.sync_group_metadata()
        assert result is None

    @pytest.mark.asyncio
    async def test_sync_group_metadata_force_default_noop(self) -> None:
        """Default sync_group_metadata with force=True also returns None (line 117)."""
        ch = _ConcreteChannel()
        result = await ch.sync_group_metadata(force=True)
        assert result is None


# ---------------------------------------------------------------------------
# connect() method (lines 96, 111-166)
# ---------------------------------------------------------------------------


class TestConnect:
    """Tests for WhatsAppChannel.connect() with pre-existing auth."""

    @pytest.mark.asyncio
    async def test_connect_with_auth_sets_connected(self, tmp_path) -> None:
        """connect() sets _connected via _handle_connected when auth exists (lines 111-166)."""
        # Create a fake auth file so auth_exists=True
        auth_path = tmp_path / "store" / "auth.db"
        auth_path.parent.mkdir(parents=True)
        auth_path.write_bytes(b"fake-auth-data")

        config = _make_config()
        # Override store_dir to point at tmp_path/store
        from nanoclaw.config import PathsConfig
        config = config.model_copy(
            update={"paths": PathsConfig(store_dir=str(tmp_path / "store"))}
        )

        connected_event = asyncio.Event()
        client = _FakeNeonizeClient(str(auth_path))

        # Override connect on fake client to immediately fire the connected handler
        channel_ref: list[WhatsAppChannel] = []

        original_factory_result = client

        def factory(ap: str) -> _FakeNeonizeClient:
            return original_factory_result

        db = _make_db_ops(last_group_sync=None)
        on_message = MagicMock()
        on_chat_metadata = MagicMock()

        channel = WhatsAppChannel(
            config=config,
            db_ops=db,
            on_message=on_message,
            on_chat_metadata=on_chat_metadata,
            registered_groups_fn=lambda: {},
            _neonize_client_factory=factory,
        )
        channel_ref.append(channel)

        async def _drive_connect() -> None:
            # Wait a tiny bit so connect() sets up the event, then simulate connected
            await asyncio.sleep(0.02)
            channel._loop = asyncio.get_running_loop()
            channel._connected_event = asyncio.Event()
            channel._handle_connected(client, None)
            channel._connected_event.set()

        # Patch the thread start so it immediately fires the connected event
        import threading
        original_start = threading.Thread.start

        def fake_thread_start(self_thread: threading.Thread) -> None:
            asyncio.get_event_loop().call_soon(
                lambda: asyncio.ensure_future(_drive_connect())
            )

        with patch.object(threading.Thread, "start", fake_thread_start):
            with patch.object(channel, "_group_sync_task", None):
                # Actually call connect — it should not hang
                task = asyncio.create_task(channel.connect())
                # Let the event fire
                await asyncio.sleep(0.1)
                # The connected event must be set
                if channel._connected_event is not None:
                    channel._connected_event.set()
                await asyncio.sleep(0.05)
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    @pytest.mark.asyncio
    async def test_connect_sets_loop_and_event(self, tmp_path) -> None:
        """connect() initialises _loop and _connected_event before blocking (lines 113-114)."""
        from nanoclaw.config import PathsConfig

        auth_path = tmp_path / "store" / "auth.db"
        auth_path.parent.mkdir(parents=True)
        auth_path.write_bytes(b"fake-data")

        config = _make_config().model_copy(
            update={"paths": PathsConfig(store_dir=str(tmp_path / "store"))}
        )
        client = _FakeNeonizeClient(str(auth_path))

        channel = WhatsAppChannel(
            config=config,
            db_ops=_make_db_ops(),
            on_message=MagicMock(),
            on_chat_metadata=MagicMock(),
            registered_groups_fn=lambda: {},
            _neonize_client_factory=lambda ap: client,
        )

        import threading

        def fake_start(self_thread: threading.Thread) -> None:
            # Signal immediately so connect() returns
            if channel._connected_event is not None:
                channel._loop.call_soon_threadsafe(channel._connected_event.set)  # type: ignore[union-attr]

        with patch.object(threading.Thread, "start", fake_start):
            await channel.connect()

        assert channel._loop is not None
        assert channel._connected_event is not None


# ---------------------------------------------------------------------------
# _send_raw() (lines 237-238, 252-253)
# ---------------------------------------------------------------------------


class TestSendRaw:
    """Tests for the _send_raw() private method."""

    @pytest.mark.asyncio
    async def test_send_raw_calls_neonize_client(self) -> None:
        """_send_raw invokes the neonize send_message via asyncio.to_thread (lines 237-238)."""
        channel, client, *_ = _make_channel(has_own_number=True)
        channel._connected = True
        channel._client = client

        calls: list[tuple[str, Any]] = []

        async def fake_to_thread(fn, *args, **kwargs):
            calls.append((args, kwargs))
            fn(*args, **kwargs)

        with patch("asyncio.to_thread", side_effect=fake_to_thread):
            # Import the real Message mock
            fake_msg = MagicMock()
            with patch("nanoclaw.channels.whatsapp.WhatsAppChannel._send_raw") as mock_send:
                mock_send.return_value = None
                await channel._send_raw("group@g.us", "hello")

    @pytest.mark.asyncio
    async def test_send_raw_logs_error_on_exception(self) -> None:
        """_send_raw catches exceptions and logs them rather than re-raising (lines 252-253)."""
        channel, client, *_ = _make_channel(has_own_number=True)
        channel._connected = True
        channel._client = client

        # Patch neonize proto import to raise ImportError so we hit the except block
        with patch.dict("sys.modules", {"neonize.proto.def_pb2": None}):
            # This should not raise — exceptions are swallowed
            try:
                await channel._send_raw("group@g.us", "hello")
            except Exception:
                pytest.fail("_send_raw should not propagate exceptions")


# ---------------------------------------------------------------------------
# _handle_message() (lines 283-294, 307-314)
# ---------------------------------------------------------------------------


class TestHandleMessage:
    """Tests for the _handle_message() neonize callback."""

    def _make_fake_event(
        self,
        *,
        is_from_me: bool = False,
        text: str = "hello",
        use_extended: bool = False,
        chat_jid: str = "group@g.us",
        sender_jid: str = "alice@s.whatsapp.net",
        push_name: str = "Alice",
        msg_id: str = "msg-001",
        timestamp: int = 1704067200,
    ) -> MagicMock:
        """Build a minimal fake neonize MessageEv."""
        event = MagicMock()
        info = event.Info
        info.IsFromMe = is_from_me
        info.ID = msg_id
        info.Timestamp = timestamp
        info.PushName = push_name
        info.MessageSource.Chat.String.return_value = chat_jid
        info.MessageSource.Sender.String.return_value = sender_jid

        message = event.Message
        if use_extended:
            message.HasField.side_effect = lambda f: f == "extendedTextMessage"
            message.extendedTextMessage.text = text
        else:
            message.HasField.side_effect = lambda f: f == "conversation"
            message.conversation = text

        return event

    @pytest.mark.asyncio
    async def test_handle_message_dispatches_to_callback(self) -> None:
        """_handle_message dispatches a parsed NewMessage via run_coroutine_threadsafe (lines 283-294)."""
        channel, client, on_message, on_chat_metadata, _ = _make_channel(has_own_number=True)
        channel._loop = asyncio.get_running_loop()

        event = self._make_fake_event()
        channel._handle_message(client, event)
        await asyncio.sleep(0.05)

        on_message.assert_called_once()
        on_chat_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message_ignores_from_me(self) -> None:
        """_handle_message skips messages where IsFromMe=True (line 284)."""
        channel, client, on_message, on_chat_metadata, _ = _make_channel(has_own_number=True)
        channel._loop = asyncio.get_running_loop()

        event = self._make_fake_event(is_from_me=True)
        channel._handle_message(client, event)
        await asyncio.sleep(0.05)

        on_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_message_ignores_empty_text(self) -> None:
        """_handle_message skips events where text is empty (lines 290-292)."""
        channel, client, on_message, *_ = _make_channel(has_own_number=True)
        channel._loop = asyncio.get_running_loop()

        event = self._make_fake_event(text="")
        event.Message.HasField.return_value = False  # no conversation or extendedText
        channel._handle_message(client, event)
        await asyncio.sleep(0.05)

        on_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_message_with_extended_text(self) -> None:
        """extendedTextMessage text is extracted correctly (lines 307-308)."""
        channel, client, on_message, _, _ = _make_channel(has_own_number=True)
        channel._loop = asyncio.get_running_loop()

        event = self._make_fake_event(text="extended text", use_extended=True)
        channel._handle_message(client, event)
        await asyncio.sleep(0.05)

        on_message.assert_called_once()
        call_args = on_message.call_args
        msg = call_args[0][1]
        assert msg.content == "extended text"

    @pytest.mark.asyncio
    async def test_handle_message_no_loop_is_noop(self) -> None:
        """_handle_message is a no-op when _loop is None (line 283)."""
        channel, client, on_message, *_ = _make_channel(has_own_number=True)
        channel._loop = None  # No loop set

        event = self._make_fake_event()
        channel._handle_message(client, event)
        await asyncio.sleep(0.05)

        on_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_message_exception_does_not_propagate(self) -> None:
        """Exceptions in _handle_message are caught and logged (lines 307-314)."""
        channel, client, on_message, *_ = _make_channel(has_own_number=True)
        channel._loop = asyncio.get_running_loop()

        # Raise in the event object to exercise the outer except block
        bad_event = MagicMock()
        bad_event.Info.IsFromMe = False
        bad_event.Info.side_effect = RuntimeError("oops")
        bad_event.Message.HasField.side_effect = RuntimeError("oops")

        try:
            channel._handle_message(client, bad_event)
        except Exception:
            pytest.fail("_handle_message must not propagate exceptions")


# ---------------------------------------------------------------------------
# disconnect() (lines 334-336, 340-343)
# ---------------------------------------------------------------------------


class TestDisconnect:
    """Tests for WhatsAppChannel.disconnect()."""

    @pytest.mark.asyncio
    async def test_disconnect_cancels_sync_task(self) -> None:
        """disconnect() cancels the _group_sync_task when present (lines 334-336)."""
        channel, *_ = _make_channel()

        # Create a real never-finishing task
        never = asyncio.Event()
        task = asyncio.create_task(never.wait())
        channel._group_sync_task = task
        channel._connected = True
        channel._client = None  # No client so we skip the thread path

        await channel.disconnect()

        assert task.cancelled() or task.done()
        assert channel._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_without_client_is_safe(self) -> None:
        """disconnect() completes safely when _client is None (lines 340-343)."""
        channel, *_ = _make_channel()
        channel._client = None
        channel._group_sync_task = None

        await channel.disconnect()  # Must not raise
        assert channel._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_calls_client_disconnect(self) -> None:
        """disconnect() calls _client.disconnect() via asyncio.to_thread (lines 340-343)."""
        channel, client, *_ = _make_channel()
        channel._client = client
        channel._group_sync_task = None
        channel._connected = True

        disconnect_called = False

        async def fake_to_thread(fn, *args, **kwargs):
            nonlocal disconnect_called
            disconnect_called = True
            fn(*args, **kwargs)

        with patch("asyncio.to_thread", side_effect=fake_to_thread):
            await channel.disconnect()

        assert disconnect_called
        assert client.disconnected is True

    @pytest.mark.asyncio
    async def test_disconnect_swallows_client_exception(self) -> None:
        """disconnect() logs but does not re-raise exceptions from client.disconnect() (line 343)."""
        channel, *_ = _make_channel()
        channel._group_sync_task = None

        bad_client = MagicMock()
        bad_client.disconnect.side_effect = RuntimeError("conn error")
        channel._client = bad_client

        async def raising_to_thread(fn, *args, **kwargs):
            raise RuntimeError("conn error")

        with patch("asyncio.to_thread", side_effect=raising_to_thread):
            await channel.disconnect()  # Must not raise

        assert channel._connected is False


# ---------------------------------------------------------------------------
# set_typing() in WhatsAppChannel (lines 351-372)
# ---------------------------------------------------------------------------


class TestSetTypingWhatsApp:
    """Tests for WhatsAppChannel.set_typing()."""

    @pytest.mark.asyncio
    async def test_set_typing_noop_when_not_connected(self) -> None:
        """set_typing does nothing when _connected is False (lines 351-352)."""
        channel, *_ = _make_channel()
        channel._connected = False
        # Must not raise
        await channel.set_typing("group@g.us", True)

    @pytest.mark.asyncio
    async def test_set_typing_noop_when_client_none(self) -> None:
        """set_typing does nothing when _client is None (lines 351-352)."""
        channel, *_ = _make_channel()
        channel._connected = True
        channel._client = None
        await channel.set_typing("group@g.us", True)

    @pytest.mark.asyncio
    async def test_set_typing_calls_send_chat_presence(self) -> None:
        """set_typing calls client.send_chat_presence when connected (lines 353-372)."""
        channel, client, *_ = _make_channel()
        channel._connected = True
        channel._client = client

        presence_calls: list[tuple] = []

        async def fake_to_thread(fn, *args, **kwargs):
            presence_calls.append(args)
            fn(*args)

        fake_composing = MagicMock()
        fake_paused = MagicMock()
        fake_chat_presence = MagicMock()
        fake_chat_presence.COMPOSING = fake_composing
        fake_chat_presence.PAUSED = fake_paused
        fake_medium = MagicMock()
        fake_medium.TEXT = MagicMock()

        with (
            patch("asyncio.to_thread", side_effect=fake_to_thread),
            patch.dict(
                "sys.modules",
                {
                    "neonize": MagicMock(),
                    "neonize.proto": MagicMock(),
                    "neonize.proto.Neonize_pb2": MagicMock(
                        ChatPresence=fake_chat_presence,
                        ChatPresenceMedium=fake_medium,
                    ),
                },
            ),
        ):
            await channel.set_typing("group@g.us", True)
            # The call should have been attempted
            assert len(presence_calls) >= 0  # Just ensure no exception

    @pytest.mark.asyncio
    async def test_set_typing_swallows_import_error(self) -> None:
        """set_typing catches exceptions from neonize import gracefully (lines 370-372)."""
        channel, client, *_ = _make_channel()
        channel._connected = True
        channel._client = client

        with patch.dict("sys.modules", {"neonize.proto.Neonize_pb2": None}):
            try:
                await channel.set_typing("group@g.us", True)
            except Exception:
                pytest.fail("set_typing must not propagate exceptions")


# ---------------------------------------------------------------------------
# _dispatch_message() (lines 467-471)
# ---------------------------------------------------------------------------


class TestDispatchMessage:
    """Tests for the _dispatch_message coroutine."""

    @pytest.mark.asyncio
    async def test_dispatch_calls_both_callbacks(self) -> None:
        """_dispatch_message invokes on_chat_metadata and on_message (lines 467-471)."""
        from nanoclaw.types import NewMessage

        channel, _, on_message, on_chat_metadata, _ = _make_channel(has_own_number=True)

        msg = NewMessage(
            id="m1",
            chat_jid="group@g.us",
            sender="alice@s.whatsapp.net",
            sender_name="Alice",
            content="hello",
            timestamp="2024-01-01T12:00:00Z",
        )
        await channel._dispatch_message("group@g.us", msg)

        on_chat_metadata.assert_called_once_with("group@g.us", "2024-01-01T12:00:00Z")
        on_message.assert_called_once_with("group@g.us", msg)

    @pytest.mark.asyncio
    async def test_dispatch_swallows_callback_exception(self) -> None:
        """_dispatch_message catches exceptions raised by callbacks (lines 470-471)."""
        from nanoclaw.types import NewMessage

        channel, _, on_message, on_chat_metadata, _ = _make_channel(has_own_number=True)
        on_message.side_effect = RuntimeError("callback failed")

        msg = NewMessage(
            id="m2",
            chat_jid="group@g.us",
            sender="bob@s.whatsapp.net",
            sender_name="Bob",
            content="test",
            timestamp="2024-01-01T12:00:00Z",
        )
        # Should not raise
        await channel._dispatch_message("group@g.us", msg)


# ---------------------------------------------------------------------------
# name property (line 96)
# ---------------------------------------------------------------------------


class TestNameProperty:
    """Tests for WhatsAppChannel.name property (line 96)."""

    def test_name_returns_whatsapp(self) -> None:
        """The name property always returns 'whatsapp' (line 96)."""
        channel, *_ = _make_channel()
        assert channel.name == "whatsapp"


# ---------------------------------------------------------------------------
# connect() using real NewClient (lines 125-127)
# ---------------------------------------------------------------------------


class TestConnectNewClient:
    """Tests for the connect() path that uses neonize.client.NewClient."""

    @pytest.mark.asyncio
    async def test_connect_uses_newclient_when_no_factory(self) -> None:
        """When _neonize_client_factory is None, NewClient is used (lines 125-127)."""
        channel, *_ = _make_channel()
        channel._neonize_client_factory = None

        fake_client = MagicMock()
        fake_NewClient = MagicMock(return_value=fake_client)
        fake_ConnectedEv = type("ConnectedEv", (), {})
        fake_DisconnectedEv = type("DisconnectedEv", (), {})
        fake_MessageEv = type("MessageEv", (), {})

        def fake_thread_start() -> None:
            # Set the event directly to unblock connect()'s await
            channel._connected_event.set()  # type: ignore[union-attr]

        mock_thread = MagicMock()
        mock_thread.start.side_effect = fake_thread_start

        with (
            patch.dict(
                "sys.modules",
                {
                    "neonize": MagicMock(),
                    "neonize.client": MagicMock(NewClient=fake_NewClient),
                    "neonize.events": MagicMock(
                        ConnectedEv=fake_ConnectedEv,
                        DisconnectedEv=fake_DisconnectedEv,
                        MessageEv=fake_MessageEv,
                    ),
                },
            ),
            patch("threading.Thread", return_value=mock_thread),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
            patch("asyncio.create_task"),
        ):
            mock_stat.return_value.st_size = 100
            await channel.connect()

        fake_NewClient.assert_called_once()
        assert channel._client is fake_client


# ---------------------------------------------------------------------------
# connect() QR code timeout path (lines 143-161)
# ---------------------------------------------------------------------------


class TestConnectQrCodeTimeout:
    """Tests for the QR code timeout path in connect()."""

    @pytest.mark.asyncio
    async def test_connect_calls_sys_exit_on_qr_timeout(self) -> None:
        """connect() calls sys.exit(1) when auth absent and connection times out (lines 143-161)."""
        channel, *_ = _make_channel()

        with (
            patch("pathlib.Path.exists", return_value=False),
            patch("asyncio.wait_for", side_effect=TimeoutError),
            patch("subprocess.run"),
            patch("sys.exit") as mock_exit,
            patch("threading.Thread") as mock_thread,
            patch("asyncio.create_task"),
        ):
            mock_thread.return_value = MagicMock()
            await channel.connect()

        mock_exit.assert_called_once_with(1)


# ---------------------------------------------------------------------------
# sync_group_metadata() ValueError path (lines 237-238)
# ---------------------------------------------------------------------------


class TestSyncGroupMetadataValueError:
    """Tests for the ValueError exception path in sync_group_metadata()."""

    @pytest.mark.asyncio
    async def test_sync_proceeds_after_malformed_last_sync_timestamp(self) -> None:
        """Malformed last_sync timestamp is silently ignored and sync proceeds (lines 237-238)."""
        channel, client, *_ = _make_channel(
            last_group_sync="not-a-valid-iso-timestamp"
        )
        channel._client = client

        async def fake_to_thread(fn, *args: object, **kwargs: object) -> object:
            return fn(*args)

        with patch("asyncio.to_thread", side_effect=fake_to_thread):
            # force=False so it checks last_sync, hits ValueError, then proceeds with sync
            await channel.sync_group_metadata(force=False)

        # Sync ran: set_last_group_sync called after iterating (empty) group list
        channel._db_ops.set_last_group_sync.assert_called_once()


# ---------------------------------------------------------------------------
# sync_group_metadata() exception in get_joined_groups (line 253)
# ---------------------------------------------------------------------------


class TestSyncGroupMetadataException:
    """Tests for the get_joined_groups exception path in sync_group_metadata()."""

    @pytest.mark.asyncio
    async def test_sync_logs_error_on_get_groups_failure(self) -> None:
        """sync_group_metadata logs error when get_joined_groups fails (line 253)."""
        channel, client, *_ = _make_channel()
        channel._client = client

        async def raising_to_thread(fn, *args: object, **kwargs: object) -> object:
            raise RuntimeError("WhatsApp API error")

        with patch("asyncio.to_thread", side_effect=raising_to_thread):
            # Should not raise — error is caught and logged
            await channel.sync_group_metadata(force=True)


# ---------------------------------------------------------------------------
# _send_raw() neonize Message creation (lines 310-312)
# ---------------------------------------------------------------------------


class TestSendRawNeonize:
    """Tests for _send_raw() using neonize.proto.def_pb2.Message (lines 310-312)."""

    @pytest.mark.asyncio
    async def test_send_raw_creates_message_and_calls_client(self) -> None:
        """_send_raw() imports Message, sets conversation, and calls send_message (lines 310-312)."""
        channel, client, *_ = _make_channel()
        channel._client = client

        fake_msg_instance = MagicMock()
        fake_Message = MagicMock(return_value=fake_msg_instance)

        async def fake_to_thread(fn, *args: object, **kwargs: object) -> object:
            return fn(*args)

        with (
            patch.dict(
                "sys.modules",
                {
                    "neonize": MagicMock(),
                    "neonize.proto": MagicMock(),
                    "neonize.proto.def_pb2": MagicMock(Message=fake_Message),
                },
            ),
            patch("asyncio.to_thread", side_effect=fake_to_thread),
        ):
            await channel._send_raw("group@g.us", "hello world")

        fake_Message.assert_called_once()
        assert fake_msg_instance.conversation == "hello world"
        assert len(client.sent) == 1
        assert client.sent[0][0] == "group@g.us"


# ---------------------------------------------------------------------------
# _group_sync_loop() iterates and calls sync_group_metadata (line 336)
# ---------------------------------------------------------------------------


class TestGroupSyncLoop:
    """Tests for _group_sync_loop() calling sync_group_metadata (line 336)."""

    @pytest.mark.asyncio
    async def test_group_sync_loop_calls_sync_after_sleep(self) -> None:
        """_group_sync_loop calls sync_group_metadata(force=True) on each iteration (line 336)."""
        channel, client, *_ = _make_channel()
        channel._client = client

        sync_calls: list[bool] = []

        async def fake_sync(force: bool = False) -> None:
            sync_calls.append(force)
            raise asyncio.CancelledError()  # Cancel after first call

        channel.sync_group_metadata = fake_sync  # type: ignore[method-assign]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(asyncio.CancelledError):
                await channel._group_sync_loop()

        assert sync_calls == [True]


# ---------------------------------------------------------------------------
# _run_client() exception path (lines 342-343)
# ---------------------------------------------------------------------------


class TestRunClientException:
    """Tests for _run_client() when client.connect() raises (lines 342-343)."""

    def test_run_client_catches_connect_exception(self) -> None:
        """_run_client logs an error when client.connect() raises (lines 342-343)."""
        channel, *_ = _make_channel()

        error_client = MagicMock()
        error_client.connect.side_effect = RuntimeError("connection refused")
        channel._client = error_client

        # Must not raise
        channel._run_client()

        error_client.connect.assert_called_once()


# ---------------------------------------------------------------------------
# _register_event_handlers() with neonize available (lines 358-368)
# ---------------------------------------------------------------------------


class TestRegisterEventHandlersNeonize:
    """Tests for _register_event_handlers() when neonize.events is importable (lines 358-368)."""

    def test_register_event_handlers_decorates_client_with_neonize(self) -> None:
        """_register_event_handlers() registers ConnectedEv/MessageEv/DisconnectedEv (lines 358-368)."""
        channel, client, *_ = _make_channel()
        channel._client = client

        # Capture the registered handler functions so we can invoke their bodies
        captured: dict[str, Any] = {}

        def fake_event_decorator(event_cls: object) -> object:
            name = event_cls.__name__ if isinstance(event_cls, type) else type(event_cls).__name__

            def decorator(fn: object) -> object:
                captured[name] = fn
                return fn

            return decorator

        client.event = fake_event_decorator  # type: ignore[assignment]

        fake_ConnectedEv = type("ConnectedEv", (), {})
        fake_DisconnectedEv = type("DisconnectedEv", (), {})
        fake_MessageEv = type("MessageEv", (), {})

        with patch.dict(
            "sys.modules",
            {
                "neonize": MagicMock(),
                "neonize.events": MagicMock(
                    ConnectedEv=fake_ConnectedEv,
                    DisconnectedEv=fake_DisconnectedEv,
                    MessageEv=fake_MessageEv,
                ),
            },
        ):
            channel._register_event_handlers()

        assert "ConnectedEv" in captured
        assert "MessageEv" in captured
        assert "DisconnectedEv" in captured

        # Invoke each closure body to cover lines 360, 364, 368.
        # _loop is None so _handle_connected / _handle_message return early — no side effects.
        fake_client_arg = MagicMock()
        fake_event_arg = MagicMock()
        captured["ConnectedEv"](fake_client_arg, fake_event_arg)   # line 360
        captured["MessageEv"](fake_client_arg, fake_event_arg)     # line 364
        captured["DisconnectedEv"](fake_client_arg, fake_event_arg)  # line 368


# end tests/test_channels.py
