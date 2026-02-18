# start src/nanoclaw/channels/whatsapp.py
"""WhatsApp channel implementation using neonize.

Connects to WhatsApp via the neonize library (Python bindings for the
whatsmeow Go library). Handles connection, inbound message routing,
outbound message sending with a disconnect queue, and periodic group
metadata sync.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import subprocess
import sys
import threading
from collections.abc import Callable
from datetime import UTC
from typing import Any

from nanoclaw.channels.base import Channel, OnChatMetadata, OnInboundMessage
from nanoclaw.config import AppConfig
from nanoclaw.types import NewMessage, RegisteredGroup

logger = logging.getLogger(__name__)

GROUP_SYNC_INTERVAL_S = 24 * 60 * 60  # 24 hours


class WhatsAppChannel(Channel):
    """WhatsApp channel backed by neonize (whatsmeow Go library).

    Manages a persistent WhatsApp connection, delivers inbound messages
    via the ``on_message`` callback, and sends outbound messages with
    automatic queueing while disconnected. Group metadata is synced on
    connect and refreshed every 24 hours.

    Attributes:
        name: Always ``'whatsapp'``.
    """

    def __init__(
        self,
        config: AppConfig,
        db_ops: Any,
        on_message: OnInboundMessage,
        on_chat_metadata: OnChatMetadata,
        registered_groups_fn: Callable[[], dict[str, RegisteredGroup]],
        *,
        _neonize_client_factory: Callable[..., Any] | None = None,
    ) -> None:
        """Initialise the channel (does not connect yet).

        Args:
            config: Application configuration.
            db_ops: Database operations module (must expose
                ``get_last_group_sync``, ``set_last_group_sync``, and
                ``update_chat_name``).
            on_message: Callback invoked for every inbound message.
            on_chat_metadata: Callback invoked with ``(jid, timestamp)``
                when chat metadata is discovered.
            registered_groups_fn: Callable that returns the current set of
                registered groups keyed by JID.
            _neonize_client_factory: Optional factory override for
                dependency injection in tests. When *None* the real
                ``neonize.client.NewClient`` is used.
        """
        self._config = config
        self._db_ops = db_ops
        self._on_message = on_message
        self._on_chat_metadata = on_chat_metadata
        self._registered_groups_fn = registered_groups_fn
        self._neonize_client_factory = _neonize_client_factory

        self._client: Any = None
        self._connected = False
        self._connected_event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        self._outgoing_queue: list[tuple[str, str]] = []
        self._flushing = False
        self._group_sync_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Channel interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable channel identifier.

        Returns:
            Always ``'whatsapp'``.
        """
        return "whatsapp"

    async def connect(self) -> None:
        """Connect to WhatsApp.

        Creates a neonize client, registers event handlers, starts the
        blocking ``client.connect()`` in a background thread, then waits
        for the first ``ConnectedEv`` before returning.

        If no auth state exists and a QR code is required the user is
        notified via a macOS dialog (osascript) and the process exits.

        Raises:
            SystemExit: If QR code setup is required.
        """
        from pathlib import Path

        self._loop = asyncio.get_running_loop()
        self._connected_event = asyncio.Event()

        store_dir = Path(self._config.paths.store_dir)
        auth_path = str(store_dir / "auth.db")

        # Determine whether auth state exists before creating the client.
        auth_exists = Path(auth_path).exists() and Path(auth_path).stat().st_size > 0

        if self._neonize_client_factory is not None:
            self._client = self._neonize_client_factory(auth_path)
        else:
            from neonize.client import NewClient

            self._client = NewClient(auth_path)

        # Register event handlers.
        self._register_event_handlers()

        # Start the blocking connect call in a daemon thread.
        connect_thread = threading.Thread(
            target=self._run_client,
            daemon=True,
            name="neonize-connect",
        )
        connect_thread.start()

        # If there is no pre-existing auth we expect a QR code.  Give the
        # thread a moment then bail if still not connected.
        if not auth_exists:
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._connected_event.wait()),
                    timeout=1.0,
                )
            except TimeoutError:
                msg = (
                    f"NanoClaw ({self._config.assistant.name}): "
                    "WhatsApp QR code scan required. "
                    "Run the setup skill to authenticate."
                )
                logger.error(msg)
                with contextlib.suppress(Exception):
                    subprocess.run(
                        ["osascript", "-e", f'display dialog "{msg}" buttons {{"OK"}}'],
                        check=False,
                        timeout=5,
                    )
                sys.exit(1)
        else:
            await self._connected_event.wait()

        # Kick off periodic group sync.
        self._group_sync_task = asyncio.create_task(self._group_sync_loop())

    async def send_message(self, jid: str, text: str) -> None:
        """Send a text message to the specified JID.

        Prepends ``"{assistant_name}: "`` when ``has_own_number`` is
        False (the bot shares a human's phone number).

        If the channel is not currently connected the message is queued
        and will be flushed automatically on the next reconnect.

        Args:
            jid: Destination WhatsApp JID.
            text: Message body.
        """
        if not self._config.assistant.has_own_number:
            text = f"{self._config.assistant.name}: {text}"

        if not self._connected:
            logger.debug("Not connected — queuing message to %s", jid)
            self._outgoing_queue.append((jid, text))
            return

        await self._send_raw(jid, text)

    async def set_typing(self, jid: str, is_typing: bool) -> None:
        """Send a typing presence indicator.

        Args:
            jid: The JID to send the indicator to.
            is_typing: ``True`` for composing, ``False`` for paused.
        """
        if not self._connected or self._client is None:
            return
        try:
            from neonize.proto.Neonize_pb2 import (
                ChatPresence,
                ChatPresenceMedium,
            )

            presence = ChatPresence.COMPOSING if is_typing else ChatPresence.PAUSED
            await asyncio.to_thread(
                self._client.send_chat_presence,
                jid,
                presence,
                ChatPresenceMedium.TEXT,
            )
        except Exception as exc:
            logger.warning("set_typing failed for %s: %s", jid, exc)

    async def sync_group_metadata(self, force: bool = False) -> None:
        """Sync group name/metadata from WhatsApp.

        Fetches all joined groups and updates the chat names in the
        database.  Skips the sync if a full sync was performed within the
        last 24 hours, unless *force* is ``True``.

        Args:
            force: When ``True`` bypass the 24-hour cache and always sync.
        """
        if not force:
            last_sync = self._db_ops.get_last_group_sync()
            if last_sync is not None:
                from datetime import datetime

                try:
                    last_dt = datetime.fromisoformat(last_sync)
                    age_s = (datetime.now(UTC) - last_dt).total_seconds()
                    if age_s < GROUP_SYNC_INTERVAL_S:
                        logger.debug("Skipping group sync — last sync was %.0f s ago", age_s)
                        return
                except ValueError:
                    pass  # Malformed timestamp — proceed with sync.

        if self._client is None:
            return

        logger.info("Syncing WhatsApp group metadata…")
        try:
            groups = await asyncio.to_thread(self._client.get_joined_groups)
            for group in groups:
                jid_str: str = group.JID.String()
                name: str = group.Name
                self._db_ops.update_chat_name(jid_str, name)
            self._db_ops.set_last_group_sync()
            logger.info("Group sync complete — %d groups updated.", len(groups))
        except Exception as exc:
            logger.error("Group metadata sync failed: %s", exc)

    def is_connected(self) -> bool:
        """Return whether the channel is currently connected.

        Returns:
            ``True`` if a live WhatsApp session is active.
        """
        return self._connected

    def owns_jid(self, jid: str) -> bool:
        """Return whether this channel handles the given JID.

        WhatsApp owns all group JIDs (``@g.us``) and individual JIDs
        (``@s.whatsapp.net``).

        Args:
            jid: JID to test.

        Returns:
            ``True`` for ``@g.us`` and ``@s.whatsapp.net`` JIDs.
        """
        return jid.endswith("@g.us") or jid.endswith("@s.whatsapp.net")

    async def disconnect(self) -> None:
        """Disconnect from WhatsApp gracefully.

        Cancels the background group sync task and disconnects the
        neonize client if present.
        """
        if self._group_sync_task is not None:
            self._group_sync_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._group_sync_task

        self._connected = False

        if self._client is not None:
            try:
                await asyncio.to_thread(self._client.disconnect)
            except Exception as exc:
                logger.warning("Error during disconnect: %s", exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _send_raw(self, jid: str, text: str) -> None:
        """Invoke the neonize send_message without further transformation.

        Args:
            jid: Destination JID.
            text: Pre-formatted message text.
        """
        try:
            from neonize.proto.def_pb2 import Message

            msg = Message()
            msg.conversation = text
            await asyncio.to_thread(self._client.send_message, jid, msg)
        except Exception as exc:
            logger.error("Failed to send message to %s: %s", jid, exc)

    async def _flush_outgoing_queue(self) -> None:
        """Drain the outgoing queue and send all pending messages.

        No-op if a flush is already in progress or the queue is empty.
        """
        if self._flushing or not self._outgoing_queue:
            return
        self._flushing = True
        try:
            while self._outgoing_queue:
                jid, text = self._outgoing_queue.pop(0)
                logger.debug("Flushing queued message to %s", jid)
                await self._send_raw(jid, text)
        finally:
            self._flushing = False

    async def _group_sync_loop(self) -> None:
        """Periodically re-sync group metadata every 24 hours."""
        while True:
            await asyncio.sleep(GROUP_SYNC_INTERVAL_S)
            await self.sync_group_metadata(force=True)

    def _run_client(self) -> None:
        """Target for the background thread: block on neonize connect."""
        try:
            self._client.connect()
        except Exception as exc:
            logger.error("neonize client.connect() raised: %s", exc)

    def _register_event_handlers(self) -> None:
        """Attach neonize event callbacks to the client.

        Falls back gracefully if neonize is not installed (test environments
        that supply a mock client skip real event registration).
        """
        try:
            from neonize.events import (
                ConnectedEv,
                DisconnectedEv,
                MessageEv,
            )

            @self._client.event(ConnectedEv)  # type: ignore[untyped-decorator]
            def _on_connected(client: Any, event: Any) -> None:
                self._handle_connected(client, event)

            @self._client.event(MessageEv)  # type: ignore[untyped-decorator]
            def _on_message(client: Any, event: Any) -> None:
                self._handle_message(client, event)

            @self._client.event(DisconnectedEv)  # type: ignore[untyped-decorator]
            def _on_disconnected(client: Any, event: Any) -> None:
                self._handle_disconnected(client, event)

        except ImportError:
            # Running without neonize (tests with injected mock client).
            pass

    def _handle_connected(self, client: Any, event: Any) -> None:
        """Neonize ConnectedEv callback.

        Marks the channel as connected, signals the asyncio event so that
        ``connect()`` can unblock, then flushes any queued outgoing
        messages.

        Args:
            client: The neonize client instance.
            event: The ConnectedEv payload (unused).
        """
        logger.info("WhatsApp connected.")
        self._connected = True

        if self._loop is not None and self._connected_event is not None:
            self._loop.call_soon_threadsafe(self._connected_event.set)
            asyncio.run_coroutine_threadsafe(self._flush_outgoing_queue(), self._loop)
            asyncio.run_coroutine_threadsafe(self.sync_group_metadata(), self._loop)

    def _handle_disconnected(self, client: Any, event: Any) -> None:
        """Neonize DisconnectedEv callback.

        Marks the channel as disconnected so that subsequent send calls
        are queued rather than dropped.

        Args:
            client: The neonize client instance.
            event: The DisconnectedEv payload (unused).
        """
        logger.warning("WhatsApp disconnected.")
        self._connected = False

    def _handle_message(self, client: Any, event: Any) -> None:
        """Neonize MessageEv callback.

        Extracts the message text and metadata and dispatches to the
        ``on_message`` callback via the event loop.

        Args:
            client: The neonize client instance.
            event: The MessageEv payload containing message info.
        """
        if self._loop is None:
            return

        try:
            info = event.Info
            # Ignore our own outgoing messages.
            if info.IsFromMe:
                return

            message = event.Message
            text = ""
            if message.HasField("conversation"):
                text = message.conversation
            elif message.HasField("extendedTextMessage"):
                text = message.extendedTextMessage.text
            if not text:
                return

            from datetime import datetime

            ts = datetime.fromtimestamp(info.Timestamp, tz=UTC).isoformat()

            chat_jid = info.MessageSource.Chat.String()
            sender_jid = info.MessageSource.Sender.String()
            push_name = getattr(info, "PushName", "") or ""

            new_msg = NewMessage(
                id=info.ID,
                chat_jid=chat_jid,
                sender=sender_jid,
                sender_name=push_name,
                content=text,
                timestamp=ts,
                is_from_me=False,
                is_bot_message=False,
            )

            asyncio.run_coroutine_threadsafe(
                self._dispatch_message(chat_jid, new_msg),
                self._loop,
            )
        except Exception as exc:
            logger.error("Error processing inbound message: %s", exc)

    async def _dispatch_message(self, chat_jid: str, msg: NewMessage) -> None:
        """Invoke the on_message callback and record chat metadata.

        Args:
            chat_jid: JID of the originating chat.
            msg: Parsed inbound message.
        """
        try:
            self._on_chat_metadata(chat_jid, msg.timestamp)
            self._on_message(chat_jid, msg)
        except Exception as exc:
            logger.error("on_message callback raised: %s", exc)


# end src/nanoclaw/channels/whatsapp.py
