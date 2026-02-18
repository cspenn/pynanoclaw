"""Abstract base class for NanoClaw message channels.

Replaces the Channel interface from src/types.ts.
All channel implementations (WhatsApp, Telegram, etc.) must inherit from Channel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from nanoclaw.types import NewMessage

# Callback type for inbound messages delivered by a channel.
# Called with (chat_jid, message) when a new message is received.
OnInboundMessage = Callable[[str, NewMessage], None]

# Callback for chat metadata discovery.
# Called with (chat_jid, timestamp, name?) when metadata is available.
# name is optional — channels that deliver names inline (Telegram) pass it here;
# channels that sync names separately (WhatsApp syncGroupMetadata) omit it.
OnChatMetadata = Callable[[str, str], None]


class Channel(ABC):
    """Abstract base class for NanoClaw message channels.

    Each channel implementation handles one messaging platform (WhatsApp,
    Telegram, etc.). Channels are responsible for connecting, receiving
    inbound messages, and sending outbound messages.

    Subclasses must implement all abstract methods. Optional capabilities
    (like typing indicators) are implemented by overriding the default stubs.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable channel identifier (e.g., 'whatsapp', 'telegram').

        Returns:
            The channel name string.
        """

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the messaging platform.

        Should block until the first successful connection is established.
        Implementations should handle reconnection internally.

        Raises:
            ConnectionError: If the connection cannot be established.
        """

    @abstractmethod
    async def send_message(self, jid: str, text: str) -> None:
        """Send a text message to the specified JID.

        Args:
            jid: The destination JID (group or individual).
            text: The message text to send.

        Raises:
            RuntimeError: If not connected and queueing is not supported.
        """

    @abstractmethod
    def is_connected(self) -> bool:
        """Check whether the channel is currently connected.

        Returns:
            True if connected and able to send/receive messages.
        """

    @abstractmethod
    def owns_jid(self, jid: str) -> bool:
        """Check whether this channel owns (can route messages for) a JID.

        Used by the router to find the correct channel for a given JID.

        Args:
            jid: The JID to check.

        Returns:
            True if this channel should handle messages for this JID.
        """

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the messaging platform gracefully.

        Called during shutdown. Should not raise exceptions.
        """

    async def set_typing(self, jid: str, is_typing: bool) -> None:
        """Send typing indicator for the specified JID.

        Optional capability. Channels that support it should override this method.
        Default implementation is a no-op.

        Args:
            jid: The JID to send the typing indicator to.
            is_typing: True to show composing, False to show paused.
        """
        return

    async def sync_group_metadata(self, force: bool = False) -> None:
        """Sync group name/metadata from the platform.

        Optional capability. Called on startup and periodically.
        Default implementation is a no-op.

        Args:
            force: If True, bypass the cache and force a sync.
        """
        return
