# start src/nanoclaw/main.py
"""NanoClaw orchestrator entry point.

Connects WhatsApp, routes messages to Claude agent containers,
manages IPC, scheduling, and state persistence.

Usage:
    python -m nanoclaw.main
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import logging.handlers
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from nanoclaw.config import AppConfig, load_config, load_credentials
from nanoclaw.container import run_container_agent, write_groups_snapshot, write_tasks_snapshot
from nanoclaw.db.operations import (
    get_all_chats,
    get_all_registered_groups,
    get_all_sessions,
    get_all_tasks,
    get_messages_since,
    get_new_messages,
    get_router_state,
    init_database,
    set_registered_group,
    set_router_state,
    set_session,
    store_chat_metadata,
    store_message,
)
from nanoclaw.ipc import IpcDeps, start_ipc_watcher
from nanoclaw.queue import GroupQueue
from nanoclaw.router import format_messages, format_outbound
from nanoclaw.scheduler import start_scheduler_loop
from nanoclaw.types import (
    AvailableGroup,
    ContainerInput,
    ContainerOutput,
    NewMessage,
    RegisteredGroup,
)

logger = logging.getLogger(__name__)

MAIN_GROUP_FOLDER = "main"

# Router-state keys stored in the DB
_STATE_KEY_LAST_TIMESTAMP = "last_timestamp"
_STATE_KEY_LAST_AGENT_TIMESTAMPS = "last_agent_timestamps"


class NanoClawOrchestrator:
    """Main orchestrator that ties all subsystems together.

    Manages the lifecycle of all NanoClaw subsystems: database, WhatsApp
    channel, IPC watcher, scheduler loop, and the core message-processing
    loop. State (cursors, sessions, registered groups) is persisted to
    SQLite so restarts are seamless.

    Args:
        config_path: Optional path to config.yml. Defaults to config.yml
            in the current working directory.
        credentials_path: Optional path to credentials.yml. Defaults to
            credentials.yml in the current working directory.
    """

    def __init__(
        self,
        config_path: Path | None = None,
        credentials_path: Path | None = None,
    ) -> None:
        """Initialize the orchestrator with configuration."""
        self.config: AppConfig = load_config(config_path)
        self.credentials = load_credentials(credentials_path)

        # Resolve directories relative to the worktree root (three levels up
        # from src/nanoclaw/main.py → src/nanoclaw → src → project root).
        self.project_root: Path = Path(__file__).parent.parent.parent.resolve()
        self.groups_dir: Path = self.project_root / self.config.paths.groups_dir
        self.data_dir: Path = self.project_root / self.config.paths.data_dir
        self.store_dir: Path = self.project_root / self.config.paths.store_dir

        # Mutable state - persisted to DB
        self.last_timestamp: str = ""
        self.sessions: dict[str, str] = {}  # group_folder → session_id
        self.registered_groups: dict[str, RegisteredGroup] = {}  # jid → RegisteredGroup
        self.last_agent_timestamp: dict[str, str] = {}  # jid → last processed timestamp

        # Subsystems initialised in run()
        self.queue: GroupQueue | None = None
        self.channel: Any = None  # Channel subclass

        self._shutting_down = False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start all subsystems and run the main message loop.

        Sequence:
            1. Ensure Apple Container system is running.
            2. Initialise SQLite database.
            3. Load persistent state.
            4. Configure logging.
            5. Create GroupQueue.
            6. Install OS signal handlers.
            7. Connect WhatsApp channel.
            8. Start scheduler and IPC watcher background tasks.
            9. Recover pending messages from previous run.
            10. Enter the main polling loop.
        """
        self._ensure_container_system()

        # Ensure store directory exists before initialising the DB.
        self.store_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(self.store_dir / "messages.db")
        init_database(db_path)

        self._load_state()

        # Configure logging now that config is available.
        setup_logging(self.config)

        # Queue with process callback.
        self.queue = GroupQueue(
            max_concurrent=self.config.container.max_concurrent,
            data_dir=str(self.data_dir),
        )
        self.queue.set_process_messages_fn(self._process_group_messages)

        # Install OS signal handlers for graceful shutdown.
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):

            def _make_shutdown_handler(s: signal.Signals) -> None:
                asyncio.create_task(self._shutdown(s.name))

            loop.add_signal_handler(sig, _make_shutdown_handler, sig)

        # Connect messaging channel.
        await self._connect_channel()

        # Wire up IPC dependencies.
        ipc_deps = IpcDeps(
            send_message=self._ipc_send_message,
            registered_groups=lambda: self.registered_groups,
            register_group=self._register_group,
            sync_group_metadata=self._sync_group_metadata,
            get_available_groups=self._get_available_groups,
            write_groups_snapshot=self._ipc_write_groups_snapshot,
            timezone=self.config.timezone,
        )

        # Import db_ops as module object so IPC/scheduler can call it.
        import nanoclaw.db.operations as db_ops_module

        # Start background tasks.
        asyncio.create_task(
            start_scheduler_loop(
                poll_interval_s=self.config.timing.scheduler_poll_interval_s,
                idle_timeout_ms=self.config.timing.idle_timeout_ms,
                groups_dir=self.groups_dir,
                data_dir=self.data_dir,
                timezone_str=self.config.timezone,
                registered_groups_fn=lambda: self.registered_groups,
                get_sessions_fn=lambda: self.sessions,
                queue=self.queue,
                send_message_fn=self._ipc_send_message,
                db_ops=db_ops_module,
                run_agent_fn=self._run_agent,
            )
        )

        asyncio.create_task(
            start_ipc_watcher(
                data_dir=self.data_dir,
                poll_interval_s=self.config.timing.ipc_poll_interval_s,
                deps=ipc_deps,
                db_ops=db_ops_module,
            )
        )

        self._recover_pending_messages()
        await self._start_message_loop()

    # ------------------------------------------------------------------
    # Container system helpers
    # ------------------------------------------------------------------

    def _start_container_system(self) -> None:
        """Start the Apple Container system.

        Raises:
            RuntimeError: If the container system fails to start.
        """
        logger.info("Container system not running, starting it...")
        start_result = subprocess.run(
            ["container", "system", "start"],
            capture_output=True,
            timeout=30,
        )
        if start_result.returncode != 0:
            stderr = start_result.stderr.decode(errors="replace")
            msg = (
                "┌─────────────────────────────────────────────────┐\n"
                "│  ERROR: Apple Container system failed to start  │\n"
                "└─────────────────────────────────────────────────┘\n"
                f"stderr: {stderr}"
            )
            logger.error(msg)
            raise RuntimeError(f"Container system start failed: {stderr}")
        logger.info("Container system started.")

    def _stop_running_nanoclaw_containers(self, raw: str) -> None:
        """Parse container list JSON and stop any running nanoclaw-* containers.

        Args:
            raw: JSON string from ``container ls --format json``.
        """
        containers: list[dict[str, str]] | dict[str, str] = json.loads(raw)
        if isinstance(containers, dict):
            # Some versions return a single object instead of a list.
            containers = [containers]
        for ct in containers:
            name = ct.get("name", "") or ct.get("Name", "")
            status = ct.get("status", "") or ct.get("Status", "")
            if (
                isinstance(name, str)
                and name.startswith("nanoclaw-")
                and "running" in status.lower()
            ):
                logger.info("Killing orphaned container %s (status=%s)", name, status)
                subprocess.run(
                    ["container", "stop", name],
                    capture_output=True,
                    timeout=10,
                )

    def _kill_orphaned_containers(self) -> None:
        """Kill any nanoclaw-* containers left running from a previous crash."""
        try:
            ls_result = subprocess.run(
                ["container", "ls", "--format", "json"],
                capture_output=True,
                timeout=10,
            )
            if ls_result.returncode == 0:
                raw = ls_result.stdout.decode(errors="replace").strip()
                if raw:
                    self._stop_running_nanoclaw_containers(raw)
        except Exception as exc:
            logger.warning("Could not check for orphaned containers: %s", exc)

    def _ensure_container_system(self) -> None:
        """Ensure Apple Container system is running; kill orphaned containers.

        Checks container system status and starts it if needed. Also kills
        any nanoclaw-* containers left running from a previous crash.

        Raises:
            RuntimeError: If the container system cannot be started.
        """
        result = subprocess.run(
            ["container", "system", "status"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            self._start_container_system()
        self._kill_orphaned_containers()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Load persistent state from database.

        Restores last_timestamp, last_agent_timestamps, sessions, and
        registered groups from the SQLite database.
        """
        raw_ts = get_router_state(_STATE_KEY_LAST_TIMESTAMP)
        if raw_ts:
            self.last_timestamp = raw_ts

        raw_agent_ts = get_router_state(_STATE_KEY_LAST_AGENT_TIMESTAMPS)
        if raw_agent_ts:
            try:
                self.last_agent_timestamp = json.loads(raw_agent_ts)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Could not parse last_agent_timestamps from DB; resetting.")

        self.sessions = get_all_sessions()
        self.registered_groups = get_all_registered_groups()

        logger.info(
            "State loaded: %d groups, %d sessions, last_timestamp=%r",
            len(self.registered_groups),
            len(self.sessions),
            self.last_timestamp,
        )

    def _save_state(self) -> None:
        """Persist last_timestamp and last_agent_timestamp to DB."""
        set_router_state(_STATE_KEY_LAST_TIMESTAMP, self.last_timestamp)
        set_router_state(
            _STATE_KEY_LAST_AGENT_TIMESTAMPS,
            json.dumps(self.last_agent_timestamp),
        )

    # ------------------------------------------------------------------
    # Group registration
    # ------------------------------------------------------------------

    def _register_group(self, jid: str, group: RegisteredGroup) -> None:
        """Register a group in memory and DB, creating its directory.

        Args:
            jid: WhatsApp JID of the group.
            group: RegisteredGroup data to store.
        """
        group_dir = self.groups_dir / group.folder
        group_dir.mkdir(parents=True, exist_ok=True)
        self.registered_groups[jid] = group
        set_registered_group(jid, group)
        logger.info("Group registered: %s → %s", group.name, group.folder)

    def _get_available_groups(self) -> list[AvailableGroup]:
        """Get list of WhatsApp groups ordered by recent activity.

        Returns:
            List of AvailableGroup objects built from the chats table,
            ordered by most recent message time descending.
        """
        chats = get_all_chats()
        registered_jids = set(self.registered_groups.keys())
        return [
            AvailableGroup(
                jid=c.jid,
                name=c.name,
                last_activity=c.last_message_time,
                is_registered=c.jid in registered_jids,
            )
            for c in chats
            if not c.jid.startswith("__")
        ]

    # ------------------------------------------------------------------
    # Channel connection
    # ------------------------------------------------------------------

    async def _connect_channel(self) -> None:
        """Create and connect the WhatsApp channel, or fall back to no-op.

        The WhatsApp channel module may not exist in all environments
        (e.g., during CI or early development). A no-op channel is used
        in those cases so the orchestrator still starts.
        """
        try:
            from nanoclaw.channels.whatsapp import WhatsAppChannel

            def _on_message(jid: str, msg: NewMessage) -> None:
                store_message(msg)

            def _on_chat_metadata(jid: str, ts: str) -> None:
                store_chat_metadata(jid, ts)

            self.channel = WhatsAppChannel(
                config=self.config,
                db_ops=None,  # operations imported directly
                on_message=_on_message,
                on_chat_metadata=_on_chat_metadata,
                registered_groups_fn=lambda: self.registered_groups,
            )
            await self.channel.connect()
            logger.info("WhatsApp channel connected.")
        except ImportError:
            logger.warning(
                "WhatsApp channel not available - running without messaging. "
                "Install nanoclaw.channels.whatsapp to enable WhatsApp support."
            )

            from nanoclaw.channels.base import Channel

            class NoOpChannel(Channel):
                """Stub channel for environments without WhatsApp support."""

                @property
                def name(self) -> str:
                    """Return channel name."""
                    return "noop"

                async def connect(self) -> None:
                    """No-op connect."""

                async def send_message(self, jid: str, text: str) -> None:
                    """Log outbound message instead of sending."""
                    logger.info("NOOP send to %s: %s", jid, text[:100])

                def is_connected(self) -> bool:
                    """Always report connected for no-op channel."""
                    return True

                def owns_jid(self, jid: str) -> bool:
                    """Accept all JIDs in no-op mode."""
                    return True

                async def disconnect(self) -> None:
                    """No-op disconnect."""

            self.channel = NoOpChannel()

    # ------------------------------------------------------------------
    # IPC / scheduler callbacks
    # ------------------------------------------------------------------

    async def _ipc_send_message(self, jid: str, text: str) -> None:
        """Send a formatted outbound message via the active channel.

        Used as the send_message callback by the IPC watcher and scheduler.

        Args:
            jid: Destination WhatsApp JID.
            text: Raw agent output text (will be passed through format_outbound).
        """
        clean = format_outbound(text)
        if clean and self.channel:
            await self.channel.send_message(jid, clean)

    async def _sync_group_metadata(self, force: bool = False) -> None:
        """Delegate group metadata sync to the active channel.

        Args:
            force: If True, bypass channel-level caching.
        """
        if self.channel:
            await self.channel.sync_group_metadata(force)

    def _ipc_write_groups_snapshot(
        self,
        group_folder: str,
        is_main: bool,
        groups: list[AvailableGroup],
        registered_jids: set[str],
    ) -> None:
        """Write a groups snapshot file for the container to read.

        Wraps container.write_groups_snapshot, injecting data_dir.

        Args:
            group_folder: The source group's folder name.
            is_main: Whether the source group is the main group.
            groups: Available WhatsApp groups.
            registered_jids: JIDs already registered with NanoClaw.
        """
        write_groups_snapshot(
            group_folder=group_folder,
            is_main=is_main,
            groups=groups,
            registered_jids=registered_jids,
            data_dir=self.data_dir,
        )

    # ------------------------------------------------------------------
    # Startup recovery
    # ------------------------------------------------------------------

    def _recover_pending_messages(self) -> None:
        """On startup, re-queue groups with unprocessed messages.

        If NanoClaw crashed while a group had pending messages, those
        messages will be picked up here so no messages are silently dropped.
        """
        if not self.registered_groups:
            return

        jids = list(self.registered_groups.keys())
        for jid in jids:
            since = self.last_agent_timestamp.get(jid, "")
            missed = get_messages_since(jid, since, self.config.assistant.name)
            if missed:
                logger.info("Recovering %d pending message(s) for %s", len(missed), jid)
                if self.queue:
                    self.queue.enqueue_message_check(jid)

    # ------------------------------------------------------------------
    # Main message loop
    # ------------------------------------------------------------------

    def _dispatch_chat_to_queue(self, chat_jid: str, group_msgs: list[NewMessage]) -> None:
        """Check trigger and pipe or enqueue messages for one chat group.

        Fetches all pending messages since the last agent run, then either
        pipes them to an active container or enqueues a new one.

        Args:
            chat_jid: JID of the group to dispatch.
            group_msgs: New messages just arrived for this group.
        """
        group = self.registered_groups.get(chat_jid)
        if not group:
            return

        is_main = group.folder == MAIN_GROUP_FOLDER
        needs_trigger = not is_main and group.requires_trigger is not False

        if needs_trigger:
            trigger_pattern = self.config.trigger_pattern
            has_trigger = any(trigger_pattern.search(m.content.strip()) for m in group_msgs)
            if not has_trigger:
                # Message stored in DB; wait for a trigger mention.
                return

        # Gather all pending messages since the last agent run.
        all_pending = get_messages_since(
            chat_jid,
            self.last_agent_timestamp.get(chat_jid, ""),
            self.config.assistant.name,
        )
        messages_to_send = all_pending if all_pending else group_msgs
        formatted = format_messages(messages_to_send)

        assert self.queue is not None
        if self.queue.send_message(chat_jid, formatted):
            # Successfully piped to the active container.
            self.last_agent_timestamp[chat_jid] = messages_to_send[-1].timestamp
            self._save_state()
            if self.channel:
                asyncio.create_task(self.channel.set_typing(chat_jid, True))
        else:
            # No active container — enqueue a new one.
            self.queue.enqueue_message_check(chat_jid)

    async def _start_message_loop(self) -> None:
        """Poll for new messages and route them to agent containers.

        Runs continuously until shutdown. Every poll_interval_s seconds:
          1. Fetch all messages newer than last_timestamp.
          2. Group by chat JID.
          3. For each group, check trigger requirements.
          4. Pipe to active container or enqueue for a new one.
        """
        logger.info("NanoClaw running (trigger: @%s)", self.config.assistant.name)

        while not self._shutting_down:
            try:
                jids = list(self.registered_groups.keys())
                messages, new_timestamp = get_new_messages(
                    jids, self.last_timestamp, self.config.assistant.name
                )

                if messages:
                    self.last_timestamp = new_timestamp
                    self._save_state()

                    # Group messages by chat JID.
                    by_group: dict[str, list[NewMessage]] = {}
                    for msg in messages:
                        by_group.setdefault(msg.chat_jid, []).append(msg)

                    for chat_jid, group_msgs in by_group.items():
                        self._dispatch_chat_to_queue(chat_jid, group_msgs)

            except Exception as exc:
                logger.error("Error in message loop: %s", exc, exc_info=True)

            await asyncio.sleep(self.config.timing.poll_interval_s)

    # ------------------------------------------------------------------
    # Group message processor (called by GroupQueue)
    # ------------------------------------------------------------------

    def _fetch_pending_messages(
        self, chat_jid: str, group: RegisteredGroup, is_main: bool
    ) -> list[NewMessage] | None:
        """Fetch messages to process and check trigger requirements.

        Args:
            chat_jid: JID of the group.
            group: The registered group record.
            is_main: Whether this is the main group.

        Returns:
            List of messages to process, or None if nothing should run.
        """
        since = self.last_agent_timestamp.get(chat_jid, "")
        missed = get_messages_since(chat_jid, since, self.config.assistant.name)
        if not missed:
            return None
        if not is_main and group.requires_trigger is not False:
            trigger_pattern = self.config.trigger_pattern
            if not any(trigger_pattern.search(m.content.strip()) for m in missed):
                return None
        return missed

    def _finalize_group_run(
        self,
        chat_jid: str,
        status: str,
        had_error: bool,
        output_sent: bool,
        prev_cursor: str,
    ) -> bool:
        """Handle post-run cursor state based on success or error.

        Args:
            chat_jid: JID of the group.
            status: Agent run status string.
            had_error: Whether an error frame was received during streaming.
            output_sent: Whether any output was sent to the user.
            prev_cursor: Cursor value to restore on failure with no output.

        Returns:
            True to signal success; False to request retry with backoff.
        """
        if status == "error" or had_error:
            if output_sent:
                logger.warning("Agent error after output sent for %s; keeping cursor", chat_jid)
                return True
            # Roll back the cursor so the messages will be retried.
            self.last_agent_timestamp[chat_jid] = prev_cursor
            self._save_state()
            return False
        return True

    async def _process_group_messages(self, chat_jid: str) -> bool:
        """Process all pending messages for a group.

        Called by the GroupQueue when it is this group's turn to run.
        Fetches missed messages, formats them, runs the agent container,
        streams output back to WhatsApp, and updates the cursor.

        Args:
            chat_jid: JID of the group to process.

        Returns:
            True on success or if there is nothing to do.
            False to signal the queue to retry with backoff.
        """
        group = self.registered_groups.get(chat_jid)
        if not group:
            return True

        is_main = group.folder == MAIN_GROUP_FOLDER
        missed = self._fetch_pending_messages(chat_jid, group, is_main)
        if missed is None:
            return True

        prompt = format_messages(missed)

        # Advance cursor; will be rolled back on error if no output was sent.
        prev_cursor = self.last_agent_timestamp.get(chat_jid, "")
        self.last_agent_timestamp[chat_jid] = missed[-1].timestamp
        self._save_state()

        # Idle timer: close container stdin after idle_timeout_s of no output.
        idle_task: asyncio.Task[None] | None = None

        async def reset_idle_timer() -> None:
            """(Re)start the idle close timer."""
            nonlocal idle_task
            if idle_task and not idle_task.done():
                idle_task.cancel()
            idle_task = asyncio.create_task(self._idle_close(chat_jid))

        if self.channel:
            await self.channel.set_typing(chat_jid, True)

        had_error = False
        output_sent = False

        async def on_output(result: ContainerOutput) -> None:
            """Handle each streamed output frame from the container.

            Args:
                result: Parsed container output frame.
            """
            nonlocal had_error, output_sent
            if result.result:
                text = format_outbound(result.result)
                if text and self.channel:
                    await self.channel.send_message(chat_jid, text)
                    output_sent = True
                await reset_idle_timer()
            if result.status == "error":
                had_error = True

        status = await self._run_agent(group, prompt, chat_jid, on_output=on_output)

        if self.channel:
            await self.channel.set_typing(chat_jid, False)
        if idle_task and not idle_task.done():
            idle_task.cancel()

        return self._finalize_group_run(chat_jid, status, had_error, output_sent, prev_cursor)

    async def _idle_close(self, chat_jid: str) -> None:
        """Close container stdin after idle timeout.

        Args:
            chat_jid: JID of the group whose container should be closed.
        """
        await asyncio.sleep(self.config.timing.idle_timeout_s)
        logger.debug("Idle timeout for %s, closing container stdin", chat_jid)
        if self.queue:
            self.queue.close_stdin(chat_jid)

    # ------------------------------------------------------------------
    # Agent runner
    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        group: RegisteredGroup,
        prompt: str,
        chat_jid: str,
        session_id: str | None = None,
        is_scheduled_task: bool = False,
        on_output: Any = None,
        on_streaming_output: Any = None,
    ) -> str:
        """Run an agent container for a group.

        Writes tasks and groups snapshots, then delegates to
        run_container_agent(). Tracks the new session ID on success.

        Args:
            group: The registered group to run the agent for.
            prompt: Formatted prompt string for the agent.
            chat_jid: JID of the originating chat (for session tracking).
            session_id: Optional session ID to resume. Defaults to the
                stored session for this group.
            is_scheduled_task: Whether this is a scheduled (not user-triggered) run.
            on_output: Optional async callback for each ContainerOutput frame
                (used by _process_group_messages).
            on_streaming_output: Optional async callback with signature
                (result: str, had_error: bool) (used by scheduler).

        Returns:
            'success' or 'error'.
        """
        is_main = group.folder == MAIN_GROUP_FOLDER

        # Resolve session ID.
        if session_id is None:
            session_id = self.sessions.get(group.folder)

        # Write pre-run snapshots so containers see current state.
        try:
            all_tasks = get_all_tasks()
            write_tasks_snapshot(
                group_folder=group.folder,
                is_main=is_main,
                tasks=all_tasks,
                data_dir=self.data_dir,
            )
        except Exception as exc:
            logger.warning("Failed to write tasks snapshot for %s: %s", group.folder, exc)

        try:
            available = self._get_available_groups()
            registered_jids = set(self.registered_groups.keys())
            write_groups_snapshot(
                group_folder=group.folder,
                is_main=is_main,
                groups=available,
                registered_jids=registered_jids,
                data_dir=self.data_dir,
            )
        except Exception as exc:
            logger.warning("Failed to write groups snapshot for %s: %s", group.folder, exc)

        container_input = ContainerInput(
            prompt=prompt,
            session_id=session_id,
            group_folder=group.folder,
            chat_jid=chat_jid,
            is_main=is_main,
            is_scheduled_task=is_scheduled_task,
        )

        # Build an on_output callback that also handles on_streaming_output.
        effective_on_output = on_output

        if on_streaming_output is not None and on_output is None:
            # Scheduler-style callback: (result_str, had_error) → None
            async def _scheduler_on_output(result: ContainerOutput) -> None:
                """Adapt ContainerOutput to the scheduler's callback signature.

                Args:
                    result: Parsed container output frame.
                """
                await on_streaming_output(
                    result.result or "",
                    result.status == "error",
                )

            effective_on_output = _scheduler_on_output

        # Track the container name so the queue can pipe follow-up messages.
        assert self.queue is not None
        _queue = self.queue

        async def on_container_name(name: str) -> None:
            """Register container name with the queue for message piping.

            Args:
                name: Apple Container container name.
            """
            _queue.register_container(chat_jid, name, group.folder)

        try:
            output = await run_container_agent(
                group=group,
                container_input=container_input,
                groups_dir=self.groups_dir,
                data_dir=self.data_dir,
                project_root=self.project_root,
                image=self.config.container.image,
                timeout_ms=self.config.container.timeout_ms,
                idle_timeout_ms=self.config.timing.idle_timeout_ms,
                max_output_size=self.config.container.max_output_size_bytes,
                on_container_name=on_container_name,
                on_output=effective_on_output,
            )
        except Exception as exc:
            logger.error(
                "Container run raised exception for %s: %s",
                group.folder,
                exc,
                exc_info=True,
            )
            return "error"

        # Persist updated session ID.
        if output.new_session_id:
            self.sessions[group.folder] = output.new_session_id
            try:
                set_session(group.folder, output.new_session_id)
            except Exception as exc:
                logger.warning("Failed to persist session ID for %s: %s", group.folder, exc)

        return output.status

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    async def _shutdown(self, sig_name: str) -> None:
        """Handle OS signal for graceful shutdown.

        Args:
            sig_name: Name of the signal that triggered shutdown (e.g. 'SIGTERM').
        """
        if self._shutting_down:
            return
        self._shutting_down = True
        logger.info("NanoClaw shutting down (signal=%s)...", sig_name)

        self._save_state()

        if self.queue:
            await self.queue.shutdown()

        if self.channel:
            try:
                await self.channel.disconnect()
            except Exception as exc:
                logger.warning("Channel disconnect error: %s", exc)

        logger.info("NanoClaw shutdown complete.")

        # Cancel remaining tasks and stop the event loop.
        loop = asyncio.get_event_loop()
        tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()


# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------


def setup_logging(config: AppConfig) -> None:
    """Configure stdlib logging from application config.

    Sets the root logger level, attaches a StreamHandler (stderr), and
    optionally attaches a RotatingFileHandler if config.logging.file is set.

    Args:
        config: Loaded application configuration.
    """
    log_level = getattr(logging, config.logging.level, logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any handlers already attached (e.g., from basicConfig).
    root_logger.handlers.clear()

    # Console handler (stderr).
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Optional rotating file handler.
    if config.logging.file:
        log_file = Path(config.logging.file)
        if not log_file.is_absolute():
            # Resolve relative to current working directory.
            log_file = Path.cwd() / log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                str(log_file),
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logger.debug("Log file: %s", log_file)
        except OSError as exc:
            logger.warning("Could not open log file %s: %s", log_file, exc)


# ------------------------------------------------------------------
# Entry points
# ------------------------------------------------------------------


async def main_async() -> None:
    """Async entry point for the orchestrator."""
    orchestrator = NanoClawOrchestrator()
    await orchestrator.run()


def main() -> None:
    """Synchronous entry point for ``python -m nanoclaw.main``."""
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main_async())


if __name__ == "__main__":
    main()
# end src/nanoclaw/main.py
