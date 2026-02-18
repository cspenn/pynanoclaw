# start src/nanoclaw/queue.py
"""Concurrency control for NanoClaw agent containers.

Replaces src/group-queue.ts from TypeScript.
Uses asyncio.Semaphore for concurrency limiting and asyncio.Task for background work.
Tenacity handles retry with exponential backoff.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
BASE_RETRY_S = 5.0


@dataclass
class QueuedTask:
    """A task waiting to run in the group queue.

    Attributes:
        task_id: Unique task identifier (prevents double-queuing).
        group_jid: JID of the group this task belongs to.
        fn: Async callable that performs the work.
    """

    task_id: str
    group_jid: str
    fn: Callable[[], Coroutine[Any, Any, None]]


@dataclass
class GroupState:
    """Runtime state for a single group's queue slot.

    Attributes:
        active: Whether a container is currently running for this group.
        pending_messages: Whether new messages arrived while the container was busy.
        pending_tasks: Ordered list of tasks waiting to run.
        container_name: Name of the active container, or None.
        group_folder: Filesystem folder of the active group, or None.
        retry_count: Number of consecutive failures for backoff calculation.
    """

    active: bool = False
    pending_messages: bool = False
    pending_tasks: list[QueuedTask] = field(default_factory=list)
    container_name: str | None = None
    group_folder: str | None = None
    retry_count: int = 0


class GroupQueue:
    """Per-group concurrency queue for NanoClaw agent containers.

    Limits simultaneous containers via an asyncio.Semaphore.
    Each group can have at most one container running at a time.
    Additional messages or tasks for a busy group are queued and processed
    after the active container exits.

    Args:
        max_concurrent: Maximum number of simultaneously running containers.
        data_dir: Path to the data directory for IPC file writes.
    """

    def __init__(self, max_concurrent: int = 5, data_dir: str = "data") -> None:
        """Initialize the GroupQueue."""
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._data_dir = Path(data_dir)
        self._groups: dict[str, GroupState] = {}
        self._active_count = 0
        self._shutting_down = False
        self._process_messages_fn: Callable[[str], Coroutine[Any, Any, bool]] | None = None

    def _get_group(self, group_jid: str) -> GroupState:
        """Get or create the state slot for a group.

        Args:
            group_jid: The group JID.

        Returns:
            The GroupState for this JID.
        """
        if group_jid not in self._groups:
            self._groups[group_jid] = GroupState()
        return self._groups[group_jid]

    def set_process_messages_fn(self, fn: Callable[[str], Coroutine[Any, Any, bool]]) -> None:
        """Register the callback used to process messages for a group.

        Args:
            fn: Async function that processes messages for the given JID.
                Returns True on success, False to trigger retry.
        """
        self._process_messages_fn = fn

    def enqueue_message_check(self, group_jid: str) -> None:
        """Request that a group's pending messages be processed.

        If a container is already active for the group, sets the
        pending_messages flag so processing runs after the container exits.
        If the concurrency limit is reached, the group waits for a free slot.

        Args:
            group_jid: The JID of the group with new messages.
        """
        if self._shutting_down:
            return
        state = self._get_group(group_jid)
        if state.active:
            state.pending_messages = True
            logger.debug("Container active for %s, message queued", group_jid)
            return
        if self._active_count >= self._max_concurrent:
            state.pending_messages = True
            logger.debug(
                "At concurrency limit (%d), message queued for %s",
                self._active_count,
                group_jid,
            )
            return
        # Claim the slot eagerly so subsequent synchronous calls see an updated count.
        state.active = True
        state.pending_messages = False
        self._active_count += 1
        asyncio.create_task(self._run_for_group(group_jid, reason="messages"))

    def enqueue_task(
        self,
        group_jid: str,
        task_id: str,
        fn: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        """Queue a scheduled task to run for a group.

        Prevents double-queuing if the same task_id is already pending.

        Args:
            group_jid: The JID of the group to run the task for.
            task_id: Unique identifier for deduplication.
            fn: Async function that executes the task.
        """
        if self._shutting_down:
            return
        state = self._get_group(group_jid)
        if any(t.task_id == task_id for t in state.pending_tasks):
            logger.debug("Task %s already queued, skipping", task_id)
            return
        queued = QueuedTask(task_id=task_id, group_jid=group_jid, fn=fn)
        if state.active or self._active_count >= self._max_concurrent:
            state.pending_tasks.append(queued)
            logger.debug("Container busy for %s, task %s queued", group_jid, task_id)
            return
        # Claim the slot eagerly.
        state.active = True
        self._active_count += 1
        asyncio.create_task(self._run_task(group_jid, queued))

    def register_container(
        self,
        group_jid: str,
        container_name: str,
        group_folder: str | None = None,
    ) -> None:
        """Record the active container name for a group.

        Called by the container runner immediately after spawning.

        Args:
            group_jid: The group JID.
            container_name: The Apple Container container name.
            group_folder: The group filesystem folder name.
        """
        state = self._get_group(group_jid)
        state.container_name = container_name
        if group_folder:
            state.group_folder = group_folder

    def send_message(self, group_jid: str, text: str) -> bool:
        """Write a follow-up message to the active container's IPC input dir.

        Args:
            group_jid: The group JID with an active container.
            text: Message text to pipe into the container.

        Returns:
            True if the message was written, False if no active container.
        """
        state = self._get_group(group_jid)
        if not state.active or not state.group_folder:
            return False
        input_dir = self._data_dir / "ipc" / state.group_folder / "input"
        try:
            input_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{int(time.time() * 1000)}-{id(text)}.json"
            temp_path = input_dir / f"{filename}.tmp"
            final_path = input_dir / filename
            import json

            temp_path.write_text(json.dumps({"type": "message", "text": text}))
            temp_path.rename(final_path)
            return True
        except OSError:
            return False

    def close_stdin(self, group_jid: str) -> None:
        """Write the _close sentinel to signal the container to exit.

        Args:
            group_jid: The group JID with an active container.
        """
        state = self._get_group(group_jid)
        if not state.active or not state.group_folder:
            return
        input_dir = self._data_dir / "ipc" / state.group_folder / "input"
        try:
            input_dir.mkdir(parents=True, exist_ok=True)
            (input_dir / "_close").write_text("")
        except OSError:
            pass

    async def shutdown(self, _grace_period_ms: int = 10000) -> None:
        """Signal shutdown. Active containers are detached, not killed.

        Args:
            _grace_period_ms: Grace period in ms (unused; containers self-terminate).
        """
        self._shutting_down = True
        active_containers = [
            state.container_name
            for state in self._groups.values()
            if state.active and state.container_name
        ]
        logger.info(
            "GroupQueue shutting down (active=%d, detached=%d)",
            self._active_count,
            len(active_containers),
        )

    # --- Private helpers ---

    async def _run_for_group(self, group_jid: str, reason: str) -> None:
        """Process messages for a group inside a semaphore slot.

        The slot (state.active / _active_count) is already claimed by the
        caller before this coroutine is scheduled.

        Args:
            group_jid: The group JID to process.
            reason: Logging label ('messages' or 'drain').
        """
        state = self._get_group(group_jid)
        # state.active and _active_count are already set by the caller.
        logger.debug(
            "Starting container for %s (reason=%s, active=%d)",
            group_jid,
            reason,
            self._active_count,
        )
        try:
            if self._process_messages_fn:
                success = await self._process_messages_fn(group_jid)
                if success:
                    state.retry_count = 0
                else:
                    await self._schedule_retry(group_jid, state)
        except Exception as exc:
            logger.error("Error processing messages for %s: %s", group_jid, exc)
            await self._schedule_retry(group_jid, state)
        finally:
            state.active = False
            state.container_name = None
            state.group_folder = None
            self._active_count -= 1
            self._drain_group(group_jid)

    async def _run_task(self, group_jid: str, task: QueuedTask) -> None:
        """Execute a single queued task inside a semaphore slot.

        The slot (state.active / _active_count) is already claimed by the
        caller before this coroutine is scheduled.

        Args:
            group_jid: The group JID.
            task: The task to execute.
        """
        state = self._get_group(group_jid)
        # state.active and _active_count are already set by the caller.
        logger.debug(
            "Running task %s for %s (active=%d)",
            task.task_id,
            group_jid,
            self._active_count,
        )
        try:
            await task.fn()
        except Exception as exc:
            logger.error("Error running task %s: %s", task.task_id, exc)
        finally:
            state.active = False
            state.container_name = None
            state.group_folder = None
            self._active_count -= 1
            self._drain_group(group_jid)

    async def _schedule_retry(self, group_jid: str, state: GroupState) -> None:
        """Schedule a retry with exponential backoff.

        Args:
            group_jid: The group JID to retry.
            state: The current group state.
        """
        state.retry_count += 1
        if state.retry_count > MAX_RETRIES:
            logger.error(
                "Max retries exceeded for %s, dropping (will retry on next message)",
                group_jid,
            )
            state.retry_count = 0
            return
        delay_s = BASE_RETRY_S * (2 ** (state.retry_count - 1))
        logger.info(
            "Scheduling retry for %s (attempt=%d, delay=%.1fs)",
            group_jid,
            state.retry_count,
            delay_s,
        )
        await asyncio.sleep(delay_s)
        if not self._shutting_down:
            self.enqueue_message_check(group_jid)

    def _drain_group(self, group_jid: str) -> None:
        """After a group slot frees, start pending work (tasks first).

        Args:
            group_jid: The group JID that just finished.
        """
        if self._shutting_down:
            return
        state = self._get_group(group_jid)
        if state.pending_tasks:
            task = state.pending_tasks.pop(0)
            # Claim the slot before scheduling.
            state.active = True
            self._active_count += 1
            asyncio.create_task(self._run_task(group_jid, task))
            return
        if state.pending_messages:
            state.active = True
            state.pending_messages = False
            self._active_count += 1
            asyncio.create_task(self._run_for_group(group_jid, reason="drain"))
            return
        # Nothing pending for this group — check other waiting groups
        self._drain_waiting()

    def _drain_waiting(self) -> None:
        """Start work for groups that were waiting for a concurrency slot."""
        for jid, state in self._groups.items():
            if self._active_count >= self._max_concurrent:
                break
            if state.active:
                continue
            if state.pending_tasks:
                task = state.pending_tasks.pop(0)
                state.active = True
                self._active_count += 1
                asyncio.create_task(self._run_task(jid, task))
            elif state.pending_messages:
                state.active = True
                state.pending_messages = False
                self._active_count += 1
                asyncio.create_task(self._run_for_group(jid, reason="drain"))


# end src/nanoclaw/queue.py
