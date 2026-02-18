# start tests/test_queue.py
"""Tests for nanoclaw.queue.GroupQueue.

Replaces src/group-queue.test.ts from TypeScript.
Uses pytest-asyncio for async test support.
"""

from __future__ import annotations

import asyncio

import pytest

from nanoclaw.queue import GroupQueue


@pytest.fixture
def queue(tmp_path) -> GroupQueue:
    """Create a fresh GroupQueue with a tmp data dir."""
    return GroupQueue(max_concurrent=2, data_dir=str(tmp_path))


class TestEnqueueMessageCheck:
    """Tests for the enqueue_message_check method."""

    @pytest.mark.asyncio
    async def test_calls_process_messages_fn(self, queue: GroupQueue) -> None:
        """enqueue_message_check triggers the process_messages_fn."""
        called_for: list[str] = []

        async def fake_fn(jid: str) -> bool:
            called_for.append(jid)
            return True

        queue.set_process_messages_fn(fake_fn)
        queue.enqueue_message_check("group1@g.us")
        await asyncio.sleep(0.05)
        assert "group1@g.us" in called_for

    @pytest.mark.asyncio
    async def test_queues_when_active(self, queue: GroupQueue) -> None:
        """Messages are queued when group container is already active."""
        ready = asyncio.Event()
        release = asyncio.Event()

        async def slow_fn(jid: str) -> bool:
            ready.set()
            await release.wait()
            return True

        queue.set_process_messages_fn(slow_fn)
        queue.enqueue_message_check("group1@g.us")
        await ready.wait()  # Fn is now running

        queue.enqueue_message_check("group1@g.us")
        state = queue._get_group("group1@g.us")
        assert state.pending_messages is True
        release.set()
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self, queue: GroupQueue) -> None:
        """Queue respects max_concurrent limit."""
        active: list[str] = []
        max_seen = 0
        release = asyncio.Event()

        async def counting_fn(jid: str) -> bool:
            nonlocal max_seen
            active.append(jid)
            max_seen = max(max_seen, len(active))
            await release.wait()
            active.remove(jid)
            return True

        queue.set_process_messages_fn(counting_fn)
        for i in range(4):
            queue.enqueue_message_check(f"group{i}@g.us")
        await asyncio.sleep(0.05)
        assert queue._active_count <= 2
        release.set()
        await asyncio.sleep(0.1)


class TestEnqueueTask:
    """Tests for the enqueue_task method."""

    @pytest.mark.asyncio
    async def test_task_runs(self, queue: GroupQueue) -> None:
        """Enqueued tasks are executed."""
        ran: list[str] = []

        async def task_fn() -> None:
            ran.append("done")

        queue.enqueue_task("group1@g.us", "task-1", task_fn)
        await asyncio.sleep(0.05)
        assert "done" in ran

    @pytest.mark.asyncio
    async def test_no_duplicate_tasks(self, queue: GroupQueue) -> None:
        """Same task_id is not queued twice."""
        ran_count = 0
        active_lock = asyncio.Event()
        release = asyncio.Event()

        async def slow_fn(jid: str) -> bool:
            active_lock.set()
            await release.wait()
            return True

        async def task_fn() -> None:
            nonlocal ran_count
            ran_count += 1

        queue.set_process_messages_fn(slow_fn)
        queue.enqueue_message_check("group1@g.us")
        await active_lock.wait()

        # While group is active, try to enqueue same task twice
        queue.enqueue_task("group1@g.us", "task-abc", task_fn)
        queue.enqueue_task("group1@g.us", "task-abc", task_fn)

        state = queue._get_group("group1@g.us")
        assert len([t for t in state.pending_tasks if t.task_id == "task-abc"]) == 1
        release.set()


class TestShutdown:
    """Tests for the shutdown method."""

    @pytest.mark.asyncio
    async def test_shutdown_stops_new_enqueues(self, queue: GroupQueue) -> None:
        """After shutdown, enqueue_message_check is a no-op."""
        await queue.shutdown()
        called: list[str] = []

        async def fn(jid: str) -> bool:
            called.append(jid)
            return True

        queue.set_process_messages_fn(fn)
        queue.enqueue_message_check("group1@g.us")
        await asyncio.sleep(0.05)
        assert called == []
# end tests/test_queue.py
