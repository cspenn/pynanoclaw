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


# ---------------------------------------------------------------------------
# enqueue_task when shutting down (line 153)
# ---------------------------------------------------------------------------


class TestEnqueueTaskShuttingDown:
    """Tests for enqueue_task early-return when _shutting_down is True."""

    @pytest.mark.asyncio
    async def test_enqueue_task_noop_when_shutting_down(self, queue: GroupQueue) -> None:
        """enqueue_task is a no-op after shutdown is signalled (line 153)."""
        await queue.shutdown()
        ran: list[str] = []

        async def task_fn() -> None:
            ran.append("ran")

        queue.enqueue_task("group1@g.us", "task-x", task_fn)
        await asyncio.sleep(0.05)
        assert ran == []


# ---------------------------------------------------------------------------
# register_container (lines 183-186)
# ---------------------------------------------------------------------------


class TestRegisterContainer:
    """Tests for the register_container method."""

    def test_register_container_sets_container_name(self, queue: GroupQueue) -> None:
        """register_container stores the container name on the group state (line 184)."""
        queue.register_container("group@g.us", "nanoclaw-abc-123")
        state = queue._get_group("group@g.us")
        assert state.container_name == "nanoclaw-abc-123"

    def test_register_container_sets_group_folder_when_provided(
        self, queue: GroupQueue
    ) -> None:
        """register_container stores group_folder when it is supplied (lines 185-186)."""
        queue.register_container("group@g.us", "nanoclaw-abc-123", group_folder="my-group")
        state = queue._get_group("group@g.us")
        assert state.group_folder == "my-group"

    def test_register_container_does_not_overwrite_group_folder_when_none(
        self, queue: GroupQueue
    ) -> None:
        """register_container leaves group_folder unchanged when not provided (lines 185-186)."""
        queue.register_container("group@g.us", "c1", group_folder="original")
        queue.register_container("group@g.us", "c2")
        state = queue._get_group("group@g.us")
        assert state.group_folder == "original"


# ---------------------------------------------------------------------------
# send_message (lines 198-213)
# ---------------------------------------------------------------------------


class TestSendMessage:
    """Tests for the send_message method."""

    def test_send_message_returns_false_when_not_active(
        self, queue: GroupQueue
    ) -> None:
        """send_message returns False when group is not active (line 200)."""
        result = queue.send_message("group@g.us", "hello")
        assert result is False

    def test_send_message_returns_false_when_no_group_folder(
        self, queue: GroupQueue
    ) -> None:
        """send_message returns False when group_folder is not set (line 200)."""
        state = queue._get_group("group@g.us")
        state.active = True
        state.group_folder = None
        result = queue.send_message("group@g.us", "hello")
        assert result is False

    def test_send_message_writes_file_and_returns_true(
        self, queue: GroupQueue, tmp_path
    ) -> None:
        """send_message writes a JSON file to the IPC input dir and returns True (lines 201-211)."""
        state = queue._get_group("group@g.us")
        state.active = True
        state.group_folder = "my-group"

        result = queue.send_message("group@g.us", "test message")
        assert result is True

        input_dir = tmp_path / "ipc" / "my-group" / "input"
        files = list(input_dir.glob("*.json"))
        assert len(files) == 1
        import json
        data = json.loads(files[0].read_text())
        assert data["type"] == "message"
        assert data["text"] == "test message"

    def test_send_message_returns_false_on_oserror(
        self, queue: GroupQueue, tmp_path
    ) -> None:
        """send_message returns False on OSError (lines 212-213)."""
        state = queue._get_group("group@g.us")
        state.active = True
        state.group_folder = "my-group"

        from unittest.mock import patch as mpatch

        with mpatch("pathlib.Path.write_text", side_effect=OSError("disk full")):
            result = queue.send_message("group@g.us", "oops")
        assert result is False


# ---------------------------------------------------------------------------
# close_stdin (lines 221-229)
# ---------------------------------------------------------------------------


class TestCloseStdin:
    """Tests for the close_stdin method."""

    def test_close_stdin_noop_when_not_active(self, queue: GroupQueue) -> None:
        """close_stdin does nothing when group is not active (line 222-223)."""
        queue.close_stdin("group@g.us")  # Must not raise

    def test_close_stdin_writes_sentinel_file(
        self, queue: GroupQueue, tmp_path
    ) -> None:
        """close_stdin writes a _close sentinel to the IPC input dir (lines 224-227)."""
        state = queue._get_group("group@g.us")
        state.active = True
        state.group_folder = "my-group"

        queue.close_stdin("group@g.us")

        close_file = tmp_path / "ipc" / "my-group" / "input" / "_close"
        assert close_file.exists()

    def test_close_stdin_swallows_oserror(
        self, queue: GroupQueue
    ) -> None:
        """close_stdin swallows OSError without raising (lines 228-229)."""
        state = queue._get_group("group@g.us")
        state.active = True
        state.group_folder = "my-group"

        from unittest.mock import patch as mpatch

        with mpatch("pathlib.Path.write_text", side_effect=OSError("disk full")):
            queue.close_stdin("group@g.us")  # Must not raise


# ---------------------------------------------------------------------------
# _run_for_group exception handling (lines 275-278)
# ---------------------------------------------------------------------------


class TestRunForGroupExceptionHandling:
    """Tests for exception handling in _run_for_group."""

    @pytest.mark.asyncio
    async def test_exception_in_process_fn_triggers_retry(
        self, queue: GroupQueue
    ) -> None:
        """Unhandled exception in process_messages_fn triggers retry scheduling (lines 275-278)."""
        retried: list[str] = []

        async def failing_fn(jid: str) -> bool:
            raise RuntimeError("unexpected failure")

        queue.set_process_messages_fn(failing_fn)

        # Patch _schedule_retry to record calls without actually sleeping
        async def fake_retry(jid: str, state) -> None:
            retried.append(jid)

        queue._schedule_retry = fake_retry  # type: ignore[method-assign]

        queue.enqueue_message_check("group@g.us")
        await asyncio.sleep(0.1)

        assert "group@g.us" in retried


# ---------------------------------------------------------------------------
# _run_task exception handling (lines 306-307)
# ---------------------------------------------------------------------------


class TestRunTaskExceptionHandling:
    """Tests for exception handling in _run_task."""

    @pytest.mark.asyncio
    async def test_exception_in_task_fn_is_logged_not_raised(
        self, queue: GroupQueue
    ) -> None:
        """Unhandled exception in a task function does not crash the queue (lines 306-307)."""
        errored: list[str] = []

        async def bad_task() -> None:
            raise ValueError("task error")

        queue.enqueue_task("group@g.us", "bad-task", bad_task)
        await asyncio.sleep(0.1)

        # Queue must still be functional after the exception
        ran: list[str] = []

        async def good_task() -> None:
            ran.append("ok")

        queue.enqueue_task("group@g.us", "good-task", good_task)
        await asyncio.sleep(0.1)
        assert "ok" in ran


# ---------------------------------------------------------------------------
# _schedule_retry (lines 322-339)
# ---------------------------------------------------------------------------


class TestScheduleRetry:
    """Tests for _schedule_retry exponential back-off logic."""

    @pytest.mark.asyncio
    async def test_retry_increments_retry_count(self, queue: GroupQueue) -> None:
        """_schedule_retry increments the retry counter (line 322)."""
        state = queue._get_group("group@g.us")
        state.retry_count = 0

        # Patch sleep to avoid actual delay
        from unittest.mock import patch as mpatch2

        with mpatch2("asyncio.sleep", return_value=None):
            await queue._schedule_retry("group@g.us", state)

        assert state.retry_count == 1

    @pytest.mark.asyncio
    async def test_retry_resets_after_max_retries(self, queue: GroupQueue) -> None:
        """After MAX_RETRIES retries, retry_count is reset to 0 (lines 323-329)."""
        from nanoclaw.queue import MAX_RETRIES

        state = queue._get_group("group@g.us")
        state.retry_count = MAX_RETRIES  # Already at the limit

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "asyncio.sleep", return_value=None
        ):
            await queue._schedule_retry("group@g.us", state)

        assert state.retry_count == 0

    @pytest.mark.asyncio
    async def test_retry_schedules_message_check_when_not_shutting_down(
        self, queue: GroupQueue
    ) -> None:
        """After sleeping, _schedule_retry re-enqueues via enqueue_message_check (lines 338-339)."""
        state = queue._get_group("group@g.us")
        state.retry_count = 0

        enqueued: list[str] = []
        original_enqueue = queue.enqueue_message_check

        def fake_enqueue(jid: str) -> None:
            enqueued.append(jid)

        queue.enqueue_message_check = fake_enqueue  # type: ignore[method-assign]

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "asyncio.sleep", return_value=None
        ):
            await queue._schedule_retry("group@g.us", state)

        assert "group@g.us" in enqueued

    @pytest.mark.asyncio
    async def test_retry_does_not_enqueue_when_shutting_down(
        self, queue: GroupQueue
    ) -> None:
        """_schedule_retry skips re-enqueue when shutting down (line 338)."""
        state = queue._get_group("group@g.us")
        state.retry_count = 0
        await queue.shutdown()

        enqueued: list[str] = []

        def fake_enqueue(jid: str) -> None:
            enqueued.append(jid)

        queue.enqueue_message_check = fake_enqueue  # type: ignore[method-assign]

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "asyncio.sleep", return_value=None
        ):
            await queue._schedule_retry("group@g.us", state)

        assert enqueued == []


# ---------------------------------------------------------------------------
# _drain_group when shutting down (line 348)
# ---------------------------------------------------------------------------


class TestDrainGroupShuttingDown:
    """Tests for _drain_group early-return when shutting down."""

    @pytest.mark.asyncio
    async def test_drain_group_noop_when_shutting_down(
        self, queue: GroupQueue
    ) -> None:
        """_drain_group returns immediately when shutting down (line 348)."""
        await queue.shutdown()

        state = queue._get_group("group@g.us")
        state.pending_messages = True
        state.active = False

        # Should not start any tasks
        queue._drain_group("group@g.us")
        await asyncio.sleep(0.05)

        # active_count must remain 0 — no work started
        assert queue._active_count == 0


# ---------------------------------------------------------------------------
# _drain_waiting with pending messages (lines 374-377)
# ---------------------------------------------------------------------------


class TestDrainWaiting:
    """Tests for _drain_waiting that starts work for waiting groups."""

    @pytest.mark.asyncio
    async def test_drain_waiting_starts_pending_task_for_waiting_group(
        self, queue: GroupQueue
    ) -> None:
        """_drain_waiting launches a pending task for a group at the concurrency limit (lines 374-377)."""
        ran: list[str] = []

        async def task_fn() -> None:
            ran.append("ran")

        # Set up a group that is waiting (not active, has a pending task)
        from nanoclaw.queue import GroupState, QueuedTask

        state = queue._get_group("group@g.us")
        state.active = False
        state.pending_tasks = [QueuedTask(task_id="t1", group_jid="group@g.us", fn=task_fn)]

        queue._drain_waiting()
        await asyncio.sleep(0.1)

        assert "ran" in ran

    @pytest.mark.asyncio
    async def test_drain_waiting_starts_pending_messages_for_waiting_group(
        self, queue: GroupQueue
    ) -> None:
        """_drain_waiting re-queues message processing for a waiting group (lines 378-382)."""
        processed: list[str] = []

        async def fake_fn(jid: str) -> bool:
            processed.append(jid)
            return True

        queue.set_process_messages_fn(fake_fn)

        from nanoclaw.queue import GroupState

        state = queue._get_group("group@g.us")
        state.active = False
        state.pending_messages = True

        queue._drain_waiting()
        await asyncio.sleep(0.1)

        assert "group@g.us" in processed

    @pytest.mark.asyncio
    async def test_drain_waiting_respects_max_concurrent(
        self, queue: GroupQueue
    ) -> None:
        """_drain_waiting does not exceed max_concurrent (line 369-370)."""
        release = asyncio.Event()

        async def slow_fn(jid: str) -> bool:
            await release.wait()
            return True

        queue.set_process_messages_fn(slow_fn)

        # Fill all slots
        for i in range(queue._max_concurrent):
            queue.enqueue_message_check(f"group{i}@g.us")
        await asyncio.sleep(0.05)
        assert queue._active_count == queue._max_concurrent

        # Add another waiting group
        state = queue._get_group("extra@g.us")
        state.active = False
        state.pending_messages = True

        queue._drain_waiting()
        await asyncio.sleep(0.05)

        # Should not exceed max_concurrent
        assert queue._active_count <= queue._max_concurrent
        release.set()
        await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# _run_for_group retry when process_messages_fn returns False (line 275)
# ---------------------------------------------------------------------------


class TestRunForGroupRetryOnFalse:
    """Tests that _run_for_group triggers retry when process_messages_fn returns False."""

    @pytest.mark.asyncio
    async def test_run_for_group_calls_schedule_retry_when_fn_returns_false(
        self, queue: GroupQueue
    ) -> None:
        """_run_for_group calls _schedule_retry when process_messages_fn returns False (line 275)."""
        retry_called: list[str] = []

        async def fake_retry(group_jid, state) -> None:
            retry_called.append(group_jid)

        queue._schedule_retry = fake_retry  # type: ignore[method-assign]

        async def false_fn(jid: str) -> bool:
            return False

        queue.set_process_messages_fn(false_fn)
        queue.enqueue_message_check("group@g.us")
        await asyncio.sleep(0.1)

        assert "group@g.us" in retry_called


# end tests/test_queue.py
