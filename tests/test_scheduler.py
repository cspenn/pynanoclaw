# start tests/test_scheduler.py
"""Tests for nanoclaw.scheduler module.

Covers the scheduled task runner: polling loop, task context resolution,
next-run calculation, error logging, and end-to-end task execution.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import nanoclaw.scheduler as sched_mod
from nanoclaw.scheduler import (
    _calculate_next_run,
    _check_due_tasks,
    _log_error,
    _log_task_completion,
    _resolve_task_context,
    _run_task,
    start_scheduler_loop,
)
from nanoclaw.types import RegisteredGroup, ScheduledTask, TaskRunLog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_scheduled_task(**kwargs: object) -> ScheduledTask:
    """Build a ScheduledTask with sensible defaults, overridable via kwargs."""
    defaults: dict[str, object] = {
        "id": "task-1",
        "group_folder": "test-group",
        "chat_jid": "group@g.us",
        "prompt": "Do something",
        "schedule_type": "interval",
        "schedule_value": "60000",
        "context_mode": "isolated",
        "next_run": "2024-01-01T00:00:00Z",
        "status": "active",
        "created_at": "2024-01-01T00:00:00Z",
    }
    defaults.update(kwargs)
    return ScheduledTask(**defaults)  # type: ignore[arg-type]


def make_registered_group(folder: str = "test-group") -> RegisteredGroup:
    """Build a RegisteredGroup pointing at the given folder."""
    return RegisteredGroup(
        name="Test Group",
        folder=folder,
        trigger="@Andy",
        added_at="2024-01-01T00:00:00Z",
    )


def make_db_ops() -> MagicMock:
    """Return a mock db_ops with common defaults configured."""
    db_ops = MagicMock()
    db_ops.get_due_tasks.return_value = []
    db_ops.get_task_by_id.return_value = None
    db_ops.log_task_run = MagicMock()
    db_ops.update_task_after_run = MagicMock()
    return db_ops


# ---------------------------------------------------------------------------
# Fixture: reset the module-level _scheduler_running flag before every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_scheduler_running() -> object:
    """Ensure _scheduler_running is False before and after each test."""
    sched_mod._scheduler_running = False
    yield
    sched_mod._scheduler_running = False


# ---------------------------------------------------------------------------
# TestCalculateNextRun
# ---------------------------------------------------------------------------


class TestCalculateNextRun:
    """Tests for _calculate_next_run."""

    def test_cron_returns_future_timestamp(self) -> None:
        """A valid cron expression yields a future ISO timestamp string."""
        result = _calculate_next_run("cron", "*/5 * * * *", "UTC")
        assert result is not None
        # Must be a non-empty string parseable as ISO
        from datetime import datetime, UTC as _UTC

        dt = datetime.fromisoformat(result)
        assert dt > datetime.now(_UTC)

    def test_cron_invalid_expression_returns_none(self) -> None:
        """An invalid cron expression returns None."""
        result = _calculate_next_run("cron", "not a cron", "UTC")
        assert result is None

    def test_interval_returns_future_timestamp(self) -> None:
        """A numeric interval in ms produces a future ISO timestamp."""
        result = _calculate_next_run("interval", "60000", "UTC")
        assert result is not None
        from datetime import datetime, UTC as _UTC

        dt = datetime.fromisoformat(result)
        assert dt > datetime.now(_UTC)

    def test_interval_invalid_value_returns_none(self) -> None:
        """A non-numeric interval value returns None."""
        result = _calculate_next_run("interval", "not-a-number", "UTC")
        assert result is None

    def test_interval_valid_large_number(self) -> None:
        """A valid large interval value (e.g. 0 ms) produces a timestamp."""
        # scheduler.py does NOT guard against zero; it just computes
        # datetime.fromtimestamp(time.time() + 0) which is valid.
        result = _calculate_next_run("interval", "0", "UTC")
        assert result is not None

    def test_once_returns_none(self) -> None:
        """schedule_type='once' returns None (no recurring run)."""
        result = _calculate_next_run("once", "2024-01-01T00:00:00Z", "UTC")
        assert result is None

    def test_unknown_schedule_type_returns_none(self) -> None:
        """An unrecognised schedule_type returns None."""
        result = _calculate_next_run("daily", "09:00", "UTC")
        assert result is None


# ---------------------------------------------------------------------------
# TestLogError
# ---------------------------------------------------------------------------


class TestLogError:
    """Tests for _log_error."""

    def test_log_error_calls_db_log_task_run(self) -> None:
        """_log_error writes a TaskRunLog with status='error' to the DB."""
        db_ops = make_db_ops()

        _log_error("task-99", 250, "Something went wrong", db_ops)

        db_ops.log_task_run.assert_called_once()
        log: TaskRunLog = db_ops.log_task_run.call_args[0][0]
        assert log.task_id == "task-99"
        assert log.duration_ms == 250
        assert log.status == "error"
        assert log.error == "Something went wrong"
        assert log.result is None

    def test_log_error_run_at_is_iso_string(self) -> None:
        """The run_at field in the logged record is a valid ISO timestamp."""
        from datetime import datetime, UTC as _UTC

        db_ops = make_db_ops()
        _log_error("task-1", 0, "err", db_ops)
        log: TaskRunLog = db_ops.log_task_run.call_args[0][0]
        # Should parse without raising
        datetime.fromisoformat(log.run_at)


# ---------------------------------------------------------------------------
# TestResolveTaskContext
# ---------------------------------------------------------------------------


class TestResolveTaskContext:
    """Tests for _resolve_task_context."""

    def test_resolves_group_and_none_session_for_isolated_context(self) -> None:
        """context_mode='isolated' returns the group with session_id=None."""
        task = make_scheduled_task(context_mode="isolated")
        group = make_registered_group("test-group")
        db_ops = make_db_ops()

        result = _resolve_task_context(
            task=task,
            start_time=0.0,
            registered_groups_fn=lambda: {"group@g.us": group},
            get_sessions_fn=lambda: {"test-group": "sess-123"},
            db_ops=db_ops,
        )

        assert result is not None
        resolved_group, session_id = result
        assert resolved_group is group
        assert session_id is None

    def test_resolves_group_and_session_for_group_context(self) -> None:
        """context_mode='group' returns the group with the matching session_id."""
        task = make_scheduled_task(context_mode="group", group_folder="test-group")
        group = make_registered_group("test-group")
        db_ops = make_db_ops()

        result = _resolve_task_context(
            task=task,
            start_time=0.0,
            registered_groups_fn=lambda: {"group@g.us": group},
            get_sessions_fn=lambda: {"test-group": "sess-abc"},
            db_ops=db_ops,
        )

        assert result is not None
        resolved_group, session_id = result
        assert resolved_group is group
        assert session_id == "sess-abc"

    def test_unknown_group_folder_returns_none(self) -> None:
        """When the task's group_folder does not match any group, return None."""
        task = make_scheduled_task(group_folder="nonexistent-group")
        db_ops = make_db_ops()

        result = _resolve_task_context(
            task=task,
            start_time=0.0,
            registered_groups_fn=lambda: {},
            get_sessions_fn=lambda: {},
            db_ops=db_ops,
        )

        assert result is None
        # Error should be logged to DB
        db_ops.log_task_run.assert_called_once()
        log: TaskRunLog = db_ops.log_task_run.call_args[0][0]
        assert log.status == "error"
        assert "nonexistent-group" in (log.error or "")

    def test_session_not_found_returns_none_session(self) -> None:
        """Group found but no session for that folder returns (group, None)."""
        task = make_scheduled_task(context_mode="group", group_folder="test-group")
        group = make_registered_group("test-group")
        db_ops = make_db_ops()

        result = _resolve_task_context(
            task=task,
            start_time=0.0,
            registered_groups_fn=lambda: {"group@g.us": group},
            get_sessions_fn=lambda: {},  # empty sessions dict
            db_ops=db_ops,
        )

        assert result is not None
        resolved_group, session_id = result
        assert resolved_group is group
        assert session_id is None


# ---------------------------------------------------------------------------
# TestLogTaskCompletion
# ---------------------------------------------------------------------------


class TestLogTaskCompletion:
    """Tests for _log_task_completion."""

    def test_logs_success_run(self) -> None:
        """A successful run logs status='success' and calls update_task_after_run."""
        task = make_scheduled_task(schedule_type="interval", schedule_value="60000")
        db_ops = make_db_ops()

        _log_task_completion(
            task=task,
            start_time=0.0,
            status="success",
            result="All done.",
            error=None,
            timezone_str="UTC",
            db_ops=db_ops,
        )

        db_ops.log_task_run.assert_called_once()
        log: TaskRunLog = db_ops.log_task_run.call_args[0][0]
        assert log.task_id == task.id
        assert log.status == "success"
        assert log.result == "All done."
        assert log.error is None

        db_ops.update_task_after_run.assert_called_once()
        call_args = db_ops.update_task_after_run.call_args[0]
        assert call_args[0] == task.id
        # next_run should be a string (not None) for interval schedule
        assert call_args[1] is not None
        assert call_args[2] == "All done."

    def test_logs_error_run(self) -> None:
        """When error is set, log status='error' and include error text."""
        task = make_scheduled_task()
        db_ops = make_db_ops()

        _log_task_completion(
            task=task,
            start_time=0.0,
            status="error",
            result=None,
            error="Something exploded",
            timezone_str="UTC",
            db_ops=db_ops,
        )

        log: TaskRunLog = db_ops.log_task_run.call_args[0][0]
        assert log.status == "error"
        assert log.error == "Something exploded"

        # result_summary in update_task_after_run uses error message
        call_args = db_ops.update_task_after_run.call_args[0]
        assert "Something exploded" in call_args[2]

    def test_result_truncated_to_200_chars(self) -> None:
        """A result longer than 200 characters is truncated in the summary."""
        task = make_scheduled_task()
        db_ops = make_db_ops()
        long_result = "x" * 500

        _log_task_completion(
            task=task,
            start_time=0.0,
            status="success",
            result=long_result,
            error=None,
            timezone_str="UTC",
            db_ops=db_ops,
        )

        call_args = db_ops.update_task_after_run.call_args[0]
        result_summary: str = call_args[2]
        assert len(result_summary) == 200
        assert result_summary == "x" * 200

    def test_no_result_uses_completed_summary(self) -> None:
        """When result is None and no error, summary is 'Completed'."""
        task = make_scheduled_task()
        db_ops = make_db_ops()

        _log_task_completion(
            task=task,
            start_time=0.0,
            status="success",
            result=None,
            error=None,
            timezone_str="UTC",
            db_ops=db_ops,
        )

        call_args = db_ops.update_task_after_run.call_args[0]
        assert call_args[2] == "Completed"


# ---------------------------------------------------------------------------
# TestRunTask
# ---------------------------------------------------------------------------


class TestRunTask:
    """Tests for _run_task."""

    @pytest.mark.asyncio
    async def test_run_task_success(self) -> None:
        """A successful task run sends the output message and logs the run."""
        task = make_scheduled_task()
        group = make_registered_group("test-group")
        db_ops = make_db_ops()
        send_fn = AsyncMock()

        async def fake_run_agent(**kwargs: object) -> str:
            on_output = kwargs.get("on_streaming_output")
            if on_output:
                await on_output("Agent result text", False)  # type: ignore[misc]
            return "success"

        await _run_task(
            task=task,
            idle_timeout_ms=1000,
            groups_dir=Path("/tmp"),
            data_dir=Path("/tmp"),
            timezone_str="UTC",
            registered_groups_fn=lambda: {"group@g.us": group},
            get_sessions_fn=lambda: {},
            queue=MagicMock(),
            send_message_fn=send_fn,
            db_ops=db_ops,
            run_agent_fn=fake_run_agent,
        )

        send_fn.assert_called_once_with("group@g.us", "Agent result text")
        db_ops.log_task_run.assert_called_once()
        db_ops.update_task_after_run.assert_called_once()

        log: TaskRunLog = db_ops.log_task_run.call_args[0][0]
        assert log.status == "success"
        assert log.error is None

    @pytest.mark.asyncio
    async def test_run_task_agent_error_status(self) -> None:
        """When run_agent_fn returns 'error', an error is recorded in the log."""
        task = make_scheduled_task()
        group = make_registered_group("test-group")
        db_ops = make_db_ops()
        send_fn = AsyncMock()

        async def fake_run_agent(**kwargs: object) -> str:
            return "error"

        await _run_task(
            task=task,
            idle_timeout_ms=1000,
            groups_dir=Path("/tmp"),
            data_dir=Path("/tmp"),
            timezone_str="UTC",
            registered_groups_fn=lambda: {"group@g.us": group},
            get_sessions_fn=lambda: {},
            queue=MagicMock(),
            send_message_fn=send_fn,
            db_ops=db_ops,
            run_agent_fn=fake_run_agent,
        )

        log: TaskRunLog = db_ops.log_task_run.call_args[0][0]
        assert log.status == "error"
        assert log.error == "Agent returned error status"

    @pytest.mark.asyncio
    async def test_run_task_exception(self) -> None:
        """An exception raised by run_agent_fn is caught and logged as an error."""
        task = make_scheduled_task()
        group = make_registered_group("test-group")
        db_ops = make_db_ops()
        send_fn = AsyncMock()

        async def fake_run_agent(**kwargs: object) -> str:
            raise RuntimeError("container exploded")

        await _run_task(
            task=task,
            idle_timeout_ms=1000,
            groups_dir=Path("/tmp"),
            data_dir=Path("/tmp"),
            timezone_str="UTC",
            registered_groups_fn=lambda: {"group@g.us": group},
            get_sessions_fn=lambda: {},
            queue=MagicMock(),
            send_message_fn=send_fn,
            db_ops=db_ops,
            run_agent_fn=fake_run_agent,
        )

        log: TaskRunLog = db_ops.log_task_run.call_args[0][0]
        assert log.status == "error"
        assert "container exploded" in (log.error or "")

    @pytest.mark.asyncio
    async def test_run_task_group_not_found(self) -> None:
        """When the group is not found, _run_task returns early without logging a run."""
        task = make_scheduled_task(group_folder="ghost-group")
        db_ops = make_db_ops()
        send_fn = AsyncMock()

        await _run_task(
            task=task,
            idle_timeout_ms=1000,
            groups_dir=Path("/tmp"),
            data_dir=Path("/tmp"),
            timezone_str="UTC",
            registered_groups_fn=lambda: {},  # no groups registered
            get_sessions_fn=lambda: {},
            queue=MagicMock(),
            send_message_fn=send_fn,
            db_ops=db_ops,
            run_agent_fn=AsyncMock(return_value="success"),
        )

        # _log_error is called (one log_task_run from _log_error)
        db_ops.log_task_run.assert_called_once()
        # but update_task_after_run is NOT called because we returned early
        db_ops.update_task_after_run.assert_not_called()
        # and the message was never sent
        send_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_task_with_output_resets_idle_timer(self) -> None:
        """Receiving output from the agent causes the idle timer to be set."""
        task = make_scheduled_task()
        group = make_registered_group("test-group")
        db_ops = make_db_ops()
        send_fn = AsyncMock()
        queue = MagicMock()

        timer_was_set = False

        original_call_later = asyncio.get_event_loop().call_later

        def tracking_call_later(delay: float, callback: object, *args: object) -> object:
            nonlocal timer_was_set
            timer_was_set = True
            return original_call_later(delay, callback, *args)

        async def fake_run_agent(**kwargs: object) -> str:
            on_output = kwargs.get("on_streaming_output")
            if on_output:
                await on_output("some output", False)  # type: ignore[misc]
            return "success"

        loop = asyncio.get_event_loop()
        with patch.object(loop, "call_later", side_effect=tracking_call_later):
            await _run_task(
                task=task,
                idle_timeout_ms=1000,
                groups_dir=Path("/tmp"),
                data_dir=Path("/tmp"),
                timezone_str="UTC",
                registered_groups_fn=lambda: {"group@g.us": group},
                get_sessions_fn=lambda: {},
                queue=queue,
                send_message_fn=send_fn,
                db_ops=db_ops,
                run_agent_fn=fake_run_agent,
            )

        assert timer_was_set, "idle timer should have been set after receiving output"

    @pytest.mark.asyncio
    async def test_run_task_had_error_flag_sets_error(self) -> None:
        """When on_streaming_output is called with had_error=True, error is recorded."""
        task = make_scheduled_task()
        group = make_registered_group("test-group")
        db_ops = make_db_ops()
        send_fn = AsyncMock()

        async def fake_run_agent(**kwargs: object) -> str:
            on_output = kwargs.get("on_streaming_output")
            if on_output:
                await on_output("partial output", True)  # had_error=True
            return "success"

        await _run_task(
            task=task,
            idle_timeout_ms=1000,
            groups_dir=Path("/tmp"),
            data_dir=Path("/tmp"),
            timezone_str="UTC",
            registered_groups_fn=lambda: {"group@g.us": group},
            get_sessions_fn=lambda: {},
            queue=MagicMock(),
            send_message_fn=send_fn,
            db_ops=db_ops,
            run_agent_fn=fake_run_agent,
        )

        log: TaskRunLog = db_ops.log_task_run.call_args[0][0]
        assert log.status == "error"
        assert log.error == "Agent reported error"

    @pytest.mark.asyncio
    async def test_run_task_empty_output_not_sent(self) -> None:
        """Empty string output from the agent is not forwarded as a message."""
        task = make_scheduled_task()
        group = make_registered_group("test-group")
        db_ops = make_db_ops()
        send_fn = AsyncMock()

        async def fake_run_agent(**kwargs: object) -> str:
            on_output = kwargs.get("on_streaming_output")
            if on_output:
                await on_output("", False)  # empty output
            return "success"

        await _run_task(
            task=task,
            idle_timeout_ms=1000,
            groups_dir=Path("/tmp"),
            data_dir=Path("/tmp"),
            timezone_str="UTC",
            registered_groups_fn=lambda: {"group@g.us": group},
            get_sessions_fn=lambda: {},
            queue=MagicMock(),
            send_message_fn=send_fn,
            db_ops=db_ops,
            run_agent_fn=fake_run_agent,
        )

        # Empty string is falsy; send_message_fn must not be called
        send_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_task_prompt_includes_scheduled_header(self) -> None:
        """The prompt passed to run_agent_fn includes the scheduled-task header."""
        task = make_scheduled_task(prompt="Check the weather")
        group = make_registered_group("test-group")
        db_ops = make_db_ops()

        captured_kwargs: dict[str, object] = {}

        async def capturing_run_agent(**kwargs: object) -> str:
            captured_kwargs.update(kwargs)
            return "success"

        await _run_task(
            task=task,
            idle_timeout_ms=1000,
            groups_dir=Path("/tmp"),
            data_dir=Path("/tmp"),
            timezone_str="UTC",
            registered_groups_fn=lambda: {"group@g.us": group},
            get_sessions_fn=lambda: {},
            queue=MagicMock(),
            send_message_fn=AsyncMock(),
            db_ops=db_ops,
            run_agent_fn=capturing_run_agent,
        )

        prompt_sent = captured_kwargs.get("prompt", "")
        assert "SCHEDULED TASK" in str(prompt_sent)
        assert "Check the weather" in str(prompt_sent)

    @pytest.mark.asyncio
    async def test_run_task_passes_is_scheduled_task_flag(self) -> None:
        """run_agent_fn is called with is_scheduled_task=True."""
        task = make_scheduled_task()
        group = make_registered_group("test-group")
        db_ops = make_db_ops()

        captured_kwargs: dict[str, object] = {}

        async def capturing_run_agent(**kwargs: object) -> str:
            captured_kwargs.update(kwargs)
            return "success"

        await _run_task(
            task=task,
            idle_timeout_ms=1000,
            groups_dir=Path("/tmp"),
            data_dir=Path("/tmp"),
            timezone_str="UTC",
            registered_groups_fn=lambda: {"group@g.us": group},
            get_sessions_fn=lambda: {},
            queue=MagicMock(),
            send_message_fn=AsyncMock(),
            db_ops=db_ops,
            run_agent_fn=capturing_run_agent,
        )

        assert captured_kwargs.get("is_scheduled_task") is True

    @pytest.mark.asyncio
    async def test_run_task_idle_timer_reset_cancels_previous_handle(self) -> None:
        """When output arrives twice, the first idle timer is cancelled before a new one is set."""
        task = make_scheduled_task()
        group = make_registered_group("test-group")
        db_ops = make_db_ops()
        send_fn = AsyncMock()
        queue = MagicMock()

        cancel_count = 0

        class TrackingHandle:
            def cancel(self) -> None:
                nonlocal cancel_count
                cancel_count += 1

        loop = asyncio.get_event_loop()
        handle_index = 0

        def patched_call_later(delay: float, cb: object, *args: object) -> TrackingHandle:
            nonlocal handle_index
            handle_index += 1
            return TrackingHandle()

        async def fake_run_agent(**kwargs: object) -> str:
            on_output = kwargs.get("on_streaming_output")
            if on_output:
                # First output: sets idle_timer_handle to a TrackingHandle
                await on_output("first output", False)  # type: ignore[misc]
                # Second output: cancels the first handle, then sets a new one
                await on_output("second output", False)  # type: ignore[misc]
            return "success"

        with patch.object(loop, "call_later", side_effect=patched_call_later):
            await _run_task(
                task=task,
                idle_timeout_ms=1000,
                groups_dir=Path("/tmp"),
                data_dir=Path("/tmp"),
                timezone_str="UTC",
                registered_groups_fn=lambda: {"group@g.us": group},
                get_sessions_fn=lambda: {},
                queue=queue,
                send_message_fn=send_fn,
                db_ops=db_ops,
                run_agent_fn=fake_run_agent,
            )

        # call_later called twice (once per output event).
        # cancel() is called at least twice: once when the second output resets the timer,
        # and once at the end of _run_task when the final handle is cancelled.
        assert handle_index == 2
        assert cancel_count >= 2, "idle timer handles should be cancelled on reset and at task end"

    @pytest.mark.asyncio
    async def test_run_task_idle_timer_cancelled_on_exception(self) -> None:
        """The idle timer handle is cancelled even when an exception is raised."""
        task = make_scheduled_task()
        group = make_registered_group("test-group")
        db_ops = make_db_ops()
        cancel_called = False

        class FakeHandle:
            def cancel(self) -> None:
                nonlocal cancel_called
                cancel_called = True

        loop = asyncio.get_event_loop()
        original_call_later = loop.call_later

        def patched_call_later(delay: float, cb: object, *args: object) -> FakeHandle:
            original_call_later(delay, cb, *args)
            return FakeHandle()

        async def run_and_output(**kwargs: object) -> str:
            on_output = kwargs.get("on_streaming_output")
            if on_output:
                await on_output("some text", False)  # type: ignore[misc]
            raise RuntimeError("boom")

        with patch.object(loop, "call_later", side_effect=patched_call_later):
            await _run_task(
                task=task,
                idle_timeout_ms=5000,
                groups_dir=Path("/tmp"),
                data_dir=Path("/tmp"),
                timezone_str="UTC",
                registered_groups_fn=lambda: {"group@g.us": group},
                get_sessions_fn=lambda: {},
                queue=MagicMock(),
                send_message_fn=AsyncMock(),
                db_ops=db_ops,
                run_agent_fn=run_and_output,
            )

        assert cancel_called, "idle timer handle.cancel() should be called on exception"


# ---------------------------------------------------------------------------
# TestCheckDueTasks
# ---------------------------------------------------------------------------


class TestCheckDueTasks:
    """Tests for _check_due_tasks."""

    def _make_common_kwargs(
        self,
        db_ops: MagicMock | None = None,
        queue: MagicMock | None = None,
        registered_groups_fn: object = None,
    ) -> dict[str, object]:
        """Return a complete set of keyword arguments for _check_due_tasks."""
        return {
            "idle_timeout_ms": 1000,
            "groups_dir": Path("/tmp"),
            "data_dir": Path("/tmp"),
            "timezone_str": "UTC",
            "registered_groups_fn": registered_groups_fn or (lambda: {}),
            "get_sessions_fn": lambda: {},
            "queue": queue or MagicMock(),
            "send_message_fn": AsyncMock(),
            "db_ops": db_ops or make_db_ops(),
            "run_agent_fn": AsyncMock(return_value="success"),
        }

    @pytest.mark.asyncio
    async def test_no_due_tasks(self) -> None:
        """When get_due_tasks returns [], nothing is enqueued."""
        db_ops = make_db_ops()
        db_ops.get_due_tasks.return_value = []
        queue = MagicMock()

        await _check_due_tasks(**self._make_common_kwargs(db_ops=db_ops, queue=queue))  # type: ignore[arg-type]

        queue.enqueue_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_due_task_enqueued(self) -> None:
        """An active due task is re-confirmed then enqueued."""
        task = make_scheduled_task(status="active")
        db_ops = make_db_ops()
        db_ops.get_due_tasks.return_value = [task]
        db_ops.get_task_by_id.return_value = task  # re-check confirms active
        queue = MagicMock()

        await _check_due_tasks(**self._make_common_kwargs(db_ops=db_ops, queue=queue))  # type: ignore[arg-type]

        queue.enqueue_task.assert_called_once_with(
            task.chat_jid,
            task.id,
            queue.enqueue_task.call_args[0][2],  # callable (closure)
        )

    @pytest.mark.asyncio
    async def test_paused_task_skipped(self) -> None:
        """A task that was paused between query and re-check is not enqueued."""
        task = make_scheduled_task(status="active")
        paused_task = make_scheduled_task(status="paused")
        db_ops = make_db_ops()
        db_ops.get_due_tasks.return_value = [task]
        # Re-check reveals it's now paused
        db_ops.get_task_by_id.return_value = paused_task
        queue = MagicMock()

        await _check_due_tasks(**self._make_common_kwargs(db_ops=db_ops, queue=queue))  # type: ignore[arg-type]

        queue.enqueue_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_task_not_found_on_recheck_skipped(self) -> None:
        """If get_task_by_id returns None on the re-check, task is skipped."""
        task = make_scheduled_task()
        db_ops = make_db_ops()
        db_ops.get_due_tasks.return_value = [task]
        db_ops.get_task_by_id.return_value = None  # disappeared between calls
        queue = MagicMock()

        await _check_due_tasks(**self._make_common_kwargs(db_ops=db_ops, queue=queue))  # type: ignore[arg-type]

        queue.enqueue_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_due_tasks_all_enqueued(self) -> None:
        """Multiple active due tasks are all enqueued independently."""
        task_a = make_scheduled_task(id="a", chat_jid="a@g.us")
        task_b = make_scheduled_task(id="b", chat_jid="b@g.us")
        db_ops = make_db_ops()
        db_ops.get_due_tasks.return_value = [task_a, task_b]
        db_ops.get_task_by_id.side_effect = lambda tid: task_a if tid == "a" else task_b
        queue = MagicMock()

        await _check_due_tasks(**self._make_common_kwargs(db_ops=db_ops, queue=queue))  # type: ignore[arg-type]

        assert queue.enqueue_task.call_count == 2

    @pytest.mark.asyncio
    async def test_enqueued_closure_runs_task(self) -> None:
        """The closure passed to enqueue_task, when awaited, actually runs the task."""
        task = make_scheduled_task(group_folder="test-group")
        group = make_registered_group("test-group")
        db_ops = make_db_ops()
        db_ops.get_due_tasks.return_value = [task]
        db_ops.get_task_by_id.return_value = task
        queue = MagicMock()
        send_fn = AsyncMock()

        async def fake_run_agent(**kwargs: object) -> str:
            on_output = kwargs.get("on_streaming_output")
            if on_output:
                await on_output("result", False)  # type: ignore[misc]
            return "success"

        await _check_due_tasks(
            idle_timeout_ms=1000,
            groups_dir=Path("/tmp"),
            data_dir=Path("/tmp"),
            timezone_str="UTC",
            registered_groups_fn=lambda: {"group@g.us": group},
            get_sessions_fn=lambda: {},
            queue=queue,
            send_message_fn=send_fn,
            db_ops=db_ops,
            run_agent_fn=fake_run_agent,
        )

        # Retrieve and invoke the enqueued closure
        closure = queue.enqueue_task.call_args[0][2]
        await closure()

        send_fn.assert_called_once_with(task.chat_jid, "result")
        db_ops.log_task_run.assert_called_once()


# ---------------------------------------------------------------------------
# TestStartSchedulerLoop
# ---------------------------------------------------------------------------


class TestStartSchedulerLoop:
    """Tests for start_scheduler_loop."""

    def _make_loop_kwargs(
        self,
        db_ops: MagicMock | None = None,
        queue: MagicMock | None = None,
    ) -> dict[str, object]:
        return {
            "poll_interval_s": 0.01,
            "idle_timeout_ms": 1000,
            "groups_dir": Path("/tmp"),
            "data_dir": Path("/tmp"),
            "timezone_str": "UTC",
            "registered_groups_fn": lambda: {},
            "get_sessions_fn": lambda: {},
            "queue": queue or MagicMock(),
            "send_message_fn": AsyncMock(),
            "db_ops": db_ops or make_db_ops(),
            "run_agent_fn": AsyncMock(return_value="success"),
        }

    @pytest.mark.asyncio
    async def test_prevents_duplicate_start(self) -> None:
        """A second call to start_scheduler_loop while the first is running returns immediately."""
        # Pre-set the flag as if a loop is already running
        sched_mod._scheduler_running = True

        db_ops = make_db_ops()
        # Should return immediately without entering the loop
        await start_scheduler_loop(**self._make_loop_kwargs(db_ops=db_ops))  # type: ignore[arg-type]

        # get_due_tasks never called because loop was skipped
        db_ops.get_due_tasks.assert_not_called()

    @pytest.mark.asyncio
    async def test_loop_calls_check_due_tasks(self) -> None:
        """The polling loop calls _check_due_tasks at least once per iteration."""
        db_ops = make_db_ops()
        db_ops.get_due_tasks.return_value = []
        call_count = 0

        async def fake_sleep(t: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise asyncio.CancelledError()

        with patch("nanoclaw.scheduler.asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(asyncio.CancelledError):
                await start_scheduler_loop(**self._make_loop_kwargs(db_ops=db_ops))  # type: ignore[arg-type]

        assert db_ops.get_due_tasks.call_count >= 1

    @pytest.mark.asyncio
    async def test_loop_sets_running_flag(self) -> None:
        """start_scheduler_loop sets _scheduler_running=True while running."""
        assert sched_mod._scheduler_running is False

        running_during_loop = False

        async def fake_sleep(t: float) -> None:
            nonlocal running_during_loop
            running_during_loop = sched_mod._scheduler_running
            raise asyncio.CancelledError()

        with patch("nanoclaw.scheduler.asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(asyncio.CancelledError):
                await start_scheduler_loop(**self._make_loop_kwargs())  # type: ignore[arg-type]

        assert running_during_loop is True

    @pytest.mark.asyncio
    async def test_loop_continues_after_check_exception(self) -> None:
        """An exception in _check_due_tasks is caught and the loop continues."""
        db_ops = make_db_ops()
        # First call raises, second call succeeds (but sleep cancels after that)
        call_count = 0

        def raise_then_succeed() -> list[object]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("db error")
            return []

        db_ops.get_due_tasks.side_effect = raise_then_succeed

        sleep_count = 0

        async def counting_sleep(t: float) -> None:
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 2:
                raise asyncio.CancelledError()

        with patch("nanoclaw.scheduler.asyncio.sleep", side_effect=counting_sleep):
            with pytest.raises(asyncio.CancelledError):
                await start_scheduler_loop(**self._make_loop_kwargs(db_ops=db_ops))  # type: ignore[arg-type]

        # Both iterations completed (exception was caught by the loop's except clause)
        assert sleep_count == 2
        assert db_ops.get_due_tasks.call_count == 2

    @pytest.mark.asyncio
    async def test_loop_uses_poll_interval(self) -> None:
        """asyncio.sleep is called with the specified poll_interval_s."""
        db_ops = make_db_ops()
        captured_sleep_args: list[float] = []

        async def capturing_sleep(t: float) -> None:
            captured_sleep_args.append(t)
            raise asyncio.CancelledError()

        with patch("nanoclaw.scheduler.asyncio.sleep", side_effect=capturing_sleep):
            with pytest.raises(asyncio.CancelledError):
                await start_scheduler_loop(
                    poll_interval_s=42.5,
                    idle_timeout_ms=1000,
                    groups_dir=Path("/tmp"),
                    data_dir=Path("/tmp"),
                    timezone_str="UTC",
                    registered_groups_fn=lambda: {},
                    get_sessions_fn=lambda: {},
                    queue=MagicMock(),
                    send_message_fn=AsyncMock(),
                    db_ops=db_ops,
                    run_agent_fn=AsyncMock(return_value="success"),
                )

        assert captured_sleep_args[0] == 42.5

# end tests/test_scheduler.py
