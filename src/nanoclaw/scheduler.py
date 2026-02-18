# start src/nanoclaw/scheduler.py
"""Scheduled task runner for NanoClaw.

Replaces src/task-scheduler.ts from TypeScript.
Polls the database for due tasks and enqueues them in the GroupQueue.
Uses croniter for timezone-aware cron expression parsing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from croniter import croniter

from nanoclaw.types import RegisteredGroup, ScheduledTask

logger = logging.getLogger(__name__)

MAIN_GROUP_FOLDER = "main"

_scheduler_running = False


async def start_scheduler_loop(
    poll_interval_s: float,
    idle_timeout_ms: int,
    groups_dir: Path,
    data_dir: Path,
    timezone_str: str,
    registered_groups_fn: Callable[[], dict[str, RegisteredGroup]],
    get_sessions_fn: Callable[[], dict[str, str]],
    queue: Any,
    send_message_fn: Callable[[str, str], Coroutine[Any, Any, None]],
    db_ops: Any,
    run_agent_fn: Callable[..., Coroutine[Any, Any, str]],
) -> None:
    """Start the scheduler polling loop.

    Polls for due scheduled tasks every poll_interval_s seconds and enqueues
    them via the GroupQueue. Each task runs in its own container.

    Args:
        poll_interval_s: Seconds between scheduler polls.
        idle_timeout_ms: Idle timeout for container processes in milliseconds.
        groups_dir: Absolute path to the groups/ directory.
        data_dir: Absolute path to the data/ directory.
        timezone_str: Timezone string for cron parsing.
        registered_groups_fn: Callable returning the current registered groups dict.
        get_sessions_fn: Callable returning the current sessions dict.
        queue: GroupQueue instance for enqueuing work.
        send_message_fn: Async callable to send a WhatsApp message.
        db_ops: Database operations module.
        run_agent_fn: Async callable that runs the agent for a group/prompt.
    """
    global _scheduler_running
    if _scheduler_running:
        logger.debug("Scheduler loop already running, skipping duplicate start")
        return
    _scheduler_running = True
    logger.info("Scheduler loop started (poll_interval=%.1fs)", poll_interval_s)

    while True:
        try:
            await _check_due_tasks(
                idle_timeout_ms=idle_timeout_ms,
                groups_dir=groups_dir,
                data_dir=data_dir,
                timezone_str=timezone_str,
                registered_groups_fn=registered_groups_fn,
                get_sessions_fn=get_sessions_fn,
                queue=queue,
                send_message_fn=send_message_fn,
                db_ops=db_ops,
                run_agent_fn=run_agent_fn,
            )
        except Exception as exc:
            logger.error("Error in scheduler loop: %s", exc)
        await asyncio.sleep(poll_interval_s)


async def _check_due_tasks(
    idle_timeout_ms: int,
    groups_dir: Path,
    data_dir: Path,
    timezone_str: str,
    registered_groups_fn: Callable[[], dict[str, RegisteredGroup]],
    get_sessions_fn: Callable[[], dict[str, str]],
    queue: Any,
    send_message_fn: Callable[[str, str], Coroutine[Any, Any, None]],
    db_ops: Any,
    run_agent_fn: Callable[..., Coroutine[Any, Any, str]],
) -> None:
    """Fetch and enqueue all currently due scheduled tasks.

    Args:
        idle_timeout_ms: Idle timeout for container processes.
        groups_dir: Absolute path to the groups/ directory.
        data_dir: Absolute path to the data/ directory.
        timezone_str: Timezone for cron parsing.
        registered_groups_fn: Callable returning registered groups.
        get_sessions_fn: Callable returning session dict.
        queue: GroupQueue instance.
        send_message_fn: Async message sender.
        db_ops: Database operations module.
        run_agent_fn: Async agent runner callable.
    """
    due_tasks = db_ops.get_due_tasks()
    if due_tasks:
        logger.info("Found %d due task(s)", len(due_tasks))

    for task in due_tasks:
        # Re-check in case task was paused/cancelled since the query
        current = db_ops.get_task_by_id(task.id)
        if not current or current.status != "active":
            continue

        def make_task_fn(t: ScheduledTask) -> Callable[[], Coroutine[Any, Any, None]]:
            """Create a closure that runs a specific task."""

            async def run() -> None:
                await _run_task(
                    task=t,
                    idle_timeout_ms=idle_timeout_ms,
                    groups_dir=groups_dir,
                    data_dir=data_dir,
                    timezone_str=timezone_str,
                    registered_groups_fn=registered_groups_fn,
                    get_sessions_fn=get_sessions_fn,
                    queue=queue,
                    send_message_fn=send_message_fn,
                    db_ops=db_ops,
                    run_agent_fn=run_agent_fn,
                )

            return run

        queue.enqueue_task(
            current.chat_jid,
            current.id,
            make_task_fn(current),
        )


def _resolve_task_context(
    task: ScheduledTask,
    start_time: float,
    registered_groups_fn: Callable[[], dict[str, RegisteredGroup]],
    get_sessions_fn: Callable[[], dict[str, str]],
    db_ops: Any,
) -> tuple[RegisteredGroup, str | None] | None:
    """Resolve the group and session for a scheduled task.

    Args:
        task: The scheduled task to resolve context for.
        start_time: Monotonic start time for duration calculation on error.
        registered_groups_fn: Callable returning registered groups.
        get_sessions_fn: Callable returning session dict.
        db_ops: Database operations module.

    Returns:
        Tuple of (group, session_id) on success, or None if group not found.
    """
    groups = registered_groups_fn()
    group = next(
        (g for g in groups.values() if g.folder == task.group_folder),
        None,
    )
    if not group:
        logger.error("Task %s references unknown group folder: %s", task.id, task.group_folder)
        _log_error(
            task.id,
            int((time.monotonic() - start_time) * 1000),
            f"Group not found: {task.group_folder}",
            db_ops,
        )
        return None
    sessions = get_sessions_fn()
    session_id = sessions.get(task.group_folder) if task.context_mode == "group" else None
    return group, session_id


def _log_task_completion(
    task: ScheduledTask,
    start_time: float,
    status: str,
    result: str | None,
    error: str | None,
    timezone_str: str,
    db_ops: Any,
) -> None:
    """Log the task run result and update the next_run timestamp.

    Args:
        task: The scheduled task that ran.
        start_time: Monotonic start time for duration calculation.
        status: Agent run status ('success' or 'error').
        result: Agent output text, if any.
        error: Error message, if any.
        timezone_str: Timezone for next-run calculation.
        db_ops: Database operations module.
    """
    from nanoclaw.types import TaskRunLog

    duration_ms = int((time.monotonic() - start_time) * 1000)
    db_ops.log_task_run(
        TaskRunLog(
            task_id=task.id,
            run_at=datetime.now(UTC).isoformat(),
            duration_ms=duration_ms,
            status="error" if error else "success",
            result=result,
            error=error,
        )
    )
    next_run = _calculate_next_run(task.schedule_type, task.schedule_value, timezone_str)
    result_summary = f"Error: {error}" if error else (result[:200] if result else "Completed")
    db_ops.update_task_after_run(task.id, next_run, result_summary)
    logger.info(
        "Task %s completed in %dms (status=%s)",
        task.id,
        duration_ms,
        "error" if error else "success",
    )


async def _run_task(
    task: ScheduledTask,
    idle_timeout_ms: int,
    groups_dir: Path,
    data_dir: Path,
    timezone_str: str,
    registered_groups_fn: Callable[[], dict[str, RegisteredGroup]],
    get_sessions_fn: Callable[[], dict[str, str]],
    queue: Any,
    send_message_fn: Callable[[str, str], Coroutine[Any, Any, None]],
    db_ops: Any,
    run_agent_fn: Callable[..., Coroutine[Any, Any, str]],
) -> None:
    """Execute a single scheduled task.

    Runs the agent, sends the result to the user, logs the run, and
    updates the next_run timestamp.

    Args:
        task: The scheduled task to execute.
        idle_timeout_ms: Idle timeout for the container process.
        groups_dir: Absolute path to groups/ directory.
        data_dir: Absolute path to data/ directory.
        timezone_str: Timezone for next-run calculation.
        registered_groups_fn: Callable returning registered groups.
        get_sessions_fn: Callable returning session dict.
        queue: GroupQueue instance.
        send_message_fn: Async message sender for results.
        db_ops: Database operations module.
        run_agent_fn: Async agent runner callable.
    """
    start_time = time.monotonic()
    logger.info("Running scheduled task %s for group %s", task.id, task.group_folder)

    resolved = _resolve_task_context(
        task, start_time, registered_groups_fn, get_sessions_fn, db_ops
    )
    if resolved is None:
        return
    group, session_id = resolved

    prompt = (
        f"[SCHEDULED TASK - The following message was sent automatically "
        f"and is not coming directly from the user or group.]\n\n{task.prompt}"
    )

    result: str | None = None
    error: str | None = None

    # Idle timer: close container stdin after idle_timeout_ms of no output
    idle_timer_handle: asyncio.TimerHandle | None = None

    def reset_idle_timer() -> None:
        """Reset the idle timeout for the container."""
        nonlocal idle_timer_handle
        loop = asyncio.get_event_loop()
        if idle_timer_handle:
            idle_timer_handle.cancel()
        idle_timer_handle = loop.call_later(
            idle_timeout_ms / 1000.0,
            lambda: queue.close_stdin(task.chat_jid),
        )

    async def on_output(output_result: str, had_error: bool) -> None:
        """Handle streamed agent output.

        Args:
            output_result: The result text from the agent.
            had_error: Whether the agent reported an error.
        """
        nonlocal result, error
        if output_result:
            result = output_result
            await send_message_fn(task.chat_jid, output_result)
            reset_idle_timer()
        if had_error:
            error = "Agent reported error"

    try:
        status = await run_agent_fn(
            group=group,
            prompt=prompt,
            chat_jid=task.chat_jid,
            session_id=session_id,
            is_scheduled_task=True,
            on_streaming_output=on_output,
        )
        if idle_timer_handle:
            idle_timer_handle.cancel()
        if status == "error" and not error:
            error = "Agent returned error status"
    except Exception as exc:
        if idle_timer_handle:
            idle_timer_handle.cancel()
        error = str(exc)
        logger.error("Task %s failed with exception: %s", task.id, exc)

    _log_task_completion(
        task, start_time, "error" if error else "success", result, error, timezone_str, db_ops
    )


def _calculate_next_run(
    schedule_type: str,
    schedule_value: str,
    timezone_str: str,
) -> str | None:
    """Compute the next scheduled run time for a task.

    Args:
        schedule_type: 'cron', 'interval', or 'once'.
        schedule_value: Cron expression, ms integer, or ISO timestamp.
        timezone_str: Timezone string for cron calculation.

    Returns:
        ISO timestamp string for the next run, or None if done.
    """
    now = datetime.now(UTC)
    if schedule_type == "cron":
        try:
            cron = croniter(schedule_value, now)
            return cron.get_next(datetime).isoformat()
        except Exception as exc:
            logger.error("Failed to parse cron expression '%s': %s", schedule_value, exc)
            return None
    elif schedule_type == "interval":
        try:
            ms = int(schedule_value)
            next_dt = datetime.fromtimestamp(time.time() + ms / 1000.0, tz=UTC)
            return next_dt.isoformat()
        except (ValueError, TypeError) as exc:
            logger.error("Invalid interval value '%s': %s", schedule_value, exc)
            return None
    # 'once' tasks have no next run
    return None


def _log_error(
    task_id: str,
    duration_ms: int,
    error_msg: str,
    db_ops: Any,
) -> None:
    """Log a task run error to the database.

    Args:
        task_id: The task ID that failed.
        duration_ms: How long the attempt took.
        error_msg: The error message to record.
        db_ops: Database operations module.
    """
    from nanoclaw.types import TaskRunLog

    db_ops.log_task_run(
        TaskRunLog(
            task_id=task_id,
            run_at=datetime.now(UTC).isoformat(),
            duration_ms=duration_ms,
            status="error",
            result=None,
            error=error_msg,
        )
    )


# end src/nanoclaw/scheduler.py
