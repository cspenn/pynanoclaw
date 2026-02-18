# start src/nanoclaw/ipc.py
"""IPC file watcher for NanoClaw.

Replaces src/ipc.ts from TypeScript. Polls per-group IPC directories
for JSON files written by container agents, processes them with authorization,
and delegates to the appropriate handler (message send, task scheduling, etc.).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

from croniter import croniter

from nanoclaw.types import ContainerConfig, RegisteredGroup

logger = logging.getLogger(__name__)

MAIN_GROUP_FOLDER = "main"


@dataclass
class IpcDeps:
    """Dependencies injected into the IPC watcher.

    Attributes:
        send_message: Async callable to send a WhatsApp message.
        registered_groups: Callable returning the current dict of registered groups.
        register_group: Callable to register a new group (main-only).
        sync_group_metadata: Async callable to refresh group metadata from WhatsApp.
        get_available_groups: Callable returning list of visible WhatsApp groups.
        write_groups_snapshot: Callable to update the groups snapshot file.
        timezone: Timezone string for cron parsing.
    """

    send_message: Callable[[str, str], Coroutine[Any, Any, None]]
    registered_groups: Callable[[], dict[str, RegisteredGroup]]
    register_group: Callable[[str, RegisteredGroup], None]
    sync_group_metadata: Callable[[bool], Coroutine[Any, Any, None]]
    get_available_groups: Callable[[], list[Any]]
    write_groups_snapshot: Callable[[str, bool, list[Any], set[str]], None]
    timezone: str = "America/New_York"


_watcher_running = False


async def start_ipc_watcher(
    data_dir: Path,
    poll_interval_s: float,
    deps: IpcDeps,
    db_ops: Any,
) -> None:
    """Start the IPC file polling loop.

    Scans per-group IPC directories every poll_interval_s seconds.
    Processes message files (forward to WhatsApp) and task files
    (schedule/cancel/register).

    Args:
        data_dir: Root data directory containing ipc/ subdirectory.
        poll_interval_s: Seconds between polls.
        deps: Injected dependencies.
        db_ops: Database operations module for task management.
    """
    global _watcher_running
    if _watcher_running:
        logger.debug("IPC watcher already running, skipping duplicate start")
        return
    _watcher_running = True

    ipc_base = data_dir / "ipc"
    ipc_base.mkdir(parents=True, exist_ok=True)
    logger.info("IPC watcher started (per-group namespaces)")

    while True:
        try:
            await _process_all_ipc_dirs(ipc_base, deps, db_ops)
        except Exception as exc:
            logger.error("Error in IPC watcher loop: %s", exc)
        await asyncio.sleep(poll_interval_s)


async def _process_all_ipc_dirs(
    ipc_base: Path,
    deps: IpcDeps,
    db_ops: Any,
) -> None:
    """Scan all group IPC directories and process pending files.

    Args:
        ipc_base: Root IPC directory (data/ipc/).
        deps: Injected dependencies.
        db_ops: Database operations module.
    """
    try:
        group_folders = [d.name for d in ipc_base.iterdir() if d.is_dir() and d.name != "errors"]
    except OSError as exc:
        logger.error("Error reading IPC base directory: %s", exc)
        return

    registered_groups = deps.registered_groups()

    for source_group in group_folders:
        is_main = source_group == MAIN_GROUP_FOLDER
        messages_dir = ipc_base / source_group / "messages"
        tasks_dir = ipc_base / source_group / "tasks"

        await _process_ipc_messages(
            messages_dir, source_group, is_main, registered_groups, deps, ipc_base
        )
        await _process_ipc_tasks(tasks_dir, source_group, is_main, deps, db_ops, ipc_base)


async def _process_single_ipc_message(
    file_path: Path,
    source_group: str,
    is_main: bool,
    registered_groups: dict[str, RegisteredGroup],
    deps: IpcDeps,
    ipc_base: Path,
) -> None:
    """Process a single IPC message JSON file.

    Args:
        file_path: Path to the message JSON file.
        source_group: The group folder identity (from directory path).
        is_main: Whether this is the main group.
        registered_groups: Current registered groups dict.
        deps: Injected dependencies.
        ipc_base: Root IPC directory for error file placement.
    """
    try:
        data = json.loads(file_path.read_text())
        if data.get("type") == "message" and data.get("chatJid") and data.get("text"):
            chat_jid = data["chatJid"]
            target_group = registered_groups.get(chat_jid)
            authorized = is_main or (
                target_group is not None and target_group.folder == source_group
            )
            if authorized:
                await deps.send_message(chat_jid, data["text"])
                logger.info("IPC message sent to %s from %s", chat_jid, source_group)
            else:
                logger.warning(
                    "Unauthorized IPC message blocked: source=%s target=%s",
                    source_group,
                    chat_jid,
                )
        file_path.unlink(missing_ok=True)
    except Exception as exc:
        logger.error("Error processing IPC message %s: %s", file_path.name, exc)
        _move_to_errors(file_path, source_group, ipc_base)


async def _process_ipc_messages(
    messages_dir: Path,
    source_group: str,
    is_main: bool,
    registered_groups: dict[str, RegisteredGroup],
    deps: IpcDeps,
    ipc_base: Path,
) -> None:
    """Process message JSON files from a group's IPC messages directory.

    Args:
        messages_dir: Path to the group's messages/ directory.
        source_group: The group folder identity (from directory path).
        is_main: Whether this is the main group.
        registered_groups: Current registered groups dict.
        deps: Injected dependencies.
        ipc_base: Root IPC directory for error file placement.
    """
    if not messages_dir.exists():
        return
    try:
        message_files = sorted(messages_dir.glob("*.json"))
    except OSError:
        return

    for file_path in message_files:
        await _process_single_ipc_message(
            file_path, source_group, is_main, registered_groups, deps, ipc_base
        )


async def _process_ipc_tasks(
    tasks_dir: Path,
    source_group: str,
    is_main: bool,
    deps: IpcDeps,
    db_ops: Any,
    ipc_base: Path,
) -> None:
    """Process task JSON files from a group's IPC tasks directory.

    Args:
        tasks_dir: Path to the group's tasks/ directory.
        source_group: The group folder identity.
        is_main: Whether this is the main group.
        deps: Injected dependencies.
        db_ops: Database operations module.
        ipc_base: Root IPC directory for error file placement.
    """
    if not tasks_dir.exists():
        return
    try:
        task_files = sorted(tasks_dir.glob("*.json"))
    except OSError:
        return

    for file_path in task_files:
        try:
            data = json.loads(file_path.read_text())
            await process_task_ipc(data, source_group, is_main, deps, db_ops)
            file_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.error("Error processing IPC task %s: %s", file_path.name, exc)
            _move_to_errors(file_path, source_group, ipc_base)


async def process_task_ipc(
    data: dict[str, Any],
    source_group: str,
    is_main: bool,
    deps: IpcDeps,
    db_ops: Any,
) -> None:
    """Handle a single IPC task request.

    Dispatches to the appropriate handler based on data['type'].
    Enforces authorization: non-main groups can only act on themselves.

    Args:
        data: Parsed JSON data from the IPC task file.
        source_group: Verified group identity from IPC directory path.
        is_main: Whether this group has main-group privileges.
        deps: Injected dependencies.
        db_ops: Database operations module.
    """
    task_type = data.get("type", "")
    registered_groups = deps.registered_groups()

    match task_type:
        case "schedule_task":
            await _handle_schedule_task(
                data, source_group, is_main, registered_groups, deps, db_ops
            )
        case "pause_task":
            _handle_task_action(data, source_group, is_main, "paused", db_ops)
        case "resume_task":
            _handle_task_action(data, source_group, is_main, "active", db_ops)
        case "cancel_task":
            _handle_cancel_task(data, source_group, is_main, db_ops)
        case "refresh_groups":
            await _handle_refresh_groups(source_group, is_main, registered_groups, deps)
        case "register_group":
            _handle_register_group(data, source_group, is_main, deps)
        case _:
            logger.warning("Unknown IPC task type: %s", task_type)


async def _handle_schedule_task(
    data: dict[str, Any],
    source_group: str,
    is_main: bool,
    registered_groups: dict[str, RegisteredGroup],
    deps: IpcDeps,
    db_ops: Any,
) -> None:
    """Handle schedule_task IPC request.

    Args:
        data: IPC request data.
        source_group: Source group folder identity.
        is_main: Main group flag.
        registered_groups: Current registered groups.
        deps: IPC dependencies.
        db_ops: Database operations.
    """
    required = ["prompt", "schedule_type", "schedule_value", "targetJid"]
    if not all(data.get(k) for k in required):
        logger.warning("schedule_task missing required fields: %s", data)
        return

    target_jid = data["targetJid"]
    target_group_entry = registered_groups.get(target_jid)
    if not target_group_entry:
        logger.warning("Cannot schedule task: target JID not registered: %s", target_jid)
        return

    target_folder = target_group_entry.folder
    if not is_main and target_folder != source_group:
        logger.warning(
            "Unauthorized schedule_task: source=%s target=%s",
            source_group,
            target_folder,
        )
        return

    schedule_type = data["schedule_type"]
    schedule_value = data["schedule_value"]
    next_run = _calculate_next_run(schedule_type, schedule_value, deps.timezone)
    if next_run is None and schedule_type in ("cron", "interval"):
        return  # already logged in _calculate_next_run

    import random
    import string
    import time

    task_id = (
        f"task-{int(time.time() * 1000)}-{''.join(random.choices(string.ascii_lowercase, k=6))}"
    )
    context_mode = data.get("context_mode", "isolated")
    if context_mode not in ("group", "isolated"):
        context_mode = "isolated"

    from datetime import datetime

    from nanoclaw.types import ScheduledTask

    db_ops.create_task(
        ScheduledTask(
            id=task_id,
            group_folder=target_folder,
            chat_jid=target_jid,
            prompt=data["prompt"],
            schedule_type=schedule_type,
            schedule_value=schedule_value,
            context_mode=context_mode,
            next_run=next_run,
            status="active",
            created_at=datetime.now(UTC).isoformat(),
        )
    )
    logger.info("Task %s created via IPC from %s", task_id, source_group)


def _handle_task_action(
    data: dict[str, Any],
    source_group: str,
    is_main: bool,
    new_status: str,
    db_ops: Any,
) -> None:
    """Handle pause_task or resume_task IPC request.

    Args:
        data: IPC request data.
        source_group: Source group folder identity.
        is_main: Main group flag.
        new_status: Status to set ('paused' or 'active').
        db_ops: Database operations.
    """
    task_id = data.get("taskId")
    if not task_id:
        return
    task = db_ops.get_task_by_id(task_id)
    if task and (is_main or task.group_folder == source_group):
        db_ops.update_task(task_id, status=new_status)
        logger.info("Task %s set to %s via IPC from %s", task_id, new_status, source_group)
    else:
        logger.warning(
            "Unauthorized task action (status=%s): task=%s source=%s",
            new_status,
            task_id,
            source_group,
        )


def _handle_cancel_task(
    data: dict[str, Any],
    source_group: str,
    is_main: bool,
    db_ops: Any,
) -> None:
    """Handle cancel_task IPC request.

    Args:
        data: IPC request data.
        source_group: Source group folder identity.
        is_main: Main group flag.
        db_ops: Database operations.
    """
    task_id = data.get("taskId")
    if not task_id:
        return
    task = db_ops.get_task_by_id(task_id)
    if task and (is_main or task.group_folder == source_group):
        db_ops.delete_task(task_id)
        logger.info("Task %s cancelled via IPC from %s", task_id, source_group)
    else:
        logger.warning("Unauthorized cancel_task: task=%s source=%s", task_id, source_group)


async def _handle_refresh_groups(
    source_group: str,
    is_main: bool,
    registered_groups: dict[str, RegisteredGroup],
    deps: IpcDeps,
) -> None:
    """Handle refresh_groups IPC request (main group only).

    Args:
        source_group: Source group folder identity.
        is_main: Main group flag.
        registered_groups: Current registered groups.
        deps: IPC dependencies.
    """
    if not is_main:
        logger.warning("Unauthorized refresh_groups from %s", source_group)
        return
    logger.info("Group metadata refresh requested via IPC from %s", source_group)
    await deps.sync_group_metadata(True)
    available = deps.get_available_groups()
    deps.write_groups_snapshot(source_group, True, available, set(registered_groups.keys()))


def _handle_register_group(
    data: dict[str, Any],
    source_group: str,
    is_main: bool,
    deps: IpcDeps,
) -> None:
    """Handle register_group IPC request (main group only).

    Args:
        data: IPC request data.
        source_group: Source group folder identity.
        is_main: Main group flag.
        deps: IPC dependencies.
    """
    if not is_main:
        logger.warning("Unauthorized register_group from %s", source_group)
        return
    required = ["jid", "name", "folder", "trigger"]
    if not all(data.get(k) for k in required):
        logger.warning("Invalid register_group request — missing fields: %s", data)
        return

    container_config = None
    if data.get("containerConfig"):
        try:
            container_config = ContainerConfig.model_validate(data["containerConfig"])
        except Exception:
            logger.warning("Invalid containerConfig in register_group request")

    from datetime import datetime

    deps.register_group(
        data["jid"],
        RegisteredGroup(
            name=data["name"],
            folder=data["folder"],
            trigger=data["trigger"],
            added_at=datetime.now(UTC).isoformat(),
            container_config=container_config,
            requires_trigger=data.get("requiresTrigger"),
        ),
    )


def _calculate_cron_next_run(schedule_value: str) -> str | None:
    """Calculate the next run time for a cron schedule.

    Args:
        schedule_value: Cron expression string.

    Returns:
        ISO timestamp string, or None if the expression is invalid.
    """
    from datetime import datetime

    try:
        cron = croniter(schedule_value, datetime.now(UTC))
        return cron.get_next(datetime).isoformat()
    except Exception:
        logger.warning("Invalid cron expression: %s", schedule_value)
        return None


def _calculate_interval_next_run(schedule_value: str) -> str | None:
    """Calculate the next run time for an interval schedule.

    Args:
        schedule_value: Interval in milliseconds as a string.

    Returns:
        ISO timestamp string, or None if the value is invalid.
    """
    import time
    from datetime import datetime

    try:
        ms = int(schedule_value)
        if ms <= 0:
            raise ValueError("Interval must be positive")
        next_ts = datetime.fromtimestamp(time.time() + ms / 1000, tz=UTC)
        return next_ts.isoformat()
    except (ValueError, TypeError):
        logger.warning("Invalid interval value: %s", schedule_value)
        return None


def _calculate_once_next_run(schedule_value: str) -> str | None:
    """Calculate the next run time for a one-shot schedule.

    Args:
        schedule_value: ISO timestamp string for the single run.

    Returns:
        ISO timestamp string, or None if the value is invalid.
    """
    from datetime import datetime

    try:
        dt = datetime.fromisoformat(schedule_value)
        return dt.isoformat()
    except ValueError:
        logger.warning("Invalid once timestamp: %s", schedule_value)
        return None


def _calculate_next_run(
    schedule_type: str,
    schedule_value: str,
    timezone_str: str,  # noqa: ARG001
) -> str | None:
    """Calculate the next run time for a scheduled task.

    Args:
        schedule_type: 'cron', 'interval', or 'once'.
        schedule_value: Cron string, milliseconds, or ISO timestamp.
        timezone_str: Timezone string (reserved for future use).

    Returns:
        ISO timestamp string, or None if the input is invalid.
    """
    if schedule_type == "cron":
        return _calculate_cron_next_run(schedule_value)
    if schedule_type == "interval":
        return _calculate_interval_next_run(schedule_value)
    if schedule_type == "once":
        return _calculate_once_next_run(schedule_value)
    return None


def _move_to_errors(file_path: Path, source_group: str, ipc_base: Path) -> None:
    """Move a failed IPC file to the errors directory.

    Args:
        file_path: The file that failed to process.
        source_group: The source group folder name (used in error filename).
        ipc_base: Root IPC directory.
    """
    error_dir = ipc_base / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    try:
        dest = error_dir / f"{source_group}-{file_path.name}"
        file_path.rename(dest)
    except OSError:
        pass


# end src/nanoclaw/ipc.py
