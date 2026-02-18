# start src/nanoclaw/db/operations.py
"""Database operations for NanoClaw.

Replaces all functions from src/db.ts using SQLAlchemy Core/ORM.
No raw SQL strings — all queries use SQLAlchemy constructs.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, select, text, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from nanoclaw.db.models import (
    Base,
    Chat,
    Message,
    RouterState,
    SessionRecord,
    create_engine_for_path,
)
from nanoclaw.db.models import (
    RegisteredGroup as RegisteredGroupRow,
)
from nanoclaw.db.models import (
    ScheduledTask as ScheduledTaskRow,
)
from nanoclaw.db.models import (
    TaskRunLog as TaskRunLogRow,
)
from nanoclaw.types import (
    ChatInfo,
    ContainerConfig,
    NewMessage,
    RegisteredGroup,
    ScheduledTask,
    TaskRunLog,
)

logger = logging.getLogger(__name__)

_engine: Engine | None = None


def init_database(db_path: str) -> None:
    """Initialize the SQLite database and create tables if needed.

    Args:
        db_path: Absolute path to the SQLite database file.
            Use ':memory:' for testing.
    """
    global _engine
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine_for_path(db_path)
    Base.metadata.create_all(_engine)
    logger.info("Database initialized at %s", db_path)


def get_engine() -> Engine:
    """Return the active database engine.

    Returns:
        The SQLAlchemy Engine instance.

    Raises:
        RuntimeError: If init_database() has not been called.
    """
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _engine


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


# --- Chat operations ---


def store_chat_metadata(chat_jid: str, timestamp: str, name: str | None = None) -> None:
    """Store or update chat metadata without message content.

    Used for all chats to enable group discovery. Preserves the most
    recent timestamp using MAX logic on conflict.

    Args:
        chat_jid: WhatsApp JID of the chat.
        timestamp: ISO timestamp string.
        name: Optional display name for the chat.
    """
    engine = get_engine()
    with Session(engine) as session:
        stmt = sqlite_insert(Chat).values(
            jid=chat_jid,
            name=name or chat_jid,
            last_message_time=timestamp,
        )
        if name:
            stmt = stmt.on_conflict_do_update(
                index_elements=["jid"],
                set_={
                    "name": stmt.excluded.name,
                    "last_message_time": text(
                        "MAX(chats.last_message_time, excluded.last_message_time)"
                    ),
                },
            )
        else:
            stmt = stmt.on_conflict_do_update(
                index_elements=["jid"],
                set_={
                    "last_message_time": text(
                        "MAX(chats.last_message_time, excluded.last_message_time)"
                    ),
                },
            )
        session.execute(stmt)
        session.commit()


def update_chat_name(chat_jid: str, name: str) -> None:
    """Update the display name for an existing chat, or insert if new.

    Does NOT update the timestamp for existing chats (used during group sync).

    Args:
        chat_jid: WhatsApp JID of the chat.
        name: New display name.
    """
    engine = get_engine()
    with Session(engine) as session:
        stmt = sqlite_insert(Chat).values(
            jid=chat_jid,
            name=name,
            last_message_time=_now_iso(),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["jid"],
            set_={"name": stmt.excluded.name},
        )
        session.execute(stmt)
        session.commit()


def get_all_chats() -> list[ChatInfo]:
    """Return all known chats ordered by most recent activity.

    Returns:
        List of ChatInfo objects sorted by last_message_time descending.
    """
    engine = get_engine()
    with Session(engine) as session:
        rows = session.scalars(select(Chat).order_by(Chat.last_message_time.desc())).all()
        return [
            ChatInfo(
                jid=r.jid,
                name=r.name or r.jid,
                last_message_time=r.last_message_time or "",
            )
            for r in rows
        ]


def get_last_group_sync() -> str | None:
    """Return the timestamp of the last WhatsApp group metadata sync.

    Uses a sentinel entry in the chats table (jid='__group_sync__').

    Returns:
        ISO timestamp string, or None if never synced.
    """
    engine = get_engine()
    with Session(engine) as session:
        row = session.scalar(select(Chat).where(Chat.jid == "__group_sync__"))
        return row.last_message_time if row else None


def set_last_group_sync() -> None:
    """Record the current time as the last group metadata sync timestamp."""
    engine = get_engine()
    now = _now_iso()
    with Session(engine) as session:
        stmt = sqlite_insert(Chat).values(
            jid="__group_sync__",
            name="__group_sync__",
            last_message_time=now,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["jid"],
            set_={"last_message_time": stmt.excluded.last_message_time},
        )
        session.execute(stmt)
        session.commit()


# --- Message operations ---


def store_message(msg: NewMessage) -> None:
    """Store or replace a message in the messages table.

    Only call for registered groups where message history is needed.

    Args:
        msg: The inbound message to store.
    """
    engine = get_engine()
    with Session(engine) as session:
        stmt = sqlite_insert(Message).values(
            id=msg.id,
            chat_jid=msg.chat_jid,
            sender=msg.sender,
            sender_name=msg.sender_name,
            content=msg.content,
            timestamp=msg.timestamp,
            is_from_me=1 if msg.is_from_me else 0,
            is_bot_message=1 if msg.is_bot_message else 0,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["id", "chat_jid"],
            set_={
                "sender": stmt.excluded.sender,
                "sender_name": stmt.excluded.sender_name,
                "content": stmt.excluded.content,
                "timestamp": stmt.excluded.timestamp,
                "is_from_me": stmt.excluded.is_from_me,
                "is_bot_message": stmt.excluded.is_bot_message,
            },
        )
        session.execute(stmt)
        session.commit()


def get_new_messages(
    jids: list[str],
    last_timestamp: str,
    bot_prefix: str,
) -> tuple[list[NewMessage], str]:
    """Fetch messages newer than last_timestamp for the given JIDs.

    Excludes bot messages (using both the is_bot_message flag and the
    content prefix as a backstop for pre-migration rows).

    Args:
        jids: List of chat JIDs to query.
        last_timestamp: Only return messages with timestamp > this value.
        bot_prefix: Assistant name prefix (e.g., 'Andy') for backstop filtering.

    Returns:
        Tuple of (messages list, new_timestamp string). new_timestamp is the
        timestamp of the most recent returned message, or last_timestamp if none.
    """
    if not jids:
        return [], last_timestamp

    engine = get_engine()
    with Session(engine) as session:
        rows = session.scalars(
            select(Message)
            .where(
                Message.timestamp > last_timestamp,
                Message.chat_jid.in_(jids),
                Message.is_bot_message == 0,
                ~Message.content.like(f"{bot_prefix}:%"),
            )
            .order_by(Message.timestamp)
        ).all()

        messages = [
            NewMessage(
                id=r.id,
                chat_jid=r.chat_jid,
                sender=r.sender or "",
                sender_name=r.sender_name or "",
                content=r.content or "",
                timestamp=r.timestamp or "",
                is_from_me=bool(r.is_from_me),
                is_bot_message=bool(r.is_bot_message),
            )
            for r in rows
        ]

        new_timestamp = last_timestamp
        for m in messages:
            if m.timestamp > new_timestamp:
                new_timestamp = m.timestamp

        return messages, new_timestamp


def get_messages_since(
    chat_jid: str,
    since_timestamp: str,
    bot_prefix: str,
) -> list[NewMessage]:
    """Fetch all messages for a chat since a given timestamp.

    Args:
        chat_jid: The chat JID to query.
        since_timestamp: Only return messages with timestamp > this value.
        bot_prefix: Assistant name prefix for backstop bot-message filtering.

    Returns:
        List of messages ordered by timestamp ascending.
    """
    engine = get_engine()
    with Session(engine) as session:
        rows = session.scalars(
            select(Message)
            .where(
                Message.chat_jid == chat_jid,
                Message.timestamp > since_timestamp,
                Message.is_bot_message == 0,
                ~Message.content.like(f"{bot_prefix}:%"),
            )
            .order_by(Message.timestamp)
        ).all()

        return [
            NewMessage(
                id=r.id,
                chat_jid=r.chat_jid,
                sender=r.sender or "",
                sender_name=r.sender_name or "",
                content=r.content or "",
                timestamp=r.timestamp or "",
                is_from_me=bool(r.is_from_me),
                is_bot_message=bool(r.is_bot_message),
            )
            for r in rows
        ]


# --- Scheduled task operations ---


def create_task(task: ScheduledTask) -> None:
    """Insert a new scheduled task.

    Args:
        task: The task to create.
    """
    engine = get_engine()
    with Session(engine) as session:
        row = ScheduledTaskRow(
            id=task.id,
            group_folder=task.group_folder,
            chat_jid=task.chat_jid,
            prompt=task.prompt,
            schedule_type=task.schedule_type,
            schedule_value=task.schedule_value,
            context_mode=task.context_mode,
            next_run=task.next_run,
            status=task.status,
            created_at=task.created_at,
        )
        session.add(row)
        session.commit()


def get_task_by_id(task_id: str) -> ScheduledTask | None:
    """Fetch a scheduled task by its ID.

    Args:
        task_id: The task ID to look up.

    Returns:
        ScheduledTask if found, None otherwise.
    """
    engine = get_engine()
    with Session(engine) as session:
        row = session.get(ScheduledTaskRow, task_id)
        return _row_to_task(row) if row else None


def get_tasks_for_group(group_folder: str) -> list[ScheduledTask]:
    """Fetch all tasks for a specific group, ordered by creation time.

    Args:
        group_folder: The group folder name.

    Returns:
        List of ScheduledTask objects.
    """
    engine = get_engine()
    with Session(engine) as session:
        rows = session.scalars(
            select(ScheduledTaskRow)
            .where(ScheduledTaskRow.group_folder == group_folder)
            .order_by(ScheduledTaskRow.created_at.desc())
        ).all()
        return [_row_to_task(r) for r in rows]


def get_all_tasks() -> list[ScheduledTask]:
    """Fetch all scheduled tasks ordered by creation time descending.

    Returns:
        List of all ScheduledTask objects.
    """
    engine = get_engine()
    with Session(engine) as session:
        rows = session.scalars(
            select(ScheduledTaskRow).order_by(ScheduledTaskRow.created_at.desc())
        ).all()
        return [_row_to_task(r) for r in rows]


def get_due_tasks() -> list[ScheduledTask]:
    """Fetch all active tasks whose next_run is now or in the past.

    Returns:
        List of due ScheduledTask objects ordered by next_run ascending.
    """
    now = _now_iso()
    engine = get_engine()
    with Session(engine) as session:
        rows = session.scalars(
            select(ScheduledTaskRow)
            .where(
                ScheduledTaskRow.status == "active",
                ScheduledTaskRow.next_run.is_not(None),
                ScheduledTaskRow.next_run <= now,
            )
            .order_by(ScheduledTaskRow.next_run)
        ).all()
        return [_row_to_task(r) for r in rows]


def update_task(
    task_id: str,
    prompt: str | None = None,
    schedule_type: str | None = None,
    schedule_value: str | None = None,
    next_run: str | None = None,
    status: str | None = None,
) -> None:
    """Update specific fields of a scheduled task.

    Args:
        task_id: The task ID to update.
        prompt: New prompt text, or None to leave unchanged.
        schedule_type: New schedule type, or None to leave unchanged.
        schedule_value: New schedule value, or None to leave unchanged.
        next_run: New next_run timestamp, or None to leave unchanged.
        status: New status, or None to leave unchanged.
    """
    values: dict[str, str] = {}
    if prompt is not None:
        values["prompt"] = prompt
    if schedule_type is not None:
        values["schedule_type"] = schedule_type
    if schedule_value is not None:
        values["schedule_value"] = schedule_value
    if next_run is not None:
        values["next_run"] = next_run
    if status is not None:
        values["status"] = status
    if not values:
        return

    engine = get_engine()
    with Session(engine) as session:
        session.execute(
            update(ScheduledTaskRow).where(ScheduledTaskRow.id == task_id).values(**values)
        )
        session.commit()


def delete_task(task_id: str) -> None:
    """Delete a scheduled task and all its run logs.

    Args:
        task_id: The task ID to delete.
    """
    engine = get_engine()
    with Session(engine) as session:
        session.execute(delete(TaskRunLogRow).where(TaskRunLogRow.task_id == task_id))
        session.execute(delete(ScheduledTaskRow).where(ScheduledTaskRow.id == task_id))
        session.commit()


def update_task_after_run(
    task_id: str,
    next_run: str | None,
    last_result: str,
) -> None:
    """Update a task after it has been executed.

    Sets last_run to now, updates next_run, records last_result.
    If next_run is None, marks the task as 'completed'.

    Args:
        task_id: The task to update.
        next_run: Next scheduled run time, or None if task is done.
        last_result: Summary of the result for display purposes.
    """
    now = _now_iso()
    engine = get_engine()
    with Session(engine) as session:
        new_status = "completed" if next_run is None else None
        values: dict[str, str | None] = {
            "next_run": next_run,
            "last_run": now,
            "last_result": last_result,
        }
        if new_status:
            values["status"] = new_status
        session.execute(
            update(ScheduledTaskRow).where(ScheduledTaskRow.id == task_id).values(**values)
        )
        session.commit()


def log_task_run(log: TaskRunLog) -> None:
    """Insert a task run log entry.

    Args:
        log: The run log to record.
    """
    engine = get_engine()
    with Session(engine) as session:
        row = TaskRunLogRow(
            task_id=log.task_id,
            run_at=log.run_at,
            duration_ms=log.duration_ms,
            status=log.status,
            result=log.result,
            error=log.error,
        )
        session.add(row)
        session.commit()


# --- Router state operations ---


def get_router_state(key: str) -> str | None:
    """Fetch a value from the router_state key-value table.

    Args:
        key: The state key to look up.

    Returns:
        The stored value string, or None if not found.
    """
    engine = get_engine()
    with Session(engine) as session:
        row = session.get(RouterState, key)
        return row.value if row else None


def set_router_state(key: str, value: str) -> None:
    """Set a value in the router_state key-value table.

    Args:
        key: The state key.
        value: The value to store (use JSON.dumps for complex values).
    """
    engine = get_engine()
    with Session(engine) as session:
        stmt = sqlite_insert(RouterState).values(key=key, value=value)
        stmt = stmt.on_conflict_do_update(
            index_elements=["key"],
            set_={"value": stmt.excluded.value},
        )
        session.execute(stmt)
        session.commit()


# --- Session operations ---


def get_session(group_folder: str) -> str | None:
    """Fetch the stored Claude session ID for a group.

    Args:
        group_folder: The group folder name.

    Returns:
        The session ID string, or None if no session is stored.
    """
    engine = get_engine()
    with Session(engine) as session:
        row = session.get(SessionRecord, group_folder)
        return row.session_id if row else None


def set_session(group_folder: str, session_id: str) -> None:
    """Store or replace the Claude session ID for a group.

    Args:
        group_folder: The group folder name.
        session_id: The Claude session ID to persist.
    """
    engine = get_engine()
    with Session(engine) as session:
        stmt = sqlite_insert(SessionRecord).values(group_folder=group_folder, session_id=session_id)
        stmt = stmt.on_conflict_do_update(
            index_elements=["group_folder"],
            set_={"session_id": stmt.excluded.session_id},
        )
        session.execute(stmt)
        session.commit()


def get_all_sessions() -> dict[str, str]:
    """Fetch all stored Claude session IDs.

    Returns:
        Dict mapping group_folder -> session_id.
    """
    engine = get_engine()
    with Session(engine) as session:
        rows = session.scalars(select(SessionRecord)).all()
        return {r.group_folder: r.session_id for r in rows}


# --- Registered group operations ---


def get_registered_group(jid: str) -> RegisteredGroup | None:
    """Fetch a registered group by its WhatsApp JID.

    Args:
        jid: The WhatsApp JID of the group.

    Returns:
        RegisteredGroup if found, None otherwise.
    """
    engine = get_engine()
    with Session(engine) as session:
        row = session.get(RegisteredGroupRow, jid)
        return _row_to_group(row) if row else None


def set_registered_group(jid: str, group: RegisteredGroup) -> None:
    """Insert or replace a registered group record.

    Args:
        jid: The WhatsApp JID.
        group: The RegisteredGroup data to store.
    """
    engine = get_engine()
    with Session(engine) as session:
        container_config_json = (
            group.container_config.model_dump_json() if group.container_config else None
        )
        requires_trigger_int = 1 if group.requires_trigger is None or group.requires_trigger else 0
        stmt = sqlite_insert(RegisteredGroupRow).values(
            jid=jid,
            name=group.name,
            folder=group.folder,
            trigger_pattern=group.trigger,
            added_at=group.added_at,
            container_config=container_config_json,
            requires_trigger=requires_trigger_int,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["jid"],
            set_={
                "name": stmt.excluded.name,
                "folder": stmt.excluded.folder,
                "trigger_pattern": stmt.excluded.trigger_pattern,
                "added_at": stmt.excluded.added_at,
                "container_config": stmt.excluded.container_config,
                "requires_trigger": stmt.excluded.requires_trigger,
            },
        )
        session.execute(stmt)
        session.commit()


def get_all_registered_groups() -> dict[str, RegisteredGroup]:
    """Fetch all registered groups.

    Returns:
        Dict mapping JID -> RegisteredGroup.
    """
    engine = get_engine()
    with Session(engine) as session:
        rows = session.scalars(select(RegisteredGroupRow)).all()
        return {r.jid: _row_to_group(r) for r in rows}


# --- Private helpers ---


def _row_to_task(row: ScheduledTaskRow) -> ScheduledTask:
    """Convert a ScheduledTaskRow ORM object to a ScheduledTask domain model."""
    return ScheduledTask(
        id=row.id,
        group_folder=row.group_folder,
        chat_jid=row.chat_jid,
        prompt=row.prompt,
        schedule_type=row.schedule_type,
        schedule_value=row.schedule_value,
        context_mode=row.context_mode,
        next_run=row.next_run,
        last_run=row.last_run,
        last_result=row.last_result,
        status=row.status,
        created_at=row.created_at,
    )


def _row_to_group(row: RegisteredGroupRow) -> RegisteredGroup:
    """Convert a RegisteredGroupRow ORM object to a RegisteredGroup domain model."""
    container_config = None
    if row.container_config:
        try:
            container_config = ContainerConfig.model_validate_json(row.container_config)
        except Exception:
            logger.warning(
                "Failed to parse container_config for group %s",
                row.jid if hasattr(row, "jid") else "?",
            )

    requires_trigger: bool | None = None
    if row.requires_trigger is not None:
        requires_trigger = bool(row.requires_trigger)

    return RegisteredGroup(
        name=row.name,
        folder=row.folder,
        trigger=row.trigger_pattern,
        added_at=row.added_at,
        container_config=container_config,
        requires_trigger=requires_trigger,
    )


# end src/nanoclaw/db/operations.py
