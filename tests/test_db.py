# start tests/test_db.py
"""Tests for nanoclaw.db.operations module.

Replaces src/db.test.ts from TypeScript. Uses an in-memory SQLite database.
"""

from __future__ import annotations

import pytest

from nanoclaw.db.models import Base, create_engine_for_path
from nanoclaw.db import operations as db
from nanoclaw.types import NewMessage, RegisteredGroup, ScheduledTask, TaskRunLog


@pytest.fixture(autouse=True)
def fresh_db(tmp_path) -> None:
    """Reset the database to a fresh in-memory state for each test."""
    db.init_database(":memory:")
    yield
    db._engine = None


class TestChatOperations:
    """Tests for chat metadata operations."""

    def test_store_chat_metadata_with_name(self) -> None:
        """Storing chat metadata with a name saves both fields."""
        db.store_chat_metadata("chat@g.us", "2024-01-01T00:00:00Z", "Test Group")
        chats = db.get_all_chats()
        assert len(chats) == 1
        assert chats[0].jid == "chat@g.us"
        assert chats[0].name == "Test Group"

    def test_store_chat_metadata_without_name(self) -> None:
        """Storing chat metadata without name uses jid as name."""
        db.store_chat_metadata("chat@g.us", "2024-01-01T00:00:00Z")
        chats = db.get_all_chats()
        assert chats[0].jid == "chat@g.us"

    def test_store_chat_metadata_preserves_max_timestamp(self) -> None:
        """Upserting with older timestamp does not overwrite newer one."""
        db.store_chat_metadata("chat@g.us", "2024-01-02T00:00:00Z", "Group")
        db.store_chat_metadata("chat@g.us", "2024-01-01T00:00:00Z", "Group")
        chats = db.get_all_chats()
        assert chats[0].last_message_time == "2024-01-02T00:00:00Z"

    def test_update_chat_name(self) -> None:
        """update_chat_name changes name without altering timestamp for existing rows."""
        db.store_chat_metadata("chat@g.us", "2024-01-01T00:00:00Z", "Old Name")
        db.update_chat_name("chat@g.us", "New Name")
        chats = db.get_all_chats()
        assert chats[0].name == "New Name"

    def test_group_sync_sentinel(self) -> None:
        """get_last_group_sync returns None before first sync, timestamp after."""
        assert db.get_last_group_sync() is None
        db.set_last_group_sync()
        result = db.get_last_group_sync()
        assert result is not None
        assert "T" in result  # ISO format


class TestMessageOperations:
    """Tests for message storage and retrieval."""

    def setup_method(self) -> None:
        """Ensure the chat row exists for FK constraint."""
        db.store_chat_metadata("chat@g.us", "2020-01-01T00:00:00Z", "Test")

    def _make_msg(
        self,
        msg_id: str = "msg1",
        content: str = "hello",
        timestamp: str = "2024-01-01T12:00:00Z",
        is_bot: bool = False,
    ) -> NewMessage:
        return NewMessage(
            id=msg_id,
            chat_jid="chat@g.us",
            sender="alice@s.whatsapp.net",
            sender_name="Alice",
            content=content,
            timestamp=timestamp,
            is_bot_message=is_bot,
        )

    def test_store_and_retrieve_message(self) -> None:
        """Stored messages are returned by get_messages_since."""
        db.store_message(self._make_msg())
        msgs = db.get_messages_since("chat@g.us", "", "Bot")
        assert len(msgs) == 1
        assert msgs[0].content == "hello"

    def test_bot_messages_excluded(self) -> None:
        """Bot messages are excluded from get_messages_since."""
        db.store_message(self._make_msg(msg_id="bot1", is_bot=True, content="Bot: hi"))
        db.store_message(self._make_msg(msg_id="user1", content="user msg"))
        msgs = db.get_messages_since("chat@g.us", "", "Bot")
        assert len(msgs) == 1
        assert msgs[0].content == "user msg"

    def test_bot_prefix_backstop(self) -> None:
        """Messages with bot prefix in content are excluded via backstop."""
        db.store_message(self._make_msg(msg_id="p1", content="Andy: hello"))
        db.store_message(self._make_msg(msg_id="p2", content="hello world"))
        msgs = db.get_messages_since("chat@g.us", "", "Andy")
        assert len(msgs) == 1
        assert msgs[0].id == "p2"

    def test_get_new_messages(self) -> None:
        """get_new_messages filters by timestamp and returns new_timestamp."""
        db.store_message(self._make_msg(msg_id="m1", timestamp="2024-01-01T10:00:00Z"))
        db.store_message(self._make_msg(msg_id="m2", timestamp="2024-01-01T11:00:00Z"))
        msgs, new_ts = db.get_new_messages(["chat@g.us"], "2024-01-01T10:30:00Z", "Bot")
        assert len(msgs) == 1
        assert msgs[0].id == "m2"
        assert new_ts == "2024-01-01T11:00:00Z"

    def test_get_new_messages_empty_jids(self) -> None:
        """Empty JIDs list returns empty result without error."""
        msgs, ts = db.get_new_messages([], "2024-01-01T00:00:00Z", "Bot")
        assert msgs == []
        assert ts == "2024-01-01T00:00:00Z"


class TestScheduledTaskOperations:
    """Tests for scheduled task CRUD operations."""

    def _make_task(self, task_id: str = "task-1") -> ScheduledTask:
        return ScheduledTask(
            id=task_id,
            group_folder="main",
            chat_jid="chat@g.us",
            prompt="Do something",
            schedule_type="cron",
            schedule_value="0 9 * * *",
            context_mode="isolated",
            next_run="2024-01-01T09:00:00Z",
            status="active",
            created_at="2024-01-01T00:00:00Z",
        )

    def test_create_and_get_task(self) -> None:
        """Created task can be retrieved by ID."""
        db.create_task(self._make_task())
        task = db.get_task_by_id("task-1")
        assert task is not None
        assert task.prompt == "Do something"

    def test_get_task_by_id_not_found(self) -> None:
        """Returns None for unknown task ID."""
        assert db.get_task_by_id("nonexistent") is None

    def test_get_due_tasks(self) -> None:
        """Only active tasks with past next_run are returned."""
        past_task = self._make_task("past")
        past_task = past_task.model_copy(update={"next_run": "2020-01-01T00:00:00Z"})
        future_task = self._make_task("future")
        future_task = future_task.model_copy(update={"next_run": "2099-01-01T00:00:00Z"})
        db.create_task(past_task)
        db.create_task(future_task)
        due = db.get_due_tasks()
        assert len(due) == 1
        assert due[0].id == "past"

    def test_update_task_status(self) -> None:
        """update_task changes specified fields only."""
        db.create_task(self._make_task())
        db.update_task("task-1", status="paused")
        task = db.get_task_by_id("task-1")
        assert task is not None
        assert task.status == "paused"
        assert task.prompt == "Do something"  # unchanged

    def test_delete_task(self) -> None:
        """Deleted task is not found by get_task_by_id."""
        db.create_task(self._make_task())
        db.delete_task("task-1")
        assert db.get_task_by_id("task-1") is None

    def test_update_task_after_run_completes_once(self) -> None:
        """update_task_after_run sets status=completed when next_run is None."""
        db.create_task(self._make_task())
        db.update_task_after_run("task-1", None, "Done")
        task = db.get_task_by_id("task-1")
        assert task is not None
        assert task.status == "completed"
        assert task.last_result == "Done"

    def test_log_task_run(self) -> None:
        """log_task_run inserts without error."""
        db.create_task(self._make_task())
        db.log_task_run(TaskRunLog(
            task_id="task-1",
            run_at="2024-01-01T09:00:00Z",
            duration_ms=1500,
            status="success",
            result="OK",
        ))


class TestRouterState:
    """Tests for router_state key-value operations."""

    def test_set_and_get(self) -> None:
        """Set value is returned by get."""
        db.set_router_state("last_timestamp", "2024-01-01T00:00:00Z")
        assert db.get_router_state("last_timestamp") == "2024-01-01T00:00:00Z"

    def test_get_missing_key_returns_none(self) -> None:
        """Missing key returns None."""
        assert db.get_router_state("nonexistent") is None

    def test_overwrite(self) -> None:
        """Setting same key twice uses the newer value."""
        db.set_router_state("k", "v1")
        db.set_router_state("k", "v2")
        assert db.get_router_state("k") == "v2"


class TestSessionOperations:
    """Tests for Claude session ID persistence."""

    def test_set_and_get_session(self) -> None:
        """Stored session ID is returned by get_session."""
        db.set_session("main", "sess-abc")
        assert db.get_session("main") == "sess-abc"

    def test_get_session_not_found(self) -> None:
        """Unknown group returns None."""
        assert db.get_session("unknown") is None

    def test_get_all_sessions(self) -> None:
        """get_all_sessions returns all stored sessions."""
        db.set_session("main", "s1")
        db.set_session("group2", "s2")
        sessions = db.get_all_sessions()
        assert sessions == {"main": "s1", "group2": "s2"}


class TestRegisteredGroupOperations:
    """Tests for registered group CRUD operations."""

    def _make_group(self) -> RegisteredGroup:
        return RegisteredGroup(
            name="Test Group",
            folder="test-group",
            trigger="@Andy",
            added_at="2024-01-01T00:00:00Z",
        )

    def test_set_and_get_registered_group(self) -> None:
        """Stored group is returned by get_registered_group."""
        db.set_registered_group("group@g.us", self._make_group())
        group = db.get_registered_group("group@g.us")
        assert group is not None
        assert group.name == "Test Group"
        assert group.folder == "test-group"

    def test_get_registered_group_not_found(self) -> None:
        """Unknown JID returns None."""
        assert db.get_registered_group("unknown@g.us") is None

    def test_get_all_registered_groups(self) -> None:
        """get_all_registered_groups returns all registered groups."""
        db.set_registered_group("g1@g.us", self._make_group())
        group2 = self._make_group().model_copy(update={"folder": "group2", "name": "G2"})
        db.set_registered_group("g2@g.us", group2)
        all_groups = db.get_all_registered_groups()
        assert len(all_groups) == 2
        assert "g1@g.us" in all_groups
        assert "g2@g.us" in all_groups


# ---------------------------------------------------------------------------
# Additional tests to cover missing lines
# ---------------------------------------------------------------------------


class TestGetEngineNotInitialized:
    """Tests for get_engine() when the database is not yet initialised (lines 59, 75)."""

    def test_get_engine_raises_when_not_initialized(self) -> None:
        """get_engine() raises RuntimeError when init_database() has not been called (line 75)."""
        # Temporarily clear the engine
        original = db._engine
        db._engine = None
        try:
            with pytest.raises(RuntimeError, match="Database not initialized"):
                db.get_engine()
        finally:
            db._engine = original

    def test_init_database_creates_parent_dir(self, tmp_path) -> None:
        """init_database() creates parent directories for a new file-based database (line 59)."""
        db_path = str(tmp_path / "nested" / "subdir" / "nanoclaw.db")
        db.init_database(db_path)
        assert (tmp_path / "nested" / "subdir" / "nanoclaw.db").exists()
        # Reset back to :memory: for isolation
        db.init_database(":memory:")


class TestGetTasksForGroup:
    """Tests for get_tasks_for_group (lines 392-399)."""

    def _make_task(self, task_id: str, group_folder: str) -> ScheduledTask:
        return ScheduledTask(
            id=task_id,
            group_folder=group_folder,
            chat_jid=f"{group_folder}@g.us",
            prompt="Do something",
            schedule_type="cron",
            schedule_value="0 9 * * *",
            context_mode="isolated",
            next_run="2024-01-01T09:00:00Z",
            status="active",
            created_at="2024-01-01T00:00:00Z",
        )

    def test_get_tasks_for_group_returns_only_matching(self) -> None:
        """get_tasks_for_group returns only tasks belonging to the specified group (lines 392-399)."""
        db.create_task(self._make_task("t1", "main"))
        db.create_task(self._make_task("t2", "friends"))
        db.create_task(self._make_task("t3", "main"))

        tasks = db.get_tasks_for_group("main")
        assert len(tasks) == 2
        assert all(t.group_folder == "main" for t in tasks)

    def test_get_tasks_for_group_returns_empty_when_none(self) -> None:
        """get_tasks_for_group returns empty list for a group with no tasks."""
        tasks = db.get_tasks_for_group("nonexistent")
        assert tasks == []


class TestGetAllTasks:
    """Tests for get_all_tasks (lines 408-413)."""

    def _make_task(self, task_id: str) -> ScheduledTask:
        return ScheduledTask(
            id=task_id,
            group_folder="main",
            chat_jid="chat@g.us",
            prompt="Prompt",
            schedule_type="cron",
            schedule_value="0 * * * *",
            status="active",
            created_at="2024-01-01T00:00:00Z",
        )

    def test_get_all_tasks_returns_all(self) -> None:
        """get_all_tasks returns every task in the database (lines 408-413)."""
        db.create_task(self._make_task("a1"))
        db.create_task(self._make_task("a2"))
        db.create_task(self._make_task("a3"))

        tasks = db.get_all_tasks()
        assert len(tasks) == 3

    def test_get_all_tasks_empty_db(self) -> None:
        """get_all_tasks returns an empty list when no tasks exist."""
        tasks = db.get_all_tasks()
        assert tasks == []


class TestUpdateTaskFields:
    """Tests for update_task individual field updates (lines 457, 459, 461, 463, 467)."""

    def setup_method(self) -> None:
        """Insert a base task for each test."""
        self._task = ScheduledTask(
            id="upd-1",
            group_folder="main",
            chat_jid="chat@g.us",
            prompt="original prompt",
            schedule_type="cron",
            schedule_value="0 9 * * *",
            context_mode="isolated",
            next_run="2024-06-01T09:00:00Z",
            status="active",
            created_at="2024-01-01T00:00:00Z",
        )
        db.create_task(self._task)

    def test_update_task_prompt(self) -> None:
        """update_task changes only the prompt field (line 457)."""
        db.update_task("upd-1", prompt="new prompt")
        t = db.get_task_by_id("upd-1")
        assert t is not None
        assert t.prompt == "new prompt"
        assert t.status == "active"  # unchanged

    def test_update_task_schedule_type(self) -> None:
        """update_task changes only the schedule_type field (line 459)."""
        db.update_task("upd-1", schedule_type="interval")
        t = db.get_task_by_id("upd-1")
        assert t is not None
        assert t.schedule_type == "interval"

    def test_update_task_schedule_value(self) -> None:
        """update_task changes only the schedule_value field (line 461)."""
        db.update_task("upd-1", schedule_value="3600000")
        t = db.get_task_by_id("upd-1")
        assert t is not None
        assert t.schedule_value == "3600000"

    def test_update_task_next_run(self) -> None:
        """update_task changes only the next_run field (line 463)."""
        db.update_task("upd-1", next_run="2025-01-01T00:00:00Z")
        t = db.get_task_by_id("upd-1")
        assert t is not None
        assert t.next_run == "2025-01-01T00:00:00Z"

    def test_update_task_no_fields_is_noop(self) -> None:
        """update_task with no arguments returns without touching the DB (line 467)."""
        db.update_task("upd-1")  # No fields supplied — must not raise
        t = db.get_task_by_id("upd-1")
        assert t is not None
        assert t.prompt == "original prompt"


class TestRowToGroupContainerConfig:
    """Tests for _row_to_group with invalid container_config JSON (lines 718-721)."""

    def test_invalid_container_config_json_yields_none(self) -> None:
        """A malformed container_config JSON in the DB row logs a warning and returns None (lines 718-721)."""
        # Insert a group with a well-known JID
        db.set_registered_group(
            "cfg@g.us",
            RegisteredGroup(
                name="Config Group",
                folder="cfg-group",
                trigger="@Andy",
                added_at="2024-01-01T00:00:00Z",
            ),
        )

        # Manually corrupt the container_config in the DB
        from nanoclaw.db.models import RegisteredGroup as RGRow
        from sqlalchemy.orm import Session

        with Session(db.get_engine()) as session:
            row = session.get(RGRow, "cfg@g.us")
            assert row is not None
            row.container_config = "NOT-VALID-JSON{{{"
            session.commit()

        # Now get_registered_group should return the group with container_config=None
        group = db.get_registered_group("cfg@g.us")
        assert group is not None
        assert group.container_config is None
# end tests/test_db.py
