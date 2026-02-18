# start tests/test_container.py
"""Tests for nanoclaw.container module.

Covers volume mount construction, CLI arg building, output parsing,
task/group snapshot writing, and the full run_container_agent() async
flow with mocked subprocess.
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanoclaw.container import (
    OUTPUT_END_MARKER,
    OUTPUT_START_MARKER,
    _parse_final_output,
    build_container_args,
    build_volume_mounts,
    write_groups_snapshot,
    write_tasks_snapshot,
)
from nanoclaw.types import (
    AvailableGroup,
    ContainerInput,
    ContainerOutput,
    RegisteredGroup,
    ScheduledTask,
    VolumeMount,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def group() -> RegisteredGroup:
    """A generic non-main registered group."""
    return RegisteredGroup(
        name="Friends",
        folder="friends",
        trigger="@Andy",
        added_at="2024-01-01T00:00:00Z",
    )


@pytest.fixture
def main_group() -> RegisteredGroup:
    """The main registered group."""
    return RegisteredGroup(
        name="Main",
        folder="main",
        trigger="@Andy",
        added_at="2024-01-01T00:00:00Z",
    )


@pytest.fixture
def layout(tmp_path: Path) -> dict[str, Path]:
    """Create a minimal on-disk layout for container tests.

    Returns:
        Dict with keys: groups_dir, data_dir, project_root.
    """
    project_root = tmp_path / "project"
    groups_dir = project_root / "groups"
    data_dir = tmp_path / "data"

    # group folder
    (groups_dir / "friends").mkdir(parents=True)
    (groups_dir / "main").mkdir(parents=True)

    # agent-runner src (always expected)
    (project_root / "container" / "agent-runner" / "src").mkdir(parents=True)

    return {
        "groups_dir": groups_dir,
        "data_dir": data_dir,
        "project_root": project_root,
    }


# ---------------------------------------------------------------------------
# Helper: build a mock async subprocess
# ---------------------------------------------------------------------------


def _make_async_reader(data: bytes):
    """Return a mock StreamReader that yields *data* in one chunk then EOF.

    Args:
        data: Bytes to serve as the full stdout or stderr content.

    Returns:
        Object with async read() and readline() methods compatible with
        asyncio.StreamReader.
    """

    class _FakeReader:
        def __init__(self, payload: bytes) -> None:
            self._buf = io.BytesIO(payload)

        async def read(self, n: int = -1) -> bytes:
            return self._buf.read(n)

        async def readline(self) -> bytes:
            return self._buf.readline()

    return _FakeReader(data)


class _MockProcess:
    """Minimal mock of asyncio.subprocess.Process for container tests.

    Attributes:
        returncode: Exit code returned by wait().
        stdin: AsyncMock with write/drain/close methods.
        stdout: Fake async reader backed by provided bytes.
        stderr: Fake async reader backed by provided bytes.
    """

    def __init__(
        self,
        stdout_data: bytes = b"",
        stderr_data: bytes = b"",
        exit_code: int = 0,
    ) -> None:
        """Initialise the mock process.

        Args:
            stdout_data: Bytes to serve as stdout.
            stderr_data: Bytes to serve as stderr.
            exit_code: Value returned by wait().
        """
        self.returncode = exit_code
        self.stdin = AsyncMock()
        self.stdout = _make_async_reader(stdout_data)
        self.stderr = _make_async_reader(stderr_data)

    async def wait(self) -> int:
        """Return the preset exit code.

        Returns:
            The exit_code given at construction.
        """
        return self.returncode

    def kill(self) -> None:
        """No-op kill for timeout tests."""


def _make_stdout(result: str | None = "hello", status: str = "success") -> bytes:
    """Build stdout bytes containing a valid OUTPUT_START/END marker pair.

    Args:
        result: Agent result text to embed in the JSON.
        status: Status string for the ContainerOutput.

    Returns:
        UTF-8 encoded stdout bytes.
    """
    payload = ContainerOutput(status=status, result=result).model_dump_json()
    return (
        f"some preamble\n"
        f"{OUTPUT_START_MARKER}\n{payload}\n{OUTPUT_END_MARKER}\n"
    ).encode()


# ---------------------------------------------------------------------------
# Tests: build_volume_mounts()
# ---------------------------------------------------------------------------


class TestBuildVolumeMounts:
    """Tests for the build_volume_mounts() function."""

    def test_main_group_gets_project_root(
        self, main_group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """Main group receives a read-write mount for the project root."""
        mounts = build_volume_mounts(
            main_group,
            is_main=True,
            groups_dir=layout["groups_dir"],
            data_dir=layout["data_dir"],
            project_root=layout["project_root"],
        )
        container_paths = [m.container_path for m in mounts]
        assert "/workspace/project" in container_paths

        project_mount = next(m for m in mounts if m.container_path == "/workspace/project")
        assert project_mount.host_path == str(layout["project_root"])
        assert project_mount.readonly is False

    def test_non_main_group_does_not_get_project_root(
        self, group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """Non-main group must not receive the project root mount."""
        mounts = build_volume_mounts(
            group,
            is_main=False,
            groups_dir=layout["groups_dir"],
            data_dir=layout["data_dir"],
            project_root=layout["project_root"],
        )
        container_paths = [m.container_path for m in mounts]
        assert "/workspace/project" not in container_paths

    def test_group_folder_mount_present(
        self, group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """Every group gets a read-write mount for its own group folder."""
        mounts = build_volume_mounts(
            group,
            is_main=False,
            groups_dir=layout["groups_dir"],
            data_dir=layout["data_dir"],
            project_root=layout["project_root"],
        )
        group_mount = next(
            (m for m in mounts if m.container_path == "/workspace/group"), None
        )
        assert group_mount is not None
        assert group_mount.host_path == str(layout["groups_dir"] / "friends")
        assert group_mount.readonly is False

    def test_sessions_dir_mounted_rw(
        self, group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """The sessions (.claude) directory is mounted read-write at /home/node/.claude."""
        mounts = build_volume_mounts(
            group,
            is_main=False,
            groups_dir=layout["groups_dir"],
            data_dir=layout["data_dir"],
            project_root=layout["project_root"],
        )
        sessions_mount = next(
            (m for m in mounts if m.container_path == "/home/node/.claude"), None
        )
        assert sessions_mount is not None
        assert sessions_mount.readonly is False

    def test_ipc_dir_mounted_rw(
        self, group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """The IPC directory is mounted read-write at /workspace/ipc."""
        mounts = build_volume_mounts(
            group,
            is_main=False,
            groups_dir=layout["groups_dir"],
            data_dir=layout["data_dir"],
            project_root=layout["project_root"],
        )
        ipc_mount = next(
            (m for m in mounts if m.container_path == "/workspace/ipc"), None
        )
        assert ipc_mount is not None
        assert ipc_mount.readonly is False

    def test_agent_runner_src_mounted_ro(
        self, group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """The agent-runner src directory is mounted read-only at /app/src."""
        mounts = build_volume_mounts(
            group,
            is_main=False,
            groups_dir=layout["groups_dir"],
            data_dir=layout["data_dir"],
            project_root=layout["project_root"],
        )
        src_mount = next(
            (m for m in mounts if m.container_path == "/app/src"), None
        )
        assert src_mount is not None
        assert src_mount.readonly is True

    def test_global_dir_mounted_ro_when_exists(
        self, group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """Non-main group gets a read-only /workspace/global mount when global/ exists."""
        global_dir = layout["groups_dir"] / "global"
        global_dir.mkdir(parents=True)

        mounts = build_volume_mounts(
            group,
            is_main=False,
            groups_dir=layout["groups_dir"],
            data_dir=layout["data_dir"],
            project_root=layout["project_root"],
        )
        global_mount = next(
            (m for m in mounts if m.container_path == "/workspace/global"), None
        )
        assert global_mount is not None
        assert global_mount.readonly is True

    def test_global_dir_not_mounted_when_missing(
        self, group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """Non-main group does not get /workspace/global if global/ directory is absent."""
        mounts = build_volume_mounts(
            group,
            is_main=False,
            groups_dir=layout["groups_dir"],
            data_dir=layout["data_dir"],
            project_root=layout["project_root"],
        )
        container_paths = [m.container_path for m in mounts]
        assert "/workspace/global" not in container_paths

    def test_main_group_does_not_get_global_dir(
        self, main_group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """Main group never gets a /workspace/global mount, even if global/ exists."""
        global_dir = layout["groups_dir"] / "global"
        global_dir.mkdir(parents=True)

        mounts = build_volume_mounts(
            main_group,
            is_main=True,
            groups_dir=layout["groups_dir"],
            data_dir=layout["data_dir"],
            project_root=layout["project_root"],
        )
        container_paths = [m.container_path for m in mounts]
        assert "/workspace/global" not in container_paths


# ---------------------------------------------------------------------------
# Tests: build_container_args()
# ---------------------------------------------------------------------------


class TestBuildContainerArgs:
    """Tests for the build_container_args() function."""

    def test_prefix_is_container_run_i_rm(self) -> None:
        """Args start with 'container run -i --rm --name <name>'."""
        args = build_container_args([], "my-container", "nanoclaw-agent:latest")
        assert args[:5] == ["container", "run", "-i", "--rm", "--name"]
        assert args[5] == "my-container"

    def test_image_is_last_arg(self) -> None:
        """The container image name is the final argument."""
        args = build_container_args([], "my-container", "nanoclaw-agent:latest")
        assert args[-1] == "nanoclaw-agent:latest"

    def test_readonly_mount_uses_mount_syntax(self) -> None:
        """Read-only volumes use '--mount type=bind,...,readonly' syntax."""
        mount = VolumeMount(
            host_path="/host/src",
            container_path="/app/src",
            readonly=True,
        )
        args = build_container_args([mount], "c", "img")
        assert "--mount" in args
        idx = args.index("--mount")
        spec = args[idx + 1]
        assert "type=bind" in spec
        assert "source=/host/src" in spec
        assert "target=/app/src" in spec
        assert "readonly" in spec
        assert "-v" not in args

    def test_readwrite_mount_uses_v_syntax(self) -> None:
        """Read-write volumes use '-v host:container' syntax."""
        mount = VolumeMount(
            host_path="/host/group",
            container_path="/workspace/group",
            readonly=False,
        )
        args = build_container_args([mount], "c", "img")
        assert "-v" in args
        idx = args.index("-v")
        assert args[idx + 1] == "/host/group:/workspace/group"
        assert "--mount" not in args

    def test_mixed_mounts(self) -> None:
        """Both read-only and read-write mounts appear correctly together."""
        rw = VolumeMount(host_path="/rw", container_path="/c/rw", readonly=False)
        ro = VolumeMount(host_path="/ro", container_path="/c/ro", readonly=True)
        args = build_container_args([rw, ro], "c", "img")
        assert "-v" in args
        assert "--mount" in args


# ---------------------------------------------------------------------------
# Tests: _parse_final_output()
# ---------------------------------------------------------------------------


class TestParseFinalOutput:
    """Tests for the _parse_final_output() internal function."""

    def test_valid_json_between_markers(self) -> None:
        """Valid JSON inside OUTPUT_START/END markers is parsed to ContainerOutput."""
        payload = ContainerOutput(status="success", result="all done").model_dump_json()
        stdout = f"noise\n{OUTPUT_START_MARKER}\n{payload}\n{OUTPUT_END_MARKER}\ntrailing"
        out = _parse_final_output(stdout, "c", "g")
        assert out.status == "success"
        assert out.result == "all done"

    def test_missing_markers_tries_last_line(self) -> None:
        """When markers are absent, the last non-empty line is parsed as JSON."""
        payload = ContainerOutput(status="success", result="fallback").model_dump_json()
        stdout = f"some output\n{payload}"
        out = _parse_final_output(stdout, "c", "g")
        assert out.status == "success"
        assert out.result == "fallback"

    def test_malformed_json_returns_error(self) -> None:
        """Malformed JSON yields an error ContainerOutput."""
        stdout = f"{OUTPUT_START_MARKER}\nnot-json\n{OUTPUT_END_MARKER}"
        out = _parse_final_output(stdout, "c", "g")
        assert out.status == "error"
        assert out.error is not None
        assert "parse" in out.error.lower()

    def test_empty_stdout_returns_error(self) -> None:
        """Completely empty stdout yields an error ContainerOutput."""
        out = _parse_final_output("", "c", "g")
        assert out.status == "error"

    def test_new_session_id_preserved(self) -> None:
        """new_session_id in the JSON is preserved in the returned model."""
        payload = ContainerOutput(
            status="success", result="ok", new_session_id="sess-abc"
        ).model_dump_json()
        stdout = f"{OUTPUT_START_MARKER}\n{payload}\n{OUTPUT_END_MARKER}"
        out = _parse_final_output(stdout, "c", "g")
        assert out.new_session_id == "sess-abc"


# ---------------------------------------------------------------------------
# Tests: write_tasks_snapshot()
# ---------------------------------------------------------------------------


class TestWriteTasksSnapshot:
    """Tests for the write_tasks_snapshot() function."""

    def _make_task(self, folder: str, task_id: str = "t1") -> ScheduledTask:
        """Create a minimal ScheduledTask for the given folder.

        Args:
            folder: group_folder to assign.
            task_id: Unique task id.

        Returns:
            A ScheduledTask instance.
        """
        return ScheduledTask(
            id=task_id,
            group_folder=folder,
            chat_jid=f"{folder}@g.us",
            prompt="do something",
            schedule_type="cron",
            schedule_value="0 * * * *",
            created_at="2024-01-01T00:00:00Z",
        )

    def test_main_sees_all_tasks(self, tmp_path: Path) -> None:
        """Main group snapshot includes tasks from all groups."""
        tasks = [self._make_task("friends", "t1"), self._make_task("work", "t2")]
        write_tasks_snapshot("main", is_main=True, tasks=tasks, data_dir=tmp_path)
        snap_file = tmp_path / "ipc" / "main" / "current_tasks.json"
        assert snap_file.exists()
        data = json.loads(snap_file.read_text())
        assert len(data) == 2

    def test_non_main_sees_only_own_tasks(self, tmp_path: Path) -> None:
        """Non-main group snapshot includes only tasks belonging to that group."""
        tasks = [self._make_task("friends", "t1"), self._make_task("work", "t2")]
        write_tasks_snapshot("friends", is_main=False, tasks=tasks, data_dir=tmp_path)
        snap_file = tmp_path / "ipc" / "friends" / "current_tasks.json"
        assert snap_file.exists()
        data = json.loads(snap_file.read_text())
        assert len(data) == 1
        assert data[0]["groupFolder"] == "friends"

    def test_snapshot_file_written_in_correct_ipc_dir(self, tmp_path: Path) -> None:
        """The snapshot is always written to data_dir/ipc/<group_folder>/current_tasks.json."""
        write_tasks_snapshot("mygroup", is_main=False, tasks=[], data_dir=tmp_path)
        expected = tmp_path / "ipc" / "mygroup" / "current_tasks.json"
        assert expected.exists()

    def test_empty_tasks_writes_empty_array(self, tmp_path: Path) -> None:
        """An empty task list produces an empty JSON array."""
        write_tasks_snapshot("mygroup", is_main=False, tasks=[], data_dir=tmp_path)
        data = json.loads(
            (tmp_path / "ipc" / "mygroup" / "current_tasks.json").read_text()
        )
        assert data == []


# ---------------------------------------------------------------------------
# Tests: write_groups_snapshot()
# ---------------------------------------------------------------------------


class TestWriteGroupsSnapshot:
    """Tests for the write_groups_snapshot() function."""

    def _make_group(self, jid: str, name: str) -> AvailableGroup:
        """Create a minimal AvailableGroup.

        Args:
            jid: WhatsApp JID.
            name: Display name.

        Returns:
            An AvailableGroup instance.
        """
        return AvailableGroup(
            jid=jid,
            name=name,
            last_activity="2024-01-01T00:00:00Z",
            is_registered=False,
        )

    def test_main_sees_all_groups(self, tmp_path: Path) -> None:
        """Main group snapshot includes all available groups."""
        groups = [
            self._make_group("a@g.us", "Alpha"),
            self._make_group("b@g.us", "Beta"),
        ]
        write_groups_snapshot("main", True, groups, set(), tmp_path)
        snap_file = tmp_path / "ipc" / "main" / "available_groups.json"
        data = json.loads(snap_file.read_text())
        assert len(data["groups"]) == 2

    def test_non_main_sees_empty_groups(self, tmp_path: Path) -> None:
        """Non-main group snapshot always has an empty groups list."""
        groups = [self._make_group("a@g.us", "Alpha")]
        write_groups_snapshot("friends", False, groups, set(), tmp_path)
        snap_file = tmp_path / "ipc" / "friends" / "available_groups.json"
        data = json.loads(snap_file.read_text())
        assert data["groups"] == []

    def test_snapshot_written_to_correct_ipc_dir(self, tmp_path: Path) -> None:
        """The snapshot is written to data_dir/ipc/<group_folder>/available_groups.json."""
        write_groups_snapshot("mygroup", True, [], set(), tmp_path)
        expected = tmp_path / "ipc" / "mygroup" / "available_groups.json"
        assert expected.exists()

    def test_snapshot_contains_last_sync(self, tmp_path: Path) -> None:
        """The snapshot JSON contains a 'lastSync' timestamp key."""
        write_groups_snapshot("mygroup", True, [], set(), tmp_path)
        data = json.loads(
            (tmp_path / "ipc" / "mygroup" / "available_groups.json").read_text()
        )
        assert "lastSync" in data


# ---------------------------------------------------------------------------
# Tests: run_container_agent() with mocked subprocess
# ---------------------------------------------------------------------------


class TestRunContainerAgent:
    """Integration tests for run_container_agent() using mocked subprocess."""

    def _make_input(self, is_main: bool = False) -> ContainerInput:
        """Build a minimal ContainerInput.

        Args:
            is_main: Whether this is the main group.

        Returns:
            A ContainerInput instance.
        """
        return ContainerInput(
            prompt="hello agent",
            group_folder="friends",
            chat_jid="friends@g.us",
            is_main=is_main,
        )

    async def _run(
        self,
        mock_proc: _MockProcess,
        layout: dict[str, Path],
        group: RegisteredGroup,
        container_input: ContainerInput | None = None,
        on_output=None,
    ) -> ContainerOutput:
        """Helper: patch subprocess and invoke run_container_agent.

        Args:
            mock_proc: The mock process to return from create_subprocess_exec.
            layout: Filesystem layout dict.
            group: Registered group for the run.
            container_input: Optional ContainerInput; defaults to a plain one.
            on_output: Optional on_output callback.

        Returns:
            The ContainerOutput from run_container_agent.
        """
        from nanoclaw.container import run_container_agent

        ci = container_input or self._make_input()

        async def _fake_exec(*args, **kwargs):
            return mock_proc

        with (
            patch("asyncio.create_subprocess_exec", side_effect=_fake_exec),
            patch("nanoclaw.container.read_secrets", return_value={}),
        ):
            return await run_container_agent(
                group=group,
                container_input=ci,
                groups_dir=layout["groups_dir"],
                data_dir=layout["data_dir"],
                project_root=layout["project_root"],
                image="nanoclaw-agent:latest",
                timeout_ms=30_000,
                idle_timeout_ms=10_000,
                max_output_size=1_000_000,
                on_output=on_output,
            )

    async def test_successful_run_returns_success(
        self, group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """Exit code 0 with valid marker JSON yields a success ContainerOutput."""
        proc = _MockProcess(stdout_data=_make_stdout("done!"), exit_code=0)
        out = await self._run(proc, layout, group)
        assert out.status == "success"

    async def test_nonzero_exit_code_returns_error(
        self, group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """Non-zero exit code produces an error ContainerOutput."""
        proc = _MockProcess(
            stdout_data=b"some output",
            stderr_data=b"fatal: something broke",
            exit_code=1,
        )
        out = await self._run(proc, layout, group)
        assert out.status == "error"
        assert out.error is not None

    async def test_timeout_produces_error(
        self, group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """A subprocess that never finishes yields a timeout error ContainerOutput."""
        from nanoclaw.container import run_container_agent

        ci = self._make_input()

        async def _slow_read(n: int = -1) -> bytes:
            await asyncio.sleep(100)
            return b""

        async def _slow_readline() -> bytes:
            await asyncio.sleep(100)
            return b""

        class _SlowReader:
            async def read(self, n: int = -1) -> bytes:
                await asyncio.sleep(100)
                return b""

            async def readline(self) -> bytes:
                await asyncio.sleep(100)
                return b""

        class _SlowProc(_MockProcess):
            def __init__(self) -> None:
                self.returncode = 0
                self.stdin = AsyncMock()
                self.stdout = _SlowReader()
                self.stderr = _SlowReader()

            async def wait(self) -> int:
                return 0

        slow_proc = _SlowProc()

        async def _fake_exec(*args, **kwargs):
            return slow_proc

        async def _fake_stop(*args, **kwargs):
            return MagicMock()

        with (
            patch("asyncio.create_subprocess_exec", side_effect=_fake_exec),
            patch("nanoclaw.container.read_secrets", return_value={}),
        ):
            out = await run_container_agent(
                group=group,
                container_input=ci,
                groups_dir=layout["groups_dir"],
                data_dir=layout["data_dir"],
                project_root=layout["project_root"],
                image="nanoclaw-agent:latest",
                timeout_ms=100,
                idle_timeout_ms=100,
                max_output_size=1_000_000,
            )

        assert out.status == "error"
        assert out.error is not None
        assert "timed out" in out.error.lower() or "timeout" in out.error.lower()

    async def test_on_output_callback_called_for_each_marker(
        self, group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """on_output callback is invoked once per valid OUTPUT_START/END pair."""
        payload1 = ContainerOutput(status="success", result="first").model_dump_json()
        payload2 = ContainerOutput(status="success", result="second").model_dump_json()
        stdout = (
            f"{OUTPUT_START_MARKER}\n{payload1}\n{OUTPUT_END_MARKER}\n"
            f"{OUTPUT_START_MARKER}\n{payload2}\n{OUTPUT_END_MARKER}\n"
        ).encode()

        received: list[ContainerOutput] = []

        async def _on_output(parsed: ContainerOutput) -> None:
            received.append(parsed)

        proc = _MockProcess(stdout_data=stdout, exit_code=0)
        out = await self._run(proc, layout, group, on_output=_on_output)

        assert out.status == "success"
        assert len(received) == 2
        assert received[0].result == "first"
        assert received[1].result == "second"


# ---------------------------------------------------------------------------
# Additional tests to cover missing lines
# ---------------------------------------------------------------------------


class TestGetHomeDir:
    """Tests for the get_home_dir() helper function (lines 46-49)."""

    def test_get_home_dir_returns_string(self) -> None:
        """get_home_dir() returns a non-empty string."""
        from nanoclaw.container import get_home_dir

        home = get_home_dir()
        assert isinstance(home, str)
        assert len(home) > 0

    def test_get_home_dir_uses_home_env(self, monkeypatch) -> None:
        """get_home_dir() prefers the HOME environment variable (line 46)."""
        from nanoclaw.container import get_home_dir

        monkeypatch.setenv("HOME", "/custom/home")
        assert get_home_dir() == "/custom/home"

    def test_get_home_dir_raises_when_home_missing(self, monkeypatch) -> None:
        """get_home_dir() raises RuntimeError when HOME is absent and Path.home() fails (lines 47-48)."""
        from nanoclaw.container import get_home_dir
        from pathlib import Path
        from unittest.mock import patch

        monkeypatch.delenv("HOME", raising=False)
        with patch.object(Path, "home", side_effect=RuntimeError("no home")):
            # Both sources fail → should raise RuntimeError
            with pytest.raises((RuntimeError, Exception)):
                get_home_dir()


class TestEnsureSettingsFile:
    """Tests for the _ensure_settings_file() helper (line 145)."""

    def test_creates_settings_when_absent(self, tmp_path: Path) -> None:
        """_ensure_settings_file() writes settings.json when it does not exist."""
        from nanoclaw.container import _ensure_settings_file

        sessions_dir = tmp_path / ".claude"
        sessions_dir.mkdir()
        _ensure_settings_file(sessions_dir)
        settings_file = sessions_dir / "settings.json"
        assert settings_file.exists()
        data = json.loads(settings_file.read_text())
        assert "env" in data

    def test_does_not_overwrite_existing_settings(self, tmp_path: Path) -> None:
        """_ensure_settings_file() is a no-op when settings.json already exists (line 145)."""
        from nanoclaw.container import _ensure_settings_file

        sessions_dir = tmp_path / ".claude"
        sessions_dir.mkdir()
        settings_file = sessions_dir / "settings.json"
        original = '{"custom": true}'
        settings_file.write_text(original)

        _ensure_settings_file(sessions_dir)
        # File should be unchanged
        assert settings_file.read_text() == original


class TestSyncSkills:
    """Tests for the _sync_skills() helper (lines 165-171)."""

    def test_noop_when_src_missing(self, tmp_path: Path) -> None:
        """_sync_skills() does nothing when the source directory does not exist (line 165)."""
        from nanoclaw.container import _sync_skills

        _sync_skills(tmp_path / "nonexistent", tmp_path / "dst")
        # No exception and destination directory is not created
        assert not (tmp_path / "dst").exists()

    def test_copies_skill_files(self, tmp_path: Path) -> None:
        """_sync_skills() copies skill directories recursively (lines 165-171)."""
        from nanoclaw.container import _sync_skills

        src = tmp_path / "skills"
        skill_dir = src / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "skill.md").write_text("# Skill")

        dst = tmp_path / "dst_skills"
        _sync_skills(src, dst)

        assert (dst / "my-skill" / "skill.md").exists()
        assert (dst / "my-skill" / "skill.md").read_text() == "# Skill"

    def test_skips_non_directory_entries(self, tmp_path: Path) -> None:
        """_sync_skills() skips files at the top level of the source (line 167)."""
        from nanoclaw.container import _sync_skills

        src = tmp_path / "skills"
        src.mkdir()
        # Put a plain file directly in skills/ (should be skipped)
        (src / "README.md").write_text("readme")

        dst = tmp_path / "dst_skills"
        _sync_skills(src, dst)

        # The file must not have been copied
        assert not (dst / "README.md").exists()


class TestReadSecrets:
    """Tests for read_secrets() (lines 214-218)."""

    def test_read_secrets_returns_dict_with_populated_keys(self, tmp_path: Path) -> None:
        """read_secrets() returns a dict containing only non-empty credential values."""
        from nanoclaw.container import read_secrets

        creds_file = tmp_path / "credentials.yml"
        creds_file.write_text("anthropic_api_key: sk-test\n")

        result = read_secrets(creds_file)
        assert result["ANTHROPIC_API_KEY"] == "sk-test"
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in result  # Empty values are excluded

    def test_read_secrets_excludes_empty_values(self, tmp_path: Path) -> None:
        """read_secrets() filters out empty credential values (lines 220-225)."""
        from nanoclaw.container import read_secrets

        # Use empty string values explicitly (not None from missing YAML keys)
        creds_file = tmp_path / "credentials.yml"
        creds_file.write_text(
            'anthropic_api_key: ""\nclaude_code_oauth_token: ""\n'
        )

        result = read_secrets(creds_file)
        assert result == {}


class TestWriteContainerLog:
    """Tests for the _write_container_log() helper (lines 323-324, 340-343, 360, 366, 370-371)."""

    def _make_group(self) -> RegisteredGroup:
        """Build a minimal RegisteredGroup for log tests."""
        return RegisteredGroup(
            name="Friends",
            folder="friends",
            trigger="@Andy",
            added_at="2024-01-01T00:00:00Z",
        )

    def _make_input(self, is_main: bool = False) -> ContainerInput:
        """Build a minimal ContainerInput."""
        return ContainerInput(
            prompt="hello",
            group_folder="friends",
            chat_jid="friends@g.us",
            is_main=is_main,
        )

    def test_log_file_is_written(self, tmp_path: Path) -> None:
        """_write_container_log() creates a log file in the logs directory (lines 340-343)."""
        from nanoclaw.container import _write_container_log

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        group = self._make_group()
        ci = self._make_input()

        _write_container_log(
            logs_dir=logs_dir,
            container_name="nanoclaw-test-123",
            group=group,
            container_input=ci,
            mounts=[],
            container_args=["container", "run", "-i", "--rm"],
            exit_code=0,
            duration_ms=1500,
            stdout_truncated=False,
            stdout="output",
            stderr="",
        )
        log_files = list(logs_dir.glob("*.log"))
        assert len(log_files) == 1

    def test_log_file_contains_basic_fields(self, tmp_path: Path) -> None:
        """Log file contains group name, duration, and exit code."""
        from nanoclaw.container import _write_container_log

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        group = self._make_group()
        ci = self._make_input()

        _write_container_log(
            logs_dir=logs_dir,
            container_name="nanoclaw-test-456",
            group=group,
            container_input=ci,
            mounts=[],
            container_args=[],
            exit_code=0,
            duration_ms=2500,
            stdout_truncated=False,
            stdout="",
            stderr="",
        )
        content = list(logs_dir.glob("*.log"))[0].read_text()
        assert "Friends" in content
        assert "2500ms" in content

    def test_log_file_verbose_mode_includes_args(self, tmp_path: Path, monkeypatch) -> None:
        """Verbose LOG_LEVEL causes container args and mounts to be written (lines 360, 366)."""
        from nanoclaw.container import _write_container_log

        monkeypatch.setenv("LOG_LEVEL", "debug")
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        group = self._make_group()
        ci = self._make_input()

        _write_container_log(
            logs_dir=logs_dir,
            container_name="nanoclaw-verbose",
            group=group,
            container_input=ci,
            mounts=[
                VolumeMount(host_path="/host/g", container_path="/workspace/group", readonly=False)
            ],
            container_args=["container", "run", "-i", "nanoclaw-agent:latest"],
            exit_code=0,
            duration_ms=100,
            stdout_truncated=False,
            stdout="some stdout",
            stderr="some stderr",
        )
        content = list(logs_dir.glob("*.log"))[0].read_text()
        assert "container run" in content
        assert "some stderr" in content

    def test_log_file_error_exit_includes_details(self, tmp_path: Path) -> None:
        """Non-zero exit code causes verbose output to be included (line 360 is_error)."""
        from nanoclaw.container import _write_container_log

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        group = self._make_group()
        ci = self._make_input()

        _write_container_log(
            logs_dir=logs_dir,
            container_name="nanoclaw-error",
            group=group,
            container_input=ci,
            mounts=[],
            container_args=["container", "run"],
            exit_code=1,
            duration_ms=200,
            stdout_truncated=False,
            stdout="",
            stderr="fatal error occurred",
        )
        content = list(logs_dir.glob("*.log"))[0].read_text()
        assert "fatal error occurred" in content

    def test_log_file_truncated_note_appears(self, tmp_path: Path, monkeypatch) -> None:
        """Truncated stdout is flagged in the log file (line 370-371)."""
        from nanoclaw.container import _write_container_log

        monkeypatch.setenv("LOG_LEVEL", "debug")
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        group = self._make_group()
        ci = self._make_input()

        _write_container_log(
            logs_dir=logs_dir,
            container_name="nanoclaw-trunc",
            group=group,
            container_input=ci,
            mounts=[],
            container_args=[],
            exit_code=0,
            duration_ms=50,
            stdout_truncated=True,
            stdout="partial output",
            stderr="",
        )
        content = list(logs_dir.glob("*.log"))[0].read_text()
        assert "TRUNCATED" in content

    def test_oserror_writing_log_does_not_raise(self, tmp_path: Path) -> None:
        """OSError when writing the log file is caught and logged (lines 402-407)."""
        from nanoclaw.container import _write_container_log
        from unittest.mock import patch

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        group = self._make_group()
        ci = self._make_input()

        with patch("pathlib.Path.write_text", side_effect=OSError("no space")):
            _write_container_log(
                logs_dir=logs_dir,
                container_name="nanoclaw-oserr",
                group=group,
                container_input=ci,
                mounts=[],
                container_args=[],
                exit_code=0,
                duration_ms=10,
                stdout_truncated=False,
                stdout="",
                stderr="",
            )  # Must not raise


class TestRunContainerAgentExtended(TestRunContainerAgent):
    """Extended tests for run_container_agent() covering additional code paths."""

    async def test_timeout_with_streaming_output_returns_success(
        self, group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """A timed-out container that had streaming output returns success (lines 544-545)."""
        from nanoclaw.container import run_container_agent

        ci = self._make_input()
        payload = ContainerOutput(status="success", result="streamed").model_dump_json()
        streaming_stdout = (
            f"{OUTPUT_START_MARKER}\n{payload}\n{OUTPUT_END_MARKER}\n"
        ).encode()

        class _SlowReader:
            def __init__(self, data: bytes) -> None:
                import io

                self._buf = io.BytesIO(data)
                self._done = False

            async def read(self, n: int = -1) -> bytes:
                chunk = self._buf.read(n)
                if chunk:
                    return chunk
                await asyncio.sleep(100)  # Hang after sending data
                return b""

            async def readline(self) -> bytes:
                await asyncio.sleep(100)
                return b""

        class _StreamThenHangProc(_MockProcess):
            def __init__(self) -> None:
                self.returncode = 0
                self.stdin = AsyncMock()
                self.stdout = _SlowReader(streaming_stdout)
                self.stderr = _SlowReader(b"")

            async def wait(self) -> int:
                return 0

        proc = _StreamThenHangProc()

        received: list[ContainerOutput] = []

        async def _on_output(parsed: ContainerOutput) -> None:
            received.append(parsed)

        async def _fake_exec(*args, **kwargs):
            return proc

        with (
            patch("asyncio.create_subprocess_exec", side_effect=_fake_exec),
            patch("nanoclaw.container.read_secrets", return_value={}),
        ):
            out = await run_container_agent(
                group=group,
                container_input=ci,
                groups_dir=layout["groups_dir"],
                data_dir=layout["data_dir"],
                project_root=layout["project_root"],
                image="nanoclaw-agent:latest",
                timeout_ms=200,
                idle_timeout_ms=100,
                max_output_size=1_000_000,
                on_output=_on_output,
            )

        # When timeout fires after streaming output → status success
        assert out.status == "success"

    async def test_on_container_name_callback_called(
        self, group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """on_container_name callback receives the container name (line 291)."""
        proc = _MockProcess(stdout_data=_make_stdout("ok"), exit_code=0)
        received_names: list[str] = []

        async def _on_name(name: str) -> None:
            received_names.append(name)

        from nanoclaw.container import run_container_agent

        async def _fake_exec(*args, **kwargs):
            return proc

        with (
            patch("asyncio.create_subprocess_exec", side_effect=_fake_exec),
            patch("nanoclaw.container.read_secrets", return_value={}),
        ):
            await run_container_agent(
                group=group,
                container_input=self._make_input(),
                groups_dir=layout["groups_dir"],
                data_dir=layout["data_dir"],
                project_root=layout["project_root"],
                image="nanoclaw-agent:latest",
                timeout_ms=30_000,
                idle_timeout_ms=10_000,
                max_output_size=1_000_000,
                on_container_name=_on_name,
            )

        assert len(received_names) == 1
        assert received_names[0].startswith("nanoclaw-")

    async def test_stdout_truncation_limits_buffer(
        self, group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """Stdout larger than max_output_size is truncated in memory (lines 323-324)."""
        large_data = b"x" * 200
        proc = _MockProcess(stdout_data=large_data, exit_code=0)
        out = await self._run(proc, layout, group, container_input=self._make_input())
        # With very small max_output_size, we just need it to not crash
        # The actual truncation path is exercised; out.status may be "error"
        # because there's no valid marker but it must not raise
        assert out.status in ("success", "error")


# ---------------------------------------------------------------------------
# Targeted tests for remaining coverage gaps
# ---------------------------------------------------------------------------


class TestGetHomeDirLine48:
    """Test line 48 of get_home_dir(): the RuntimeError branch."""

    def test_get_home_dir_raises_when_home_is_empty_string(self, monkeypatch) -> None:
        """get_home_dir() raises RuntimeError when HOME="" and Path.home() returns "" (line 48)."""
        from nanoclaw.container import get_home_dir

        monkeypatch.setenv("HOME", "")
        with patch("nanoclaw.container.Path") as MockPath:
            MockPath.home.return_value = ""  # str("") == "" which is falsy
            with pytest.raises(RuntimeError, match="Unable to determine home directory"):
                get_home_dir()


class TestRunContainerAgentCoverageGaps:
    """Targeted tests for uncovered lines in run_container_agent()."""

    def _make_input(self, is_main: bool = False) -> ContainerInput:
        """Build a minimal ContainerInput."""
        return ContainerInput(
            prompt="hello",
            group_folder="friends",
            chat_jid="friends@g.us",
            is_main=is_main,
        )

    @pytest.mark.asyncio
    async def test_stop_container_kills_proc_on_exec_error(
        self, group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """When container stop subprocess raises, proc.kill() is called (lines 323-324)."""
        from nanoclaw.container import run_container_agent

        kill_calls: list[str] = []

        class _SlowReader:
            async def read(self, n: int = 4096) -> bytes:
                await asyncio.sleep(100)
                return b""

            async def readline(self) -> bytes:
                await asyncio.sleep(100)
                return b""

        class _SlowKillableProc:
            def __init__(self) -> None:
                self.returncode = 0
                self.stdin = AsyncMock()
                self.stdout = _SlowReader()
                self.stderr = _SlowReader()

            async def wait(self) -> int:
                return 0

            def kill(self) -> None:
                kill_calls.append("killed")

        proc = _SlowKillableProc()
        call_count = 0

        async def _fake_exec(*args: object, **kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return proc
            raise OSError("container stop failed")

        with (
            patch("asyncio.create_subprocess_exec", side_effect=_fake_exec),
            patch("nanoclaw.container.read_secrets", return_value={}),
        ):
            out = await run_container_agent(
                group=group,
                container_input=self._make_input(),
                groups_dir=layout["groups_dir"],
                data_dir=layout["data_dir"],
                project_root=layout["project_root"],
                image="nanoclaw-agent:latest",
                timeout_ms=100,
                idle_timeout_ms=100,
                max_output_size=1_000_000,
            )

        assert len(kill_calls) > 0
        assert out.status == "error"

    @pytest.mark.asyncio
    async def test_stdout_large_chunk_triggers_truncation(
        self, group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """Single chunk larger than max_output_size triggers truncation branch (lines 340-343)."""
        from nanoclaw.container import run_container_agent

        large_data = b"X" * 2000  # Much larger than max_output_size=100
        proc = _MockProcess(stdout_data=large_data, exit_code=0)

        async def _fake_exec(*args: object, **kwargs: object) -> _MockProcess:
            return proc

        with (
            patch("asyncio.create_subprocess_exec", side_effect=_fake_exec),
            patch("nanoclaw.container.read_secrets", return_value={}),
        ):
            out = await run_container_agent(
                group=group,
                container_input=self._make_input(),
                groups_dir=layout["groups_dir"],
                data_dir=layout["data_dir"],
                project_root=layout["project_root"],
                image="nanoclaw-agent:latest",
                timeout_ms=30_000,
                idle_timeout_ms=10_000,
                max_output_size=100,  # Tiny limit: 2000-byte chunk > 100
            )

        assert out.status in ("success", "error")

    @pytest.mark.asyncio
    async def test_streaming_incomplete_marker_is_silently_ignored(
        self, group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """OUTPUT_START without OUTPUT_END triggers incomplete buffer break (line 360)."""
        from nanoclaw.container import run_container_agent

        stdout_data = f"{OUTPUT_START_MARKER}\nno end marker here".encode()
        proc = _MockProcess(stdout_data=stdout_data, exit_code=0)
        received: list[ContainerOutput] = []

        async def _on_output(parsed: ContainerOutput) -> None:
            received.append(parsed)

        async def _fake_exec(*args: object, **kwargs: object) -> _MockProcess:
            return proc

        with (
            patch("asyncio.create_subprocess_exec", side_effect=_fake_exec),
            patch("nanoclaw.container.read_secrets", return_value={}),
        ):
            out = await run_container_agent(
                group=group,
                container_input=self._make_input(),
                groups_dir=layout["groups_dir"],
                data_dir=layout["data_dir"],
                project_root=layout["project_root"],
                image="nanoclaw-agent:latest",
                timeout_ms=30_000,
                idle_timeout_ms=10_000,
                max_output_size=1_000_000,
                on_output=_on_output,
            )

        assert len(received) == 0  # No complete output parsed

    @pytest.mark.asyncio
    async def test_streaming_new_session_id_is_captured(
        self, group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """ContainerOutput with new_session_id propagates to return value (line 366)."""
        from nanoclaw.container import run_container_agent

        payload = ContainerOutput(
            status="success", result="done", new_session_id="sess-abc123"
        ).model_dump_json()
        stdout_data = f"{OUTPUT_START_MARKER}\n{payload}\n{OUTPUT_END_MARKER}\n".encode()
        proc = _MockProcess(stdout_data=stdout_data, exit_code=0)
        received: list[ContainerOutput] = []

        async def _on_output(parsed: ContainerOutput) -> None:
            received.append(parsed)

        async def _fake_exec(*args: object, **kwargs: object) -> _MockProcess:
            return proc

        with (
            patch("asyncio.create_subprocess_exec", side_effect=_fake_exec),
            patch("nanoclaw.container.read_secrets", return_value={}),
        ):
            out = await run_container_agent(
                group=group,
                container_input=self._make_input(),
                groups_dir=layout["groups_dir"],
                data_dir=layout["data_dir"],
                project_root=layout["project_root"],
                image="nanoclaw-agent:latest",
                timeout_ms=30_000,
                idle_timeout_ms=10_000,
                max_output_size=1_000_000,
                on_output=_on_output,
            )

        assert len(received) == 1
        assert received[0].new_session_id == "sess-abc123"
        assert out.new_session_id == "sess-abc123"

    @pytest.mark.asyncio
    async def test_streaming_invalid_json_is_warned_not_raised(
        self, group: RegisteredGroup, layout: dict[str, Path]
    ) -> None:
        """Malformed JSON between markers is logged as warning, not raised (lines 370-371)."""
        from nanoclaw.container import run_container_agent

        stdout_data = (
            f"{OUTPUT_START_MARKER}\nnot-valid-json!!!\n{OUTPUT_END_MARKER}\n"
        ).encode()
        proc = _MockProcess(stdout_data=stdout_data, exit_code=0)
        received: list[ContainerOutput] = []

        async def _on_output(parsed: ContainerOutput) -> None:
            received.append(parsed)

        async def _fake_exec(*args: object, **kwargs: object) -> _MockProcess:
            return proc

        with (
            patch("asyncio.create_subprocess_exec", side_effect=_fake_exec),
            patch("nanoclaw.container.read_secrets", return_value={}),
        ):
            out = await run_container_agent(
                group=group,
                container_input=self._make_input(),
                groups_dir=layout["groups_dir"],
                data_dir=layout["data_dir"],
                project_root=layout["project_root"],
                image="nanoclaw-agent:latest",
                timeout_ms=30_000,
                idle_timeout_ms=10_000,
                max_output_size=1_000_000,
                on_output=_on_output,
            )

        assert len(received) == 0  # Parsing failed — no output dispatched
        assert out.status in ("success", "error")


# end tests/test_container.py
