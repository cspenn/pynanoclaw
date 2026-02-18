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
# end tests/test_container.py
