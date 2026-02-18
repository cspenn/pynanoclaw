# start src/nanoclaw/container.py
"""Apple Container subprocess management for NanoClaw.

Replaces src/container-runner.ts from TypeScript.
Uses asyncio.create_subprocess_exec to spawn agent containers.
Parses streaming output via OUTPUT_START/END sentinel markers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from nanoclaw.types import (
    AvailableGroup,
    ContainerInput,
    ContainerOutput,
    RegisteredGroup,
    ScheduledTask,
    VolumeMount,
)

logger = logging.getLogger(__name__)

OUTPUT_START_MARKER = "---NANOCLAW_OUTPUT_START---"
OUTPUT_END_MARKER = "---NANOCLAW_OUTPUT_END---"


def get_home_dir() -> str:
    """Return the user's home directory.

    Returns:
        Absolute path to the home directory.

    Raises:
        RuntimeError: If the home directory cannot be determined.
    """
    home = os.environ.get("HOME") or str(Path.home())
    if not home:
        raise RuntimeError("Unable to determine home directory")
    return home


def build_volume_mounts(
    group: RegisteredGroup,
    is_main: bool,
    groups_dir: Path,
    data_dir: Path,
    project_root: Path,
) -> list[VolumeMount]:
    """Build the list of volume mounts for an agent container.

    Args:
        group: The registered group whose container is being spawned.
        is_main: Whether this is the main group (gets full project access).
        groups_dir: Absolute path to the groups/ directory.
        data_dir: Absolute path to the data/ directory.
        project_root: Absolute path to the project root.

    Returns:
        List of VolumeMount objects for the container invocation.
    """
    mounts: list[VolumeMount] = []

    if is_main:
        mounts.append(
            VolumeMount(
                host_path=str(project_root),
                container_path="/workspace/project",
                readonly=False,
            )
        )

    mounts.append(
        VolumeMount(
            host_path=str(groups_dir / group.folder),
            container_path="/workspace/group",
            readonly=False,
        )
    )

    if not is_main:
        global_dir = groups_dir / "global"
        if global_dir.exists():
            mounts.append(
                VolumeMount(
                    host_path=str(global_dir),
                    container_path="/workspace/global",
                    readonly=True,
                )
            )

    group_sessions_dir = data_dir / "sessions" / group.folder / ".claude"
    group_sessions_dir.mkdir(parents=True, exist_ok=True)
    _ensure_settings_file(group_sessions_dir)
    _sync_skills(project_root / "container" / "skills", group_sessions_dir / "skills")
    mounts.append(
        VolumeMount(
            host_path=str(group_sessions_dir),
            container_path="/home/node/.claude",
            readonly=False,
        )
    )

    group_ipc_dir = data_dir / "ipc" / group.folder
    (group_ipc_dir / "messages").mkdir(parents=True, exist_ok=True)
    (group_ipc_dir / "tasks").mkdir(parents=True, exist_ok=True)
    (group_ipc_dir / "input").mkdir(parents=True, exist_ok=True)
    mounts.append(
        VolumeMount(
            host_path=str(group_ipc_dir),
            container_path="/workspace/ipc",
            readonly=False,
        )
    )

    agent_runner_src = project_root / "container" / "agent-runner" / "src"
    mounts.append(
        VolumeMount(
            host_path=str(agent_runner_src),
            container_path="/app/src",
            readonly=True,
        )
    )

    return mounts


def _ensure_settings_file(sessions_dir: Path) -> None:
    """Create the Claude settings.json in the session directory if absent.

    Args:
        sessions_dir: Path to the .claude/ directory inside the session store.
    """
    settings_file = sessions_dir / "settings.json"
    if settings_file.exists():
        return
    settings = {
        "env": {
            "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1",
            "CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD": "1",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "0",
        }
    }
    settings_file.write_text(json.dumps(settings, indent=2) + "\n")


def _sync_skills(skills_src: Path, skills_dst: Path) -> None:
    """Sync skills from container/skills/ into the group's .claude/skills/.

    Args:
        skills_src: Source skills directory (container/skills/).
        skills_dst: Destination skills directory (data/sessions/{group}/.claude/skills/).
    """
    if not skills_src.exists():
        return
    for skill_dir in skills_src.iterdir():
        if not skill_dir.is_dir():
            continue
        dst_dir = skills_dst / skill_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for skill_file in skill_dir.iterdir():
            shutil.copy2(str(skill_file), str(dst_dir / skill_file.name))


def build_container_args(
    mounts: list[VolumeMount],
    container_name: str,
    image: str,
) -> list[str]:
    """Build the argument list for the Apple Container CLI invocation.

    Args:
        mounts: Volume mounts to include.
        container_name: Name to assign to the container.
        image: Container image name.

    Returns:
        List of arguments for the 'container' CLI command.
    """
    args = ["container", "run", "-i", "--rm", "--name", container_name]
    for mount in mounts:
        if mount.readonly:
            args += [
                "--mount",
                f"type=bind,source={mount.host_path},target={mount.container_path},readonly",
            ]
        else:
            args += ["-v", f"{mount.host_path}:{mount.container_path}"]
    args.append(image)
    return args


def read_secrets(env_file: Path | None = None) -> dict[str, str]:
    """Read API secrets from credentials.yml for passing to containers via stdin.

    Secrets are NEVER written to disk or mounted as files.

    Args:
        env_file: Optional path to credentials.yml. Defaults to credentials.yml
            in the current directory.

    Returns:
        Dict of secret name -> value.
    """
    from nanoclaw.config import load_credentials

    creds_path = env_file or Path("credentials.yml")
    creds = load_credentials(creds_path)
    return {
        k: v
        for k, v in {
            "ANTHROPIC_API_KEY": creds.anthropic_api_key,
            "CLAUDE_CODE_OAUTH_TOKEN": creds.claude_code_oauth_token,
        }.items()
        if v
    }


async def run_container_agent(
    group: RegisteredGroup,
    container_input: ContainerInput,
    groups_dir: Path,
    data_dir: Path,
    project_root: Path,
    image: str,
    timeout_ms: int,
    idle_timeout_ms: int,
    max_output_size: int,
    on_container_name: Callable[[str], Coroutine[Any, Any, None]] | None = None,
    on_output: Callable[[ContainerOutput], Coroutine[Any, Any, None]] | None = None,
) -> ContainerOutput:
    """Spawn an Apple Container agent and process its output.

    Args:
        group: The registered group whose agent to run.
        container_input: Input data for the agent (prompt, session ID, etc.).
        groups_dir: Absolute path to groups/ directory.
        data_dir: Absolute path to data/ directory.
        project_root: Absolute path to project root.
        image: Container image name.
        timeout_ms: Hard timeout in milliseconds.
        idle_timeout_ms: Idle timeout (reset on each output) in milliseconds.
        max_output_size: Maximum bytes to buffer from stdout/stderr.
        on_container_name: Optional async callback receiving (container_name: str).
        on_output: Optional async callback for each parsed ContainerOutput.

    Returns:
        ContainerOutput with the final status and result.
    """
    start_time = time.monotonic()
    group_dir = groups_dir / group.folder
    group_dir.mkdir(parents=True, exist_ok=True)

    mounts = build_volume_mounts(group, container_input.is_main, groups_dir, data_dir, project_root)
    safe_name = re.sub(r"[^a-zA-Z0-9-]", "-", group.folder)
    container_name = f"nanoclaw-{safe_name}-{int(time.time() * 1000)}"
    container_args = build_container_args(mounts, container_name, image)

    logs_dir = group_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Spawning container %s for group %s (%d mounts)",
        container_name,
        group.name,
        len(mounts),
    )

    # Inject secrets via stdin — never written to disk
    secrets = read_secrets()
    container_input_with_secrets = container_input.model_copy(update={"secrets": secrets})
    stdin_data = container_input_with_secrets.model_dump_json().encode()

    proc = await asyncio.create_subprocess_exec(
        *container_args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    if on_container_name:
        await on_container_name(container_name)

    if proc.stdin is not None:
        proc.stdin.write(stdin_data)
        await proc.stdin.drain()
        proc.stdin.close()

    stdout_chunks: list[str] = []
    stderr_lines: list[str] = []
    stdout_total = 0
    stdout_truncated = False
    new_session_id: str | None = None
    had_streaming_output = False
    timed_out = False

    timeout_s = max(timeout_ms, idle_timeout_ms + 30_000) / 1000.0
    parse_buffer = ""
    output_tasks: list[asyncio.Task[None]] = []

    async def _stop_container() -> None:
        """Gracefully stop the container via the Apple Container CLI."""
        nonlocal timed_out
        timed_out = True
        logger.error("Container %s timed out, stopping gracefully", container_name)
        try:
            await asyncio.create_subprocess_exec(
                "container",
                "stop",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except Exception:
            proc.kill()

    async def _read_stdout() -> None:
        """Read stdout, buffer it, and parse streaming output markers."""
        nonlocal parse_buffer, new_session_id, had_streaming_output
        nonlocal stdout_total, stdout_truncated
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace")

            if not stdout_truncated:
                remaining = max_output_size - stdout_total
                if len(text) > remaining:
                    stdout_chunks.append(text[:remaining])
                    stdout_total += remaining
                    stdout_truncated = True
                    logger.warning(
                        "Container %s stdout truncated at %d bytes",
                        container_name,
                        stdout_total,
                    )
                else:
                    stdout_chunks.append(text)
                    stdout_total += len(text)

            if on_output:
                parse_buffer += text
                while True:
                    start = parse_buffer.find(OUTPUT_START_MARKER)
                    if start == -1:
                        break
                    end = parse_buffer.find(OUTPUT_END_MARKER, start)
                    if end == -1:
                        break
                    json_str = parse_buffer[start + len(OUTPUT_START_MARKER) : end].strip()
                    parse_buffer = parse_buffer[end + len(OUTPUT_END_MARKER) :]
                    try:
                        parsed = ContainerOutput.model_validate_json(json_str)
                        if parsed.new_session_id:
                            new_session_id = parsed.new_session_id
                        had_streaming_output = True
                        task = asyncio.create_task(on_output(parsed))
                        output_tasks.append(task)
                    except Exception as exc:
                        logger.warning("Failed to parse streamed output: %s", exc)

    async def _read_stderr() -> None:
        """Read and log stderr lines."""
        assert proc.stderr is not None
        while True:
            line_bytes = await proc.stderr.readline()
            if not line_bytes:
                break
            line = line_bytes.decode(errors="replace").rstrip()
            if line:
                stderr_lines.append(line)
                logger.debug("[%s] %s", group.folder, line)

    try:
        await asyncio.wait_for(
            asyncio.gather(_read_stdout(), _read_stderr()),
            timeout=timeout_s,
        )
    except TimeoutError:
        await _stop_container()

    await proc.wait()
    duration_ms = int((time.monotonic() - start_time) * 1000)

    # Wait for all on_output callbacks to complete
    if output_tasks:
        await asyncio.gather(*output_tasks, return_exceptions=True)

    if timed_out:
        if had_streaming_output:
            logger.info(
                "Container %s timed out after output (idle cleanup, %dms)",
                container_name,
                duration_ms,
            )
            return ContainerOutput(status="success", result=None, new_session_id=new_session_id)
        logger.error("Container %s timed out with no output (%dms)", container_name, duration_ms)
        return ContainerOutput(
            status="error",
            result=None,
            error=f"Container timed out after {timeout_ms}ms",
        )

    exit_code = proc.returncode or 0
    _write_container_log(
        logs_dir,
        container_name,
        group,
        container_input,
        mounts,
        container_args,
        exit_code,
        duration_ms,
        stdout_truncated,
        "".join(stdout_chunks),
        "\n".join(stderr_lines),
    )

    if exit_code != 0:
        stderr_tail = "\n".join(stderr_lines[-10:])
        logger.error(
            "Container %s exited with code %d after %dms",
            container_name,
            exit_code,
            duration_ms,
        )
        return ContainerOutput(
            status="error",
            result=None,
            error=f"Container exited with code {exit_code}: {stderr_tail[-200:]}",
        )

    if on_output:
        logger.info("Container %s completed (streaming, %dms)", container_name, duration_ms)
        return ContainerOutput(status="success", result=None, new_session_id=new_session_id)

    # Non-streaming: parse the last marker pair from accumulated stdout
    stdout_all = "".join(stdout_chunks)
    return _parse_final_output(stdout_all, container_name, group.name)


def _parse_final_output(stdout: str, container_name: str, _group_name: str) -> ContainerOutput:
    """Parse the final ContainerOutput from accumulated stdout.

    Args:
        stdout: Full stdout text from the container.
        container_name: Container name for logging.
        group_name: Group name for logging.

    Returns:
        Parsed ContainerOutput, or an error output if parsing fails.
    """
    start = stdout.find(OUTPUT_START_MARKER)
    end = stdout.find(OUTPUT_END_MARKER)
    if start != -1 and end != -1 and end > start:
        json_str = stdout[start + len(OUTPUT_START_MARKER) : end].strip()
    else:
        lines = [line for line in stdout.strip().split("\n") if line.strip()]
        json_str = lines[-1] if lines else ""

    try:
        return ContainerOutput.model_validate_json(json_str)
    except Exception as exc:
        logger.error("Failed to parse output from container %s: %s", container_name, exc)
        return ContainerOutput(
            status="error",
            result=None,
            error=f"Failed to parse container output: {exc}",
        )


def _write_container_log(
    logs_dir: Path,
    container_name: str,
    group: RegisteredGroup,
    container_input: ContainerInput,
    mounts: list[VolumeMount],
    container_args: list[str],
    exit_code: int,
    duration_ms: int,
    stdout_truncated: bool,
    stdout: str,
    stderr: str,
) -> None:
    """Write a container run log file to the group's logs directory.

    Args:
        logs_dir: Directory to write the log file into.
        container_name: Name of the container that ran.
        group: The registered group.
        container_input: Input that was passed to the container.
        mounts: Volume mounts that were applied.
        container_args: Full container CLI arguments.
        exit_code: Process exit code.
        duration_ms: Run duration in milliseconds.
        stdout_truncated: Whether stdout was truncated.
        stdout: Container stdout text.
        stderr: Container stderr text.
    """
    ts = time.strftime("%Y%m%dT%H%M%S")
    log_file = logs_dir / f"container-{ts}.log"
    is_verbose = os.environ.get("LOG_LEVEL", "").lower() in ("debug", "trace")
    is_error = exit_code != 0
    lines = [
        "=== Container Run Log ===",
        f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"Group: {group.name}",
        f"IsMain: {container_input.is_main}",
        f"Duration: {duration_ms}ms",
        f"Exit Code: {exit_code}",
        f"Stdout Truncated: {stdout_truncated}",
        "",
    ]
    if is_verbose or is_error:
        lines += [
            "=== Container Args ===",
            " ".join(container_args),
            "",
            "=== Mounts ===",
            "\n".join(
                f"{m.host_path} -> {m.container_path}{'(ro)' if m.readonly else ''}" for m in mounts
            ),
            "",
            "=== Stderr ===",
            stderr,
            "",
            f"=== Stdout {'(TRUNCATED)' if stdout_truncated else ''} ===",
            stdout,
        ]
    try:
        log_file.write_text("\n".join(lines))
        logger.debug("Container log written to %s", log_file)
    except OSError as exc:
        logger.warning("Failed to write container log: %s", exc)


def write_tasks_snapshot(
    group_folder: str,
    is_main: bool,
    tasks: list[ScheduledTask],
    data_dir: Path,
) -> None:
    """Write a tasks snapshot JSON file for the container to read.

    Args:
        group_folder: The group folder name.
        is_main: Whether this is the main group (sees all tasks).
        tasks: All scheduled tasks.
        data_dir: Root data directory.
    """
    group_ipc_dir = data_dir / "ipc" / group_folder
    group_ipc_dir.mkdir(parents=True, exist_ok=True)
    filtered = tasks if is_main else [t for t in tasks if t.group_folder == group_folder]
    snapshot = [
        {
            "id": t.id,
            "groupFolder": t.group_folder,
            "prompt": t.prompt,
            "schedule_type": t.schedule_type,
            "schedule_value": t.schedule_value,
            "status": t.status,
            "next_run": t.next_run,
        }
        for t in filtered
    ]
    (group_ipc_dir / "current_tasks.json").write_text(json.dumps(snapshot, indent=2))


def write_groups_snapshot(
    group_folder: str,
    is_main: bool,
    groups: list[AvailableGroup],
    registered_jids: set[str],
    data_dir: Path,
) -> None:
    """Write an available groups snapshot JSON file for the container to read.

    Args:
        group_folder: The group folder name.
        is_main: Whether this is the main group (sees all groups).
        groups: All available WhatsApp groups.
        registered_jids: Set of JIDs already registered with NanoClaw.
        data_dir: Root data directory.
    """
    group_ipc_dir = data_dir / "ipc" / group_folder
    group_ipc_dir.mkdir(parents=True, exist_ok=True)
    visible = groups if is_main else []
    snapshot = {
        "groups": [g.model_dump() for g in visible],
        "lastSync": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (group_ipc_dir / "available_groups.json").write_text(json.dumps(snapshot, indent=2))


# end src/nanoclaw/container.py
